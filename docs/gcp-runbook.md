# GCP spot run runbook (tinyLLM v2)

Layout: a size-1 managed instance group (MIG) built from a spot instance template. After a
preemption the MIG recreates the VM, `startup.sh` runs again, and training resumes by itself
(`resolve_resume`) from the checkpoint directory `/mnt/ckpt/<tag>` (gcsfuse over the bucket).
Data is copied from GCS to the VM's local disk at boot (random reads over gcsfuse are too slow).
Because the template uses the DELETE termination action, a recreated VM has a fresh disk and
re-copies the data (80GB+) before it trains; expect several minutes of lost time per preemption.

All scripts read `scripts/gcp/env` (copy from `env.example`). No script contains a secret.

## 0. Before ANY run: you must confirm each item yourself

Do not start until you have confirmed all of these. Nothing below does it for you.

- [ ] Paid account upgrade is done (free-trial accounts cannot use GPUs).
- [ ] GPU quotas approved in your region: H100 (`a3-highgpu-1g`, spot) and/or L4 (`g2-standard-8`, spot). Preemptible/spot GPU quota is separate from on-demand quota.
- [ ] A budget alert exists. A budget alert only notifies; it does not stop spending. Nothing caps the bill automatically.
- [ ] You have looked up the current per-hour price of the chosen machine type, boot disk (300GB pd-ssd) and bucket storage, and multiplied by the expected hours.
- [ ] You understand that once free credits run out, real charges start on your card.
- [ ] W&B API key is stored in Secret Manager (see below). HF key only if needed.

## 1. One-time setup (you run these; they create billable resources)

Prerequisites, in order:

1. Push the branch so the VM can clone it. The repo is public, so a plain HTTPS clone works
   once the branch exists on GitHub: `git push -u origin feat/tinyllm-v2`. The VM checks out
   `BRANCH` from `env`, so that name must match.
2. Create the env file first (`scripts/gcp/.gitignore` keeps it out of git; it holds no secrets),
   then edit it: `cp scripts/gcp/env.example scripts/gcp/env`
3. Enable the APIs once per project (this sources the env file from step 2):
   ```bash
   source scripts/gcp/env
   gcloud services enable compute.googleapis.com secretmanager.googleapis.com storage.googleapis.com --project "$PROJECT"
   ```
4. Create the bucket and the secret (every command needs `--project`):
   ```bash
   source scripts/gcp/env
   gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION"
   # type the key at the hidden prompt; never put it on a command line
   read -rs WANDB_API_KEY && printf %s "$WANDB_API_KEY" | gcloud secrets create "$SECRET_NAME" --project "$PROJECT" --data-file=- ; unset WANDB_API_KEY
   ```
5. The VM's service account (default: the Compute Engine default account, or `SERVICE_ACCOUNT`) needs:
   - `roles/storage.objectAdmin` on the bucket (data, checkpoints, status markers),
   - `roles/secretmanager.secretAccessor` on the secret,
   - `roles/compute.instanceAdmin.v1` on the project, so the VM can resize its own MIG to 0
     (or delete itself in the fallback) when training ends. Without this role a finished or
     failed run keeps billing.

## 2. Corpus prep (standard CPU VM, never spot)

```bash
source scripts/gcp/env
scripts/gcp/prepare_data.sh create    # n2-highmem-8, 64GB RAM (>= 32GB required), 500GB disk
# wait for gs://$BUCKET/data/v2/PREPARE_DONE
scripts/gcp/prepare_data.sh delete
```

Prep is a standard VM on purpose. A failure restarts from zero, which costs hours, and a spot
preemption would cause exactly that. Delete the VM as soon as the upload is confirmed.
Training refuses to start until `PREPARE_DONE` exists.

## 3. Start training

```bash
source scripts/gcp/env
GPU_TYPE=h100 scripts/gcp/create_template.sh      # or GPU_TYPE=l4; the command-line value wins over env
```

The script replaces any earlier MIG and template, then waits (default 15 minutes) for a VM to be
RUNNING. If none appears (no quota or capacity) it deletes the MIG and exits non-zero, so a stuck
H100 group can never come up later next to an L4 VM on the same checkpoint directory and W&B run.
To fall back to L4, just rerun with `GPU_TYPE=l4`.

`startup.sh` runs in the foreground on every boot: it clones the branch, runs `uv sync`, copies data if
missing (and fails loudly if any file is missing), checks the GPU with `nvidia-smi` and torch, mounts
checkpoints, fetches the W&B key from Secret Manager (no echo, no tracing; an empty key fails) and trains.
Logs: the last 2000 lines of `/var/log/tinyllm-train.log` are uploaded to
`gs://$BUCKET/status/log_<tag>.txt` before the VM is stopped (the VM disk is deleted with the VM), so read
that file after a DONE or FAILED run. While the VM runs, the log is at `/var/log/tinyllm-train.log`.

When billing stops by itself: training finishes (DONE marker in `gs://$BUCKET/status/`), training
fails 3 times or any boot step aborts (FAILED marker), or a DONE marker already exists. In each
case the VM resizes the MIG to 0. A preemption, a plain VM delete or any other SIGTERM is not a failure: the script does nothing and the
MIG recreates the VM. A MIG-managed VM resizes the MIG; any other VM deletes itself. If both fail the VM
runs `poweroff`; a MIG may recreate a powered-off VM, and that boot then finds the marker and tries again.
Corpus prep that fails leaves `gs://$BUCKET/data/v2/PREPARE_FAILED` (log tail) and deletes its VM.

Unverified: whether a MIG works with `a3-highgpu-1g` (H100). If MIG creation fails, the fallback is a
single spot VM named `$FALLBACK_VM` made from the same template plus a manual restart after each preemption:
```bash
gcloud compute instances create "$FALLBACK_VM" --project "$PROJECT" --zone "$ZONE" --source-instance-template "$TEMPLATE_NAME"
```
With DELETE termination a preempted VM disappears; rerun that command to bring it back, and `startup.sh`
resumes from the checkpoint. The fallback VM deletes itself when training ends. Teardown covers `$FALLBACK_VM`.

## 4. Verify resume (small run first)

Use `run=smoke`/`model=tiny`, delete the VM, and confirm the MIG recreates it and the same W&B run continues.

## 5. Stop billing but keep data

```bash
source scripts/gcp/env
gcloud compute instance-groups managed resize "$MIG_NAME" --size 0 --zone "$ZONE" --project "$PROJECT"
```

This deletes the GPU VM and its disk. Checkpoints, data and the secret stay in the bucket and
Secret Manager and keep costing a small storage fee. To train again, resize to 1 (remove the DONE
marker `gs://$BUCKET/status/DONE_<tag>` first if you want to continue a finished run).

## 6. Teardown

```bash
scripts/gcp/teardown.sh
```

It prints what it will remove (MIG, every labelled VM in any zone, `FALLBACK_VM`, instance template, disks, bucket and its
contents, secret) and requires you to type `DELETE <project>`. It tolerates only "not found" errors; if compute cleanup fails it stops before
touching data, and it lists leftovers at the end and exits non-zero if anything remains. Download anything you want to keep first (checkpoints
are deleted with the bucket). Afterwards open Billing in the console and check that no GPU, disk or
storage charges keep accruing.
