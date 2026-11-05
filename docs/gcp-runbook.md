# GCP spot run runbook (tinyLLM v2)

Layout: a size-1 managed instance group (MIG) built from a spot instance template. After a
preemption the MIG recreates the VM, `startup.sh` runs again, and training resumes by itself
(`resolve_resume`) from the checkpoint directory `/mnt/ckpt/<tag>` (gcsfuse over the bucket).
Data is copied from GCS to the VM's local disk at boot (random reads over gcsfuse are too slow).
The template uses the STOP termination action because MIGs reject DELETE on spot templates. A VM the
MIG recreates has a fresh disk and re-copies the data (about 41GB) before it trains; expect several
minutes of lost time per preemption. Unverified: how the MIG treats a VM stopped by a real preemption.

All scripts read `scripts/gcp/env` (copy from `env.example`). No script contains a secret.

## 0. Before ANY run: you must confirm each item yourself

Do not start until you have confirmed all of these. Nothing below does it for you.

- [ ] Paid account upgrade is done (free-trial accounts cannot use GPUs).
- [ ] GPU quotas approved in your region: H100 (`a3-highgpu-1g`, spot) and/or L4 (`g2-standard-8`, spot). Preemptible/spot GPU quota is separate from on-demand quota.
- [ ] A budget alert exists. A budget alert only notifies; it does not stop spending. Nothing caps the bill automatically.
- [ ] You have looked up the current per-hour price of the chosen machine type, boot disk (300GB pd-ssd) and bucket storage, and multiplied by the expected hours.
- [ ] You understand that once free credits run out, real charges start on your card.
- [ ] W&B API key is stored in Secret Manager (see below). HF key only if needed.
- [ ] The boot image exists and ships a driver new enough for the locked torch (CUDA 13, driver 580 line).
  Before creating any VM run this and read the output (`env.example` defaults to `common-cu129-ubuntu-2204-nvidia-580`):
  ```bash
  source scripts/gcp/env
  gcloud compute images describe-from-family "$IMAGE_FAMILY" --project "$IMAGE_PROJECT"
  ```
  The proof is still the boot-time `torch.cuda.is_available()` check in `startup.sh`: without CUDA the run fails before training.

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
   # --soft-delete-duration=0: no soft-delete retention, so deleted checkpoints stop billing at once
   gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" --soft-delete-duration=0
   # type the key at the hidden prompt; never put it on a command line
   read -rs WANDB_API_KEY && printf %s "$WANDB_API_KEY" | gcloud secrets create "$SECRET_NAME" --project "$PROJECT" --data-file=- ; unset WANDB_API_KEY
   ```
5. The VM's service account (default: the Compute Engine default account, or `SERVICE_ACCOUNT`) needs:
   - `roles/storage.objectAdmin` on the bucket (data, checkpoints, status markers),
   - `roles/secretmanager.secretAccessor` on the secret,
   - `roles/compute.instanceAdmin.v1` on the project, so the VM can resize its own MIG to 0
     (or delete itself in the fallback) when training ends. Without this role a finished or
     failed run keeps billing.

6. Create the network path once. The org policy `compute.vmExternalIpAccess` denies external IPs, so every
   VM is created with `--no-address`: Google APIs use Private Google Access and GitHub, Hugging Face and apt
   go through Cloud NAT (outbound only):
   ```bash
   scripts/gcp/setup_network.sh
   ```
   Cloud NAT bills per hour and per GB processed. Traffic to GCS uses Private Google Access, not NAT.
   Teardown removes the router and NAT.

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

Order: section 1 (setup), section 2 (corpus prep), then the section 4 resume test with a small
run, and only then this section with the real `pretrain` values from `env`.

```bash
source scripts/gcp/env
GPU_TYPE=h100 scripts/gcp/create_template.sh      # or GPU_TYPE=l4; the command-line value wins over env
PROVISIONING=STANDARD GPU_TYPE=l4 scripts/gcp/create_template.sh   # on demand: not preempted, costs more
```

Observed on L4 spot in us-central1 (2026-11-05): VMs lasted 5 to 58 minutes, recreation failed for 20 to 34
minutes on stockouts, and boot to training took about 11 minutes. A checkpoint save takes about 70 seconds, so
`save_every_minutes` must stay far above that. Watch a run live with `scripts/gcp/watch.sh` (read-only tmux: log with `progress step=` lines, GPU stats below;
detach with Ctrl-b d). Training runs outside tmux, so detaching or a dropped connection cannot affect it.
Keep `RUN=bench` (batch sweep, then a short run to DONE) for a new GPU.

The script replaces any earlier MIG and template, then waits (default 15 minutes) for a VM to be
RUNNING. If none appears (no quota or capacity) it deletes the MIG and exits non-zero, so a stuck
H100 group can never come up later next to an L4 VM on the same checkpoint directory and W&B run.
To fall back to L4, just rerun with `GPU_TYPE=l4`.

`startup.sh` is the VM's startup script and runs on every boot until training ends (training is a
child process; everything goes to the log file). It mounts checkpoints, clones the repo and checks out
the commit `create_template.sh` recorded (`tl-commit`, the HEAD you had when you ran it, which must be
pushed to `origin/$BRANCH`; a later push does not change a running or resumed job), runs `uv sync`, copies
data if missing (and fails loudly if any file is missing), checks the GPU with `nvidia-smi` and torch,
fetches the W&B key from Secret Manager (no echo, no tracing; an empty key fails) and trains. Every boot
step (apt, git, `uv sync`, data copy, mount) has a time limit; a timeout counts as a failed step. Training
itself has no limit.
Logs: the last 2000 lines of `/var/log/tinyllm-train.log` are uploaded to
`gs://$BUCKET/status/log_<tag>.txt` before the VM is stopped (the VM disk is deleted with the VM), so read
that file after a DONE or FAILED run. While the VM runs, the log is at `/var/log/tinyllm-train.log`.

When billing stops by itself: training finishes (DONE marker in `gs://$BUCKET/status/`), training
fails 3 times or any boot step aborts (FAILED marker), or a DONE or FAILED marker already exists
(that boot only stops the VM; it leaves the marker and the uploaded log untouched). In each
case the VM resizes the MIG to 0. A FAILED run never retrains by itself, so a missing IAM role cannot
cause a billing retrain loop. To retry after fixing the cause, read `log_<tag>.txt`, then delete the marker:
`gcloud storage rm "gs://$BUCKET/status/FAILED_<tag>"` and rerun `create_template.sh`. A preemption, a plain VM delete or any other SIGTERM is not a failure: the script does nothing and the
MIG recreates the VM. A MIG-managed VM resizes the MIG; any other VM deletes itself. If both fail the VM
runs `poweroff`; a MIG may recreate a powered-off VM, and that boot then finds the marker and tries again.
Corpus prep deletes its own VM when it finishes or fails. A failure leaves `gs://$BUCKET/data/v2/PREPARE_FAILED` (log tail);
a later success removes that marker.

SFT starts from a pretrained checkpoint: set `INIT_FROM=/mnt/ckpt/<pretrain-tag>/best.ckpt` (in `env` or on the
command line) with `RUN=sft` and a new `TAG`. It becomes the `tl-init-from` metadata key and `run.init_from`;
a restarted job resumes from its own checkpoints first, so `init_from` only applies to a fresh start.

Unverified: whether a MIG works with `a3-highgpu-1g` (H100). If MIG creation fails, the fallback is a
single spot VM named `$FALLBACK_VM` made from the same template plus a manual restart after each preemption:
```bash
gcloud compute instances create "$FALLBACK_VM" --project "$PROJECT" --zone "$ZONE" --source-instance-template "$TEMPLATE_NAME"
```
With STOP termination a preempted VM stays stopped; delete it and rerun that command to bring it back, and `startup.sh`
resumes from the checkpoint. The fallback VM deletes itself when training ends. Teardown covers `$FALLBACK_VM`.

## 4. Verify resume (small run first)

Run this after corpus prep (section 2; it trains on the real `fineweb` data, so `PREPARE_DONE` must exist) and
before the long pretrain. Use the `resume_test` run (block 256, batch 8, 3000 steps, a checkpoint every 5 minutes):

```bash
source scripts/gcp/env
MODEL=main DATA=fineweb RUN=resume_test TAG=resume-test-1 GPU_TYPE=l4 scripts/gcp/create_template.sh
```

Command-line values beat `env`. Wait until `/mnt/ckpt/resume-test-1/` holds a `.ckpt` file (a few minutes), delete the VM
(`gcloud compute instances delete <vm> --zone "$ZONE" --project "$PROJECT"`), and confirm the MIG recreates it, the log
says `resuming from ...`, the step counter does not restart at 0 and the same W&B run continues. When it finishes (DONE marker) the MIG is
resized to 0; the real run then starts in section 3 with its own `TAG`.

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

It prints what it will remove (MIG, every labelled VM in any zone, `FALLBACK_VM`, instance template, disks, Cloud Router and NAT, bucket and its
contents, secret) and requires you to type `DELETE <project>`. It tolerates only "not found" errors; if compute cleanup fails it stops before
touching data, and it lists leftovers at the end and exits non-zero if anything remains. Download anything you want to keep first (checkpoints
are deleted with the bucket). Afterwards open Billing in the console and check that no GPU, disk or
storage charges keep accruing.
