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

```bash
cp scripts/gcp/env.example scripts/gcp/env   # edit it
# bucket
gcloud storage buckets create "gs://$BUCKET" --location "$REGION"
# secret: type the key at the hidden prompt, never put it on a command line
read -rs WANDB_API_KEY && printf %s "$WANDB_API_KEY" | gcloud secrets create "$SECRET_NAME" --data-file=- ; unset WANDB_API_KEY
```

The VM service account needs Storage Object Admin on the bucket and Secret Manager Secret Accessor on the secret.

## 2. Corpus prep (standard CPU VM, never spot)

```bash
scripts/gcp/prepare_data.sh create    # n2-highmem-8, 64GB RAM (>= 32GB required), 500GB disk
# wait for gs://$BUCKET/data/v2/PREPARE_DONE
scripts/gcp/prepare_data.sh delete
```

Prep is a standard VM on purpose. A failure restarts from zero, which costs hours, and a spot
preemption would cause exactly that. Delete the VM as soon as the upload is confirmed.

## 3. Start training

```bash
GPU_TYPE=l4 scripts/gcp/create_template.sh      # or GPU_TYPE=h100
```

`startup.sh` clones the branch, runs `uv sync`, copies data if missing, mounts the checkpoints,
fetches the W&B key from Secret Manager (no echo, no tracing) and starts training. It is safe on every boot.
Logs on the VM: `/var/log/tinyllm-train.log`.

Unverified: whether a MIG works with `a3-highgpu-1g` (H100). If MIG creation fails, the fallback is a
single spot VM created from the same template (`gcloud compute instances create VM --source-instance-template
tinyllm-template --zone ZONE`) plus a manual restart after each preemption: start the stopped/recreated
VM again and `startup.sh` resumes training.

## 4. Verify resume (small run first)

Use `run=smoke`/`model=tiny`, delete the VM, and confirm the MIG recreates it and the same W&B run continues.

## 5. Teardown

```bash
scripts/gcp/teardown.sh
```

It prints what it will remove (MIG, instance template, disks, bucket and its contents, secret) and
requires you to type `DELETE <project>`. Download anything you want to keep first (checkpoints
are deleted with the bucket). Afterwards open Billing in the console and check that no GPU, disk or
storage charges keep accruing.
