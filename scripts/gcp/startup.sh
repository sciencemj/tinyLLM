#!/usr/bin/env bash
# Runs on EVERY boot (instance template startup-script). Must be idempotent:
# mount, copy data if missing, update the repo, run training in the foreground.
# Training resumes itself via tinyllm.v2.train.resolve_resume from the stable ckpt_dir.
#
# Exit policy (so a GPU VM never idles while billing):
#   training succeeds                  -> DONE marker in the bucket, MIG resized to 0
#   training fails 3 times, or any boot step aborts -> FAILED marker, MIG resized to 0
#   a DONE marker already exists       -> do not train, MIG resized to 0
# A preemption kills the VM without running the exit trap; the MIG recreates it.
set -euo pipefail
set +x   # never trace: the W&B key passes through this script
export HOME="${HOME:-/root}"
MD_BASE=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS --retry 5 --retry-connrefused -H 'Metadata-Flavor: Google' "$MD_BASE/$1"; }

BUCKET="$(md instance/attributes/tl-bucket)"; SECRET="$(md instance/attributes/tl-secret)"
REPO="$(md instance/attributes/tl-repo)"; BRANCH="$(md instance/attributes/tl-branch)"
MODEL="$(md instance/attributes/tl-model)"; DATA="$(md instance/attributes/tl-data)"
RUN="$(md instance/attributes/tl-run)"; TAG="$(md instance/attributes/tl-tag)"
MIG="$(md instance/attributes/tl-mig)"; ZONE="$(md instance/attributes/tl-zone)"
PROJECT="$(md instance/attributes/tl-project)"
APP="${APP_DIR:-/opt/tinyllm}"
CKPT_MNT="${CKPT_MNT:-/mnt/ckpt}"
LOG="${LOG_FILE:-/var/log/tinyllm-train.log}"
MAX_TRIES="${MAX_TRIES:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
FINISHED=""

# Stop billing: write the marker, then resize the MIG to 0. A VM that is not in a MIG
# (the single-VM fallback) deletes itself instead. Disks, bucket and secret are kept.
finish() {
  local state="$1"
  FINISHED=1
  set +e
  printf '%s %s\n' "$state" "$(date -u +%FT%TZ)" | gcloud storage cp - "gs://$BUCKET/status/${state}_$TAG" \
    || echo "could not write $state marker" >&2
  if ! gcloud compute instance-groups managed resize "$MIG" --size 0 --zone "$ZONE" --project "$PROJECT"; then
    gcloud compute instances delete "$(md instance/name)" --zone "$ZONE" --project "$PROJECT" --quiet
  fi
}
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ -z "$FINISHED" ]; then
    # Preempted: the VM is going away anyway and the MIG will recreate it. Do not give up.
    if [ "$(md instance/preempted 2>/dev/null || true)" = "TRUE" ]; then
      return
    fi
    echo "startup failed (exit $rc): stopping the GPU VM" >&2
    finish FAILED
  fi
}
trap on_exit EXIT

if gcloud storage ls "gs://$BUCKET/status/DONE_$TAG" >/dev/null 2>&1; then
  echo "run $TAG already DONE: not training"
  finish DONE
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
APT=(apt-get -o DPkg::Lock::Timeout=300 -y)
command -v git >/dev/null || { "${APT[@]}" update && "${APT[@]}" install git curl; }

if ! command -v gcsfuse >/dev/null; then
  CODENAME="$(lsb_release -c -s)"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$CODENAME main" > /etc/apt/sources.list.d/gcsfuse.list
  curl -fsS https://packages.cloud.google.com/apt/doc/apt-key.gpg -o /usr/share/keyrings/cloud.google.asc
  "${APT[@]}" update && "${APT[@]}" install gcsfuse
fi
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Checkpoints: one stable absolute path on every VM, so a restart sees the same directory.
mkdir -p "$CKPT_MNT"
if ! mountpoint -q "$CKPT_MNT"; then
  gcsfuse --implicit-dirs --only-dir ckpt "$BUCKET" "$CKPT_MNT"
fi
ls "$CKPT_MNT" >/dev/null   # a broken mount must fail here, not look like "no checkpoint"
CKPT_DIR="$CKPT_MNT/$TAG"

# Repo: clone once, then follow the configured branch (the repo is public; HTTPS clone).
if [ ! -d "$APP/.git" ]; then
  git clone --branch "$BRANCH" --single-branch "$REPO" "$APP"
fi
cd "$APP"
git fetch origin "$BRANCH"
git checkout -q -B "$BRANCH" "origin/$BRANCH"
uv sync

# Data: local disk, not gcsfuse (random reads are too slow). Skip files already present.
# Corpus prep must have finished: its last step writes PREPARE_DONE.
listing="$(gcloud storage ls "gs://$BUCKET/data/v2/")"
case "$listing" in
  *"/PREPARE_DONE"*) ;;
  *) echo "gs://$BUCKET/data/v2/PREPARE_DONE is missing: corpus prep is not finished" >&2; exit 1 ;;
esac
mkdir -p data/v2
while read -r uri; do
  f="$(basename "$uri")"
  case "$f" in
    *.bin|*.meta.json|tokenizer.json) ;;
    *) continue ;;
  esac
  if [ ! -f "data/v2/$f" ]; then
    gcloud storage cp "$uri" "data/v2/$f.part"
    mv "data/v2/$f.part" "data/v2/$f"
  fi
done <<< "$listing"
# Every file the chosen data config names must now exist.
while read -r want; do
  [ -f "$want" ] || { echo "missing data file $want (from configs/data/$DATA.yaml)" >&2; exit 1; }
done < <(awk -F': *' '/^(train_bin|val_bin|tokenizer):/ {sub(/[ \t]*#.*/, "", $2); print $2}' "configs/data/$DATA.yaml")

# GPU sanity: without CUDA, training silently runs on CPU while the GPU bills.
nvidia-smi >/dev/null
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# W&B key from Secret Manager. Not echoed, not written to disk, not in metadata.
set +x
key="$(gcloud secrets versions access latest --secret "$SECRET")"
if [ -z "$key" ]; then
  echo "secret $SECRET is empty: refusing to run with offline W&B" >&2
  exit 1
fi
printf -v WANDB_API_KEY %s "$key"
unset key
export WANDB_API_KEY

mkdir -p out/wandb   # live W&B files stay on local disk
rc=1
for ((i = 1; i <= MAX_TRIES; i++)); do
  rc=0
  uv run python -m tinyllm.v2.train \
    model="$MODEL" data="$DATA" run="$RUN" tag="$TAG" ckpt_dir="$CKPT_DIR" \
    >> "$LOG" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && break
  echo "training attempt $i/$MAX_TRIES exited $rc" >&2
  sleep "$RETRY_SLEEP"
done
if [ "$rc" -eq 0 ]; then
  finish DONE
  exit 0
fi
exit "$rc"   # the exit trap writes FAILED and resizes the MIG to 0
