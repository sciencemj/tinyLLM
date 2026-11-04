#!/usr/bin/env bash
# Runs on EVERY boot (instance template startup-script). Must be idempotent: mount, copy data
# if missing, check out the pinned commit, then train. The VM's startup service runs this script
# to completion (it does not return until training ends); training is a child process and all
# output goes to the log file, never the console.
# Training resumes itself via tinyllm.v2.train.resolve_resume from the stable ckpt_dir.
#
# One exit path, so a GPU VM never idles while billing:
#   SIGTERM/INT/HUP (shutdown, preemption, VM delete) -> do nothing: no marker, no resize
#   training finished                  -> finish DONE
#   training failed 3 times, or any boot step failed -> finish FAILED
#   a DONE or FAILED marker already exists -> stop the VM, no training, and neither the marker
#     nor the uploaded log is touched (clear the FAILED marker to retry: docs/gcp-runbook.md)
#   finish = marker + log tail to the bucket, then stop the VM: resize the MIG to 0 (in a MIG)
#   or delete itself (not in a MIG), and `poweroff` if that fails.
#   Every boot step runs under `timeout`; a timeout is a failed step -> finish FAILED.
#   Failure before the metadata is read (finish cannot know the bucket) -> poweroff.
set -euo pipefail
set +x   # never trace: the W&B key passes through this script
export HOME="${HOME:-/root}"
LOG="${LOG_FILE:-/var/log/tinyllm-train.log}"
FINISHED=""; READY=""; TRAIN_PID=""

on_exit() {
  local rc=$?
  [ "$rc" -ne 0 ] && [ -z "$FINISHED" ] || return 0
  echo "startup failed (exit $rc)" >&2
  if [ -n "$READY" ]; then finish FAILED; else poweroff; fi
}
on_signal() {   # shutdown or preemption: the MIG decides what happens next
  trap - EXIT
  [ -z "$TRAIN_PID" ] || kill "$TRAIN_PID" 2>/dev/null || true
  exit 143
}
trap on_exit EXIT
trap on_signal TERM INT HUP

# Installed before anything can fail, so even an unwritable log path ends in poweroff.
# From here on every line of output, including a failed boot step's stderr, goes to the log that
# finish uploads. Tracing stays off (set +x above): the W&B key must never reach the log.
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

MD_BASE=http://metadata.google.internal/computeMetadata/v1
md() { curl -fsS --retry 5 --retry-delay 2 --retry-connrefused -H 'Metadata-Flavor: Google' "$MD_BASE/$1"; }

BUCKET="$(md instance/attributes/tl-bucket)"; SECRET="$(md instance/attributes/tl-secret)"
REPO="$(md instance/attributes/tl-repo)"; BRANCH="$(md instance/attributes/tl-branch)"
MODEL="$(md instance/attributes/tl-model)"; DATA="$(md instance/attributes/tl-data)"
RUN="$(md instance/attributes/tl-run)"; TAG="$(md instance/attributes/tl-tag)"
MIG="$(md instance/attributes/tl-mig)"; ZONE="$(md instance/attributes/tl-zone)"
PROJECT="$(md instance/attributes/tl-project)"; COMMIT="$(md instance/attributes/tl-commit)"
# Optional: only templates created with INIT_FROM set carry this key (a 404 here is not an error).
INIT_FROM="$(md instance/attributes/tl-init-from 2>/dev/null || true)"
APP="${APP_DIR:-/opt/tinyllm}"
CKPT_MNT="${CKPT_MNT:-/mnt/ckpt}"
MAX_TRIES="${MAX_TRIES:-3}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"
# Upper bounds (seconds) for boot steps. Training itself has none: it is a long run.
T_APT="${T_APT:-900}"; T_GIT="${T_GIT:-600}"; T_SYNC="${T_SYNC:-1800}"
T_COPY="${T_COPY:-7200}"; T_MOUNT="${T_MOUNT:-180}"; T_LS="${T_LS:-300}"; T_NET="${T_NET:-300}"
READY=1

# Stop billing: resize the MIG to 0. created-by names the MIG only for MIG-managed VMs; any other
# VM (the fallback) deletes itself instead. If both fail, poweroff.
stop_vm() {
  FINISHED=1
  set +e
  if [[ "$(md instance/attributes/created-by 2>/dev/null)" == *instanceGroupManagers* ]]; then
    gcloud compute instance-groups managed resize "$MIG" --size 0 --zone "$ZONE" --project "$PROJECT" && return 0
  fi
  gcloud compute instances delete "$(md instance/name)" --zone "$ZONE" --project "$PROJECT" --quiet && return 0
  poweroff
}

# Marker and log first (their failure never blocks the stop), then the VM itself.
finish() {
  local state="$1"
  FINISHED=1
  set +e
  printf '%s %s\n' "$state" "$(date -u +%FT%TZ)" | gcloud storage cp - "gs://$BUCKET/status/${state}_$TAG" \
    || echo "could not write $state marker" >&2
  tail -n 2000 "$LOG" | gcloud storage cp - "gs://$BUCKET/status/log_$TAG.txt" \
    || echo "could not upload the log" >&2
  stop_vm
}

# A finished or failed run must not train again: a DONE run would waste money, and a FAILED run
# (for example a missing IAM role) would loop boot, train, fail. These boots only stop the VM and
# leave the marker and the uploaded training log exactly as they are.
for state in DONE FAILED; do
  if gcloud storage ls "gs://$BUCKET/status/${state}_$TAG" >/dev/null 2>&1; then
    echo "run $TAG already $state: not training. To retry after a FAILED run, delete gs://$BUCKET/status/FAILED_$TAG"
    stop_vm
    exit 0
  fi
done

export DEBIAN_FRONTEND=noninteractive
APT=(apt-get -o DPkg::Lock::Timeout=300 -y)
if ! command -v git >/dev/null; then
  timeout "$T_APT" "${APT[@]}" update
  timeout "$T_APT" "${APT[@]}" install git curl
fi

if ! command -v gcsfuse >/dev/null; then
  CODENAME="$(lsb_release -c -s)"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$CODENAME main" > /etc/apt/sources.list.d/gcsfuse.list
  timeout "$T_NET" curl -fsS https://packages.cloud.google.com/apt/doc/apt-key.gpg -o /usr/share/keyrings/cloud.google.asc
  timeout "$T_APT" "${APT[@]}" update
  timeout "$T_APT" "${APT[@]}" install gcsfuse
fi
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  timeout "$T_NET" curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Checkpoints: one stable absolute path on every VM, so a restart sees the same directory.
mkdir -p "$CKPT_MNT"
if ! mountpoint -q "$CKPT_MNT"; then
  timeout "$T_MOUNT" gcsfuse --implicit-dirs --only-dir ckpt "$BUCKET" "$CKPT_MNT"
fi
timeout "$T_MOUNT" ls "$CKPT_MNT" >/dev/null   # a broken mount must fail here, not look like "no checkpoint"
CKPT_DIR="$CKPT_MNT/$TAG"

# Repo: clone once, then check out the exact commit the template was created from (tl-commit),
# not the branch head, so a push during the run cannot change code under a resumed run. The
# commit must be on origin/BRANCH (the repo is public; HTTPS clone), else refuse to train.
if [ ! -d "$APP/.git" ]; then
  timeout "$T_GIT" git clone --branch "$BRANCH" --single-branch "$REPO" "$APP"
fi
cd "$APP"
timeout "$T_GIT" git fetch origin "$BRANCH"
if ! git merge-base --is-ancestor "$COMMIT" "origin/$BRANCH"; then
  echo "commit $COMMIT is not on origin/$BRANCH: push it, then recreate the template" >&2
  exit 1
fi
git checkout -q --detach "$COMMIT"
timeout "$T_SYNC" uv sync

# Data: local disk, not gcsfuse (random reads are too slow). Skip files already present.
# Corpus prep must have finished: its last step writes PREPARE_DONE.
listing="$(timeout "$T_LS" gcloud storage ls "gs://$BUCKET/data/v2/")"
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
    timeout "$T_COPY" gcloud storage cp "$uri" "data/v2/$f.part"
    mv "data/v2/$f.part" "data/v2/$f"
  fi
done <<< "$listing"
# Every file the chosen data config names must now exist.
while read -r want; do
  [ -f "$want" ] || { echo "missing data file $want (from configs/data/$DATA.yaml)" >&2; exit 1; }
done < <(awk -F': *' '/^(train_bin|val_bin|tokenizer):/ {sub(/[ \t]*#.*/, "", $2); print $2}' "configs/data/$DATA.yaml")

# GPU sanity: without CUDA, training silently runs on CPU while the GPU bills.
timeout "$T_NET" nvidia-smi >/dev/null
timeout "$T_SYNC" uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# W&B key from Secret Manager. Not echoed, not written to disk, not in metadata.
set +x
key="$(timeout "$T_NET" gcloud secrets versions access latest --secret "$SECRET")"
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
  # Background + wait, so a SIGTERM reaches the trap at once instead of after training.
  uv run python -m tinyllm.v2.train \
    model="$MODEL" data="$DATA" run="$RUN" tag="$TAG" ckpt_dir="$CKPT_DIR" \
    ${INIT_FROM:+"run.init_from=$INIT_FROM"} &
  TRAIN_PID=$!
  wait "$TRAIN_PID" || rc=$?
  TRAIN_PID=""
  [ "$rc" -eq 0 ] && break
  echo "training attempt $i/$MAX_TRIES exited $rc" >&2
  sleep "$RETRY_SLEEP"
done
if [ "$rc" -eq 0 ]; then
  finish DONE
  exit 0
fi
exit "$rc"   # the exit trap runs finish FAILED
