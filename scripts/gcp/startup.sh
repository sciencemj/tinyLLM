#!/usr/bin/env bash
# Runs on EVERY boot (instance template startup-script). Must be idempotent:
# mount, copy data if missing, update the repo, start training. Training resumes
# itself via tinyllm.v2.train.resolve_resume from the stable ckpt_dir.
set -euo pipefail
set +x   # never trace: the W&B key passes through this script
export HOME=/root
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -fsS -H 'Metadata-Flavor: Google' "$MD/$1"; }

BUCKET="$(md tl-bucket)"; SECRET="$(md tl-secret)"; REPO="$(md tl-repo)"; BRANCH="$(md tl-branch)"
MODEL="$(md tl-model)"; DATA="$(md tl-data)"; RUN="$(md tl-run)"; TAG="$(md tl-tag)"
APP=/opt/tinyllm
CKPT_MNT=/mnt/ckpt
LOG=/var/log/tinyllm-train.log

export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || { apt-get update -y && apt-get install -y git curl; }

if ! command -v gcsfuse >/dev/null; then
  CODENAME="$(lsb_release -c -s)"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$CODENAME main" > /etc/apt/sources.list.d/gcsfuse.list
  curl -fsS https://packages.cloud.google.com/apt/doc/apt-key.gpg -o /usr/share/keyrings/cloud.google.asc
  apt-get update -y && apt-get install -y gcsfuse
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
CKPT_DIR="$CKPT_MNT/$TAG"

# Repo: clone once, then follow the branch.
if [ ! -d "$APP/.git" ]; then
  git clone --branch "$BRANCH" --single-branch "$REPO" "$APP"
fi
cd "$APP"
git fetch origin "$BRANCH"
git checkout -q "$BRANCH"
git reset --hard "origin/$BRANCH"
uv sync

# Data: local disk, not gcsfuse (random reads are too slow). Skip files already present.
mkdir -p data/v2
gcloud storage ls "gs://$BUCKET/data/v2/" | while read -r uri; do
  f="$(basename "$uri")"
  case "$f" in
    *.bin|*.meta.json|tokenizer.json) ;;
    *) continue ;;
  esac
  if [ ! -f "data/v2/$f" ]; then
    gcloud storage cp "$uri" "data/v2/$f.part" && mv "data/v2/$f.part" "data/v2/$f"
  fi
done

# Already training (script re-run without a reboot): do nothing more.
if pgrep -f "tinyllm.v2.train" >/dev/null; then
  exit 0
fi

# W&B key from Secret Manager. Not echoed, not written to disk, not in metadata.
set +x
key="$(gcloud secrets versions access latest --secret "$SECRET")"
printf -v WANDB_API_KEY %s "$key"
unset key
export WANDB_API_KEY

mkdir -p out/wandb   # live W&B files stay on local disk
nohup uv run python -m tinyllm.v2.train \
  model="$MODEL" data="$DATA" run="$RUN" tag="$TAG" ckpt_dir="$CKPT_DIR" \
  >> "$LOG" 2>&1 &
