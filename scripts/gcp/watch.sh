#!/usr/bin/env bash
# Live view of the running training VM. Usage: scripts/gcp/watch.sh   (detach with Ctrl-b d)
# Attaches read-only to the `train` tmux viewer, so no keystroke can reach the training process.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"

found="$(gcloud compute instances list --project "$PROJECT" --filter "name~'^$MIG_NAME' AND status=RUNNING" --format 'value(name,zone)')"
if [ -z "$found" ]; then
  echo "no RUNNING VM named $MIG_NAME-*: preempted, stopped or not started" >&2
  exit 1
fi
read -r name zone <<< "$found"
exec gcloud compute ssh "$name" --project "$PROJECT" --zone "${zone##*/}" --tunnel-through-iap -- -t sudo tmux attach -r -t train
