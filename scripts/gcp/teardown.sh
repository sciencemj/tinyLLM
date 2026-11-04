#!/usr/bin/env bash
# DESTRUCTIVE. Deletes every billable resource of this project's run, then asks you to check Billing.
# Only "not found" errors are tolerated; any other failure stops the script before data is touched,
# and leftovers are listed at the end (non-zero exit if anything remains).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"
ROUTER_NAME="${ROUTER_NAME:-tinyllm-router}"

cat <<MSG
This will PERMANENTLY remove from project $PROJECT (all zones for VMs and disks):
  - managed instance group  $MIG_NAME and its VM
  - every VM labelled app=tinyllm-v2, and the VMs $FALLBACK_VM and $PREPARE_VM
  - instance template       $TEMPLATE_NAME
  - disks named ${MIG_NAME}-*, $FALLBACK_VM, $PREPARE_VM
  - Cloud Router $ROUTER_NAME and its NAT
  - bucket gs://$BUCKET and ALL its contents (data, checkpoints)
  - secret                  $SECRET_NAME
MSG
read -r -p "Type DELETE $PROJECT to continue: " answer
if [ "$answer" != "DELETE $PROJECT" ]; then
  echo "Aborted; nothing removed."
  exit 1
fi

FAILED=0
# try_delete DESCRIPTION CMD...: a "not found" error is fine, anything else is a failure.
try_delete() {
  local what="$1" out; shift
  if out="$("$@" 2>&1)"; then
    echo "removed $what"
  elif grep -qi "not found" <<< "$out"; then
    echo "already gone: $what"
  else
    echo "FAILED to remove $what: $out" >&2
    FAILED=1
  fi
}

# The regex operand MUST be single-quoted: unquoted, the filter parser stops at "(" and the
# filter silently selects nothing, so deletes and the leftover check would do nothing.
NAMES="name~'^($MIG_NAME|$FALLBACK_VM|$PREPARE_VM)(-|\$)'"

# 1. Compute. Anything that bills by the hour goes before data is touched.
migs="$(gcloud compute instance-groups managed list --project "$PROJECT" --filter "name='$MIG_NAME'" --format 'value(name,zone)')"
while read -r name zone; do
  [ -n "$name" ] || continue
  try_delete "MIG $name" gcloud compute instance-groups managed delete "$name" --project "$PROJECT" --zone "${zone##*/}" --quiet
done <<< "$migs"
vms="$(gcloud compute instances list --project "$PROJECT" --filter "labels.app='tinyllm-v2' OR $NAMES" --format 'value(name,zone)')"
while read -r name zone; do
  [ -n "$name" ] || continue
  try_delete "VM $name" gcloud compute instances delete "$name" --project "$PROJECT" --zone "${zone##*/}" --quiet
done <<< "$vms"
try_delete "template $TEMPLATE_NAME" gcloud compute instance-templates delete "$TEMPLATE_NAME" --project "$PROJECT" --quiet
disks="$(gcloud compute disks list --project "$PROJECT" --filter "$NAMES" --format 'value(name,zone)')"
while read -r name zone; do
  [ -n "$name" ] || continue
  try_delete "disk $name" gcloud compute disks delete "$name" --project "$PROJECT" --zone "${zone##*/}" --quiet
done <<< "$disks"
try_delete "router $ROUTER_NAME and its NAT" gcloud compute routers delete "$ROUTER_NAME" --project "$PROJECT" --region "$REGION" --quiet
if [ "$FAILED" -ne 0 ]; then
  echo "Compute cleanup failed: bucket and secret were NOT touched. Fix the errors above and rerun." >&2
  exit 1
fi

# 2. Data and secret.
try_delete "bucket gs://$BUCKET" gcloud storage rm -r "gs://$BUCKET" --project "$PROJECT"
try_delete "secret $SECRET_NAME" gcloud secrets delete "$SECRET_NAME" --project "$PROJECT" --quiet

# 3. Verify: list what is left.
left=""
# A listing failure must abort (set -e): it is never read as "nothing left".
check() { local label="$1" out; shift; out="$("$@")"; [ -z "$out" ] || left+="$label: $out"$'\n'; }
check "managed instance groups" gcloud compute instance-groups managed list --project "$PROJECT" --filter "name='$MIG_NAME'" --format 'value(name)'
check "VMs" gcloud compute instances list --project "$PROJECT" --filter "labels.app='tinyllm-v2' OR $NAMES" --format 'value(name)'
check "templates" gcloud compute instance-templates list --project "$PROJECT" --filter "name='$TEMPLATE_NAME'" --format 'value(name)'
check "disks" gcloud compute disks list --project "$PROJECT" --filter "$NAMES" --format 'value(name)'
check "routers" gcloud compute routers list --project "$PROJECT" --filter "name='$ROUTER_NAME'" --format 'value(name)'
check "buckets" gcloud storage buckets list --project "$PROJECT" --filter "name='$BUCKET'" --format 'value(name)'
check "secrets" gcloud secrets list --project "$PROJECT" --filter "name~'$SECRET_NAME'" --format 'value(name)'
if [ -n "$left" ] || [ "$FAILED" -ne 0 ]; then
  echo "Teardown INCOMPLETE. Still present:" >&2
  printf '%s' "$left" >&2
  exit 1
fi

echo
echo "Everything listed is gone. Now open Billing in the console and confirm no GPU, disk or storage charges are accruing."
