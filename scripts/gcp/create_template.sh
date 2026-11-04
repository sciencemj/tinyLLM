#!/usr/bin/env bash
# Create the spot instance template and a size-1 MIG that recreates the VM after preemption.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A value given on the command line must beat the env file (for example MODEL=main DATA=fineweb
# RUN=resume_test TAG=resume-test-1 GPU_TYPE=l4 for the resume test).
CALLER_VARS=(GPU_TYPE MODEL DATA RUN TAG INIT_FROM)
for v in "${CALLER_VARS[@]}"; do printf -v "CALLER_$v" %s "${!v:-}"; done
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"
for v in "${CALLER_VARS[@]}"; do
  c="CALLER_$v"
  [ -z "${!c}" ] || printf -v "$v" %s "${!c}"
done
INIT_FROM="${INIT_FROM:-}"

case "${GPU_TYPE:?set GPU_TYPE=h100|l4}" in
  h100) MACHINE_TYPE=a3-highgpu-1g ;;
  l4)   MACHINE_TYPE=g2-standard-8 ;;
  *) echo "GPU_TYPE must be h100 or l4, got: $GPU_TYPE" >&2; exit 2 ;;
esac

# Pin the code: the VM checks out exactly this commit, so it must already be on origin.
COMMIT="$(git -C "$HERE" rev-parse HEAD)"
if ! git -C "$HERE" merge-base --is-ancestor "$COMMIT" "origin/$BRANCH"; then
  echo "commit $COMMIT is not on origin/$BRANCH (as last fetched): git push first" >&2
  exit 1
fi

WAIT_SECS="${WAIT_SECS:-900}"
POLL_SECS="${POLL_SECS:-15}"

# Replace any earlier attempt (for example an H100 try before falling back to L4), so two
# groups can never run side by side on the same checkpoint directory and W&B run.
existing="$(gcloud compute instance-groups managed list --project "$PROJECT" --filter "name='$MIG_NAME'" --format 'value(name,zone)')"
if [ -n "$existing" ]; then
  while read -r name zone; do
    gcloud compute instance-groups managed delete "$name" --project "$PROJECT" --zone "${zone##*/}" --quiet
  done <<< "$existing"
fi
# A fallback VM from an earlier attempt would share the checkpoint directory and W&B run.
existing="$(gcloud compute instances list --project "$PROJECT" --filter "name='$FALLBACK_VM'" --format 'value(name,zone)')"
if [ -n "$existing" ]; then
  while read -r name zone; do
    gcloud compute instances delete "$name" --project "$PROJECT" --zone "${zone##*/}" --quiet
  done <<< "$existing"
fi
existing="$(gcloud compute instance-templates list --project "$PROJECT" --filter "name='$TEMPLATE_NAME'" --format 'value(name)')"
if [ -n "$existing" ]; then
  gcloud compute instance-templates delete "$TEMPLATE_NAME" --project "$PROJECT" --quiet
fi

# Both machine types have the GPU built in, so no --accelerator flag.
# GPU VMs cannot live-migrate: TERMINATE on maintenance. Spot only; never STANDARD for GPU.
# Metadata carries names only. The W&B key is fetched from Secret Manager at boot.
gcloud compute instance-templates create "$TEMPLATE_NAME" \
  --project "$PROJECT" \
  --machine-type "$MACHINE_TYPE" \
  --provisioning-model SPOT \
  --instance-termination-action DELETE \
  --maintenance-policy TERMINATE \
  --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
  --boot-disk-size "${BOOT_DISK_GB}GB" --boot-disk-type pd-ssd \
  --no-address \
  --scopes cloud-platform \
  --labels app=tinyllm-v2 \
  ${SERVICE_ACCOUNT:+--service-account "$SERVICE_ACCOUNT"} \
  --metadata "tl-bucket=$BUCKET,tl-secret=$SECRET_NAME,tl-repo=$REPO_URL,tl-branch=$BRANCH,tl-model=$MODEL,tl-data=$DATA,tl-run=$RUN,tl-tag=$TAG,tl-mig=$MIG_NAME,tl-zone=$ZONE,tl-project=$PROJECT,tl-commit=$COMMIT${INIT_FROM:+,tl-init-from=$INIT_FROM}" \
  --metadata-from-file startup-script="$HERE/startup.sh"

# Unverified: a MIG with a3-highgpu-1g. If this fails, see docs/gcp-runbook.md (single spot VM fallback).
gcloud compute instance-groups managed create "$MIG_NAME" \
  --project "$PROJECT" --zone "$ZONE" \
  --template "$TEMPLATE_NAME" --size 1

# Creating the group does not mean a VM exists: with no quota or capacity the group keeps retrying.
# Wait for a RUNNING instance; otherwise delete the group so it cannot start later unnoticed.
attempts=$(( WAIT_SECS / POLL_SECS ))
for ((i = 0; i < attempts; i++)); do
  # A transient listing failure must not abort here: set -e would leave the group (and its VM) alive.
  status="$(gcloud compute instance-groups managed list-instances "$MIG_NAME" --project "$PROJECT" --zone "$ZONE" --format 'value(instanceStatus)')" || status=""
  if [[ "$status" == *RUNNING* ]]; then
    echo "MIG $MIG_NAME has a RUNNING $MACHINE_TYPE VM. GPU billing is running; run docs/gcp-runbook.md section 5 to stop it."
    exit 0
  fi
  sleep "$POLL_SECS"
done
echo "No VM reached RUNNING within ${WAIT_SECS}s (quota or capacity?). Deleting MIG $MIG_NAME." >&2
gcloud compute instance-groups managed delete "$MIG_NAME" --project "$PROJECT" --zone "$ZONE" --quiet
exit 1
