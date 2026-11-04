#!/usr/bin/env bash
# Create the spot instance template and a size-1 MIG that recreates the VM after preemption.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"

case "${GPU_TYPE:?set GPU_TYPE=h100|l4}" in
  h100) MACHINE_TYPE=a3-highgpu-1g ;;
  l4)   MACHINE_TYPE=g2-standard-8 ;;
  *) echo "GPU_TYPE must be h100 or l4, got: $GPU_TYPE" >&2; exit 2 ;;
esac

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
  --scopes cloud-platform \
  ${SERVICE_ACCOUNT:+--service-account "$SERVICE_ACCOUNT"} \
  --metadata "tl-bucket=$BUCKET,tl-secret=$SECRET_NAME,tl-repo=$REPO_URL,tl-branch=$BRANCH,tl-model=$MODEL,tl-data=$DATA,tl-run=$RUN,tl-tag=$TAG" \
  --metadata-from-file startup-script="$HERE/startup.sh"

gcloud compute instance-groups managed create "$MIG_NAME" \
  --project "$PROJECT" --zone "$ZONE" \
  --template "$TEMPLATE_NAME" --size 1

# Unverified: a MIG with a3-highgpu-1g. If creation fails, see docs/gcp-runbook.md (single spot VM fallback).
echo "MIG $MIG_NAME created. Billing for the GPU has started."
