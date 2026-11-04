#!/usr/bin/env bash
# One-time network setup. The org policy denies external IPs, so VMs have none: Google APIs are
# reached through Private Google Access and everything else (GitHub, Hugging Face, apt) through
# Cloud NAT, which allows outbound connections only. Safe to rerun. Cloud NAT bills by the hour
# and per GB; scripts/gcp/teardown.sh removes it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"
ROUTER_NAME="${ROUTER_NAME:-tinyllm-router}"
NAT_NAME="${NAT_NAME:-tinyllm-nat}"

gcloud compute networks subnets update default --project "$PROJECT" --region "$REGION" \
  --enable-private-ip-google-access

if [ -z "$(gcloud compute routers list --project "$PROJECT" --filter "name='$ROUTER_NAME'" --format 'value(name)')" ]; then
  gcloud compute routers create "$ROUTER_NAME" --project "$PROJECT" --region "$REGION" --network default
fi
if [ -z "$(gcloud compute routers nats list --project "$PROJECT" --region "$REGION" --router "$ROUTER_NAME" --format 'value(name)')" ]; then
  gcloud compute routers nats create "$NAT_NAME" --project "$PROJECT" --region "$REGION" \
    --router "$ROUTER_NAME" --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
fi
echo "Network ready: VMs without external IPs can reach Google APIs and the internet."
