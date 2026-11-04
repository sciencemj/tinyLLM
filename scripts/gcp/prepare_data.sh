#!/usr/bin/env bash
# Corpus prep on a STANDARD (non-spot) CPU VM. Usage: prepare_data.sh create | delete
# A failure here restarts from zero and costs hours, so this VM is never spot.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"

case "${1:-}" in
  create)
    startup="$(mktemp)"
    trap 'rm -f "$startup"' EXIT
    cat > "$startup" <<VMEOF
#!/usr/bin/env bash
set -euo pipefail
export HOME=/root
apt-get update -y && apt-get install -y git curl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
[ -d /opt/tinyllm ] || git clone --branch "$BRANCH" --single-branch "$REPO_URL" /opt/tinyllm
cd /opt/tinyllm
uv sync
uv run python -m tinyllm.v2.prepare --out data/v2
gcloud storage cp --no-clobber data/v2/* "gs://$BUCKET/data/v2/"
echo done | gcloud storage cp - "gs://$BUCKET/data/v2/PREPARE_DONE"
VMEOF
    # STANDARD provisioning, 64GB RAM (>= 32GB needed), no GPU.
    gcloud compute instances create "$PREPARE_VM" \
      --project "$PROJECT" --zone "$ZONE" \
      --machine-type n2-highmem-8 \
      --provisioning-model STANDARD \
      --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
      --boot-disk-size 500GB --boot-disk-type pd-ssd \
      --scopes cloud-platform --labels app=tinyllm-v2 \
      ${SERVICE_ACCOUNT:+--service-account "$SERVICE_ACCOUNT"} \
      --metadata-from-file startup-script="$startup"
    echo "Created $PREPARE_VM. Wait for gs://$BUCKET/data/v2/PREPARE_DONE, then run: $0 delete"
    ;;
  delete)
    gcloud compute instances delete "$PREPARE_VM" --project "$PROJECT" --zone "$ZONE" --quiet
    ;;
  *)
    echo "usage: $0 create|delete" >&2
    exit 2
    ;;
esac
