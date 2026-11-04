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
LOG=/var/log/tinyllm-prepare.log
exec >> "\$LOG" 2>&1
# Nothing here may leave a billing VM idle: success and failure both end by deleting this VM
# (poweroff if that fails). Failure first leaves a PREPARE_FAILED marker with the log tail.
stop_self() {
  gcloud compute instances delete "\$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)" --zone "$ZONE" --project "$PROJECT" --quiet || poweroff
}
fail() {
  set +e
  tail -n 2000 "\$LOG" | gcloud storage cp - "gs://$BUCKET/data/v2/PREPARE_FAILED"
  stop_self
}
trap 'rc=\$?; [ "\$rc" -eq 0 ] || fail' EXIT
APT=(apt-get -o DPkg::Lock::Timeout=300 -y)
timeout 900 "\${APT[@]}" update
timeout 900 "\${APT[@]}" install git curl
command -v uv >/dev/null || timeout 300 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
export PATH="\$HOME/.local/bin:\$PATH"
[ -d /opt/tinyllm ] || timeout 600 git clone --branch "$BRANCH" --single-branch "$REPO_URL" /opt/tinyllm
cd /opt/tinyllm
timeout 1800 uv sync
uv run python -m tinyllm.v2.prepare --out data/v2
gcloud storage cp --no-clobber data/v2/* "gs://$BUCKET/data/v2/"
echo done | gcloud storage cp - "gs://$BUCKET/data/v2/PREPARE_DONE"
# Done: drop a stale failure marker from an earlier attempt, then stop billing right away.
set +e
gcloud storage rm "gs://$BUCKET/data/v2/PREPARE_FAILED" 2>/dev/null
stop_self
exit 0
VMEOF
    # STANDARD provisioning, 64GB RAM (>= 32GB needed), no GPU.
    gcloud compute instances create "$PREPARE_VM" \
      --project "$PROJECT" --zone "$ZONE" \
      --machine-type n2-highmem-8 \
      --provisioning-model STANDARD \
      --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
      --boot-disk-size 500GB --boot-disk-type pd-ssd \
      --no-address \
      --scopes cloud-platform --labels app=tinyllm-v2 \
      ${SERVICE_ACCOUNT:+--service-account "$SERVICE_ACCOUNT"} \
      --metadata-from-file startup-script="$startup"
    echo "Created $PREPARE_VM. Wait for gs://$BUCKET/data/v2/PREPARE_DONE (or PREPARE_FAILED). The VM deletes itself in both cases; '$0 delete' removes it if it is still there."
    ;;
  delete)
    # The VM normally deleted itself already; deleting a missing VM would fail.
    if [ -n "$(gcloud compute instances list --project "$PROJECT" --filter "name='$PREPARE_VM'" --format 'value(name)')" ]; then
      gcloud compute instances delete "$PREPARE_VM" --project "$PROJECT" --zone "$ZONE" --quiet
    else
      echo "$PREPARE_VM is already gone"
    fi
    ;;
  *)
    echo "usage: $0 create|delete" >&2
    exit 2
    ;;
esac
