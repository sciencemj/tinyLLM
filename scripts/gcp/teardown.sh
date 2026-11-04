#!/usr/bin/env bash
# DESTRUCTIVE. Deletes every billable resource of this project's run, then asks you to check Billing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ENV_FILE:-$HERE/env}"

cat <<MSG
This will PERMANENTLY remove from project $PROJECT:
  - managed instance group  $MIG_NAME ($ZONE) and its VM
  - instance template       $TEMPLATE_NAME
  - prepare VM              $PREPARE_VM (if present)
  - disks named             ${MIG_NAME}-* and $PREPARE_VM (if present)
  - bucket gs://$BUCKET and ALL its contents (data, checkpoints)
  - secret                  $SECRET_NAME
MSG
read -r -p "Type DELETE $PROJECT to continue: " answer
if [ "$answer" != "DELETE $PROJECT" ]; then
  echo "Aborted; nothing removed."
  exit 1
fi

# '|| true' on each step so one already-missing resource does not stop the rest.
gcloud compute instance-groups managed delete "$MIG_NAME" --project "$PROJECT" --zone "$ZONE" --quiet || true
gcloud compute instance-templates delete "$TEMPLATE_NAME" --project "$PROJECT" --quiet || true
gcloud compute instances delete "$PREPARE_VM" --project "$PROJECT" --zone "$ZONE" --quiet || true
for d in $(gcloud compute disks list --project "$PROJECT" --filter "zone:$ZONE AND (name~^${MIG_NAME}- OR name=$PREPARE_VM)" --format 'value(name)'); do
  gcloud compute disks delete "$d" --project "$PROJECT" --zone "$ZONE" --quiet || true
done
gcloud storage rm -r "gs://$BUCKET" --project "$PROJECT" || true
gcloud secrets delete "$SECRET_NAME" --project "$PROJECT" --quiet || true

echo
echo "Done. Now open Billing in the console and confirm no GPU, disk or storage charges are accruing."
