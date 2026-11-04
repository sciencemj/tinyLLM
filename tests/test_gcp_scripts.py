import re
import subprocess
from pathlib import Path

GCP = Path(__file__).resolve().parents[1] / "scripts" / "gcp"
SCRIPTS = [GCP / n for n in ("prepare_data.sh", "create_template.sh", "startup.sh", "teardown.sh")]
RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "gcp-runbook.md"


def text(name):
    return (GCP / name).read_text()


def all_script_text():
    return "\n".join(p.read_text() for p in [*SCRIPTS, GCP / "env.example", RUNBOOK])


def test_scripts_parse():
    for s in SCRIPTS:
        subprocess.run(["bash", "-n", str(s)], check=True)


def test_no_secret_literals():
    assert not re.search(r"(hf_[A-Za-z0-9]{10,}|WANDB_API_KEY=\S+)", all_script_text())


def test_teardown_removes_all_billable_resources():
    t = text("teardown.sh")
    assert all(k in t for k in ["instance-groups", "instance-templates", "disks", "storage", "secrets"])


def test_teardown_requires_typed_confirmation():
    t = text("teardown.sh")
    assert "read " in t and "DELETE" in t


def test_gpu_type_mapping_and_spot():
    t = text("create_template.sh")
    assert "a3-highgpu-1g" in t and "g2-standard-8" in t
    assert "SPOT" in t and "t4" not in t.lower()


def test_startup_is_secret_safe_and_resumable():
    t = text("startup.sh")
    assert "set +x" in t and "secrets versions access" in t
    assert "uv sync" in t and "ckpt_dir=" in t and "/mnt/ckpt" in t
    assert "echo \"$WANDB" not in t


def test_secret_not_in_metadata():
    t = text("create_template.sh")
    assert "WANDB_API_KEY" not in t


def test_prepare_uses_standard_vm():
    t = text("prepare_data.sh")
    assert "--provisioning-model STANDARD" in t and "SPOT" not in t
    assert "tinyllm.v2.prepare" in t


def test_runbook_gates():
    r = RUNBOOK.read_text()
    for k in ["budget alert", "does not stop", "quota", "Billing", "unverified", "restarts from zero"]:
        assert k.lower() in r.lower(), k


# ---------------------------------------------------------------------------
# Behaviour tests. A stub `gcloud` (and friends) on PATH records every call; no real cloud
# command can run, because the real binaries are never reachable.
# ---------------------------------------------------------------------------
import os  # noqa: E402

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

GCLOUD_STUB = r"""#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/calls.log"
args="$*"
f() { [ -f "$STUB_DIR/$1" ]; }
case "$args" in
  *"secrets versions access"*) cat "$STUB_DIR/secret_value" 2>/dev/null; exit 0 ;;
  *"storage cp - "*) cat > /dev/null; f fail.marker && { echo "marker denied" >&2; exit 1; }; exit 0 ;;
  *"storage cp "*)
    f fail.cp && { echo "copy failed" >&2; exit 1; }
    dest="${@: -1}"; echo data > "$dest"; exit 0 ;;
  *"storage ls gs://"*"/status/"*) f done && exit 0; exit 1 ;;
  *"storage ls gs://"*) cat "$STUB_DIR/listing"; exit 0 ;;
  *"managed resize"*) f fail.resize && { echo "no permission" >&2; exit 1; }; exit 0 ;;
  *list-instances*) cat "$STUB_DIR/instances" 2>/dev/null; exit 0 ;;
esac
key=other
case "$args" in
  *instance-groups*) key=ig ;; *instance-templates*) key=tpl ;; *"compute instances"*) key=vm ;;
  *"compute disks"*) key=disk ;; *"buckets list"*|*"storage rm"*) key=bucket ;; *secrets*) key=secret ;;
esac
case "$args" in
  *" list "*) f "$key.list" && cat "$STUB_DIR/$key.list"; f "fail.list.$key" && { echo "list error" >&2; exit 1; }; exit 0 ;;
  *" delete "*|*"storage rm"*)
    f "fail.$key" && { cat "$STUB_DIR/fail.$key" >&2; exit 1; }; exit 0 ;;
esac
exit 0
"""

CURL_STUB = r"""#!/usr/bin/env bash
url="${@: -1}"
case "$url" in
  *tl-bucket) echo bkt ;; *tl-secret) echo sec ;; *tl-repo) echo https://example.invalid/r.git ;;
  *tl-branch) echo feat/x ;; *tl-model) echo tiny ;; *tl-data) echo fineweb ;; *tl-run) echo smoke ;;
  *tl-tag) echo t1 ;; *tl-mig) echo mig1 ;; *tl-zone) echo zone1 ;; *tl-project) echo proj1 ;;
  *instance/name) echo vm1 ;; *instance/preempted) echo "${STUB_PREEMPTED:-FALSE}" ;;
  *) exit 1 ;;
esac
"""

GIT_STUB = r"""#!/usr/bin/env bash
echo "git $*" >> "$STUB_DIR/calls.log"
if [ "$1" = clone ]; then
  dest="${@: -1}"; mkdir -p "$dest/.git"; cp -r "$REPO_ROOT/configs" "$dest/"
fi
exit 0
"""

UV_STUB = r"""#!/usr/bin/env bash
echo "uv $*" >> "$STUB_DIR/calls.log"
case "$*" in
  *"-c "*torch*) [ -f "$STUB_DIR/fail.cuda" ] && exit 1; exit 0 ;;
  *tinyllm.v2.train*)
    echo "train key=${WANDB_API_KEY:+set} $*" >> "$STUB_DIR/trains.log"
    [ -f "$STUB_DIR/train_rc" ] && exit "$(cat "$STUB_DIR/train_rc")"; exit 0 ;;
esac
exit 0
"""


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    stub, state = tmp_path / "bin", tmp_path / "state"
    stub.mkdir(); state.mkdir()
    bodies = {"gcloud": GCLOUD_STUB, "curl": CURL_STUB, "git": GIT_STUB, "uv": UV_STUB}
    for name, body in bodies.items():
        (stub / name).write_text(body)
    for name in ("nvidia-smi", "gcsfuse", "apt-get", "mountpoint"):
        (stub / name).write_text("#!/usr/bin/env bash\n[ -f \"$STUB_DIR/fail.$(basename $0)\" ] && exit 1; exit 0\n")
    for p in stub.iterdir():
        p.chmod(0o755)
    env_file = tmp_path / "env"
    env_file.write_text(
        "PROJECT=proj1\nREGION=r1\nZONE=zone1\nBUCKET=bkt\nGPU_TYPE=l4\nREPO_URL=https://example.invalid/r.git\n"
        "BRANCH=feat/x\nMODEL=tiny\nDATA=fineweb\nRUN=smoke\nTAG=t1\nSECRET_NAME=sec\nMIG_NAME=mig1\n"
        "TEMPLATE_NAME=tpl1\nPREPARE_VM=prep1\nFALLBACK_VM=fb1\nIMAGE_PROJECT=ip\nIMAGE_FAMILY=if\nBOOT_DISK_GB=300\n"
    )
    (state / "listing").write_text(
        "gs://bkt/data/v2/PREPARE_DONE\ngs://bkt/data/v2/fineweb_train.bin\ngs://bkt/data/v2/fineweb_val.bin\n"
        "gs://bkt/data/v2/tokenizer.json\ngs://bkt/data/v2/fineweb_train.meta.json\n"
    )
    (state / "secret_value").write_text("sekrit-value-123")
    monkeypatch.setenv("PATH", f"{stub}:{os.environ['PATH']}")
    monkeypatch.setenv("STUB_DIR", str(state))
    monkeypatch.setenv("REPO_ROOT", str(REPO))
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("APP_DIR", str(tmp_path / "app"))
    monkeypatch.setenv("CKPT_MNT", str(tmp_path / "ckpt"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "train.log"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RETRY_SLEEP", "0")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    (tmp_path / "home" / ".local" / "bin").mkdir(parents=True)
    (tmp_path / "home" / ".local" / "bin" / "uv").symlink_to(stub / "uv")
    return state


def run(script, stdin="", **kw):
    return subprocess.run(
        ["bash", str(GCP / script)], input=stdin, capture_output=True, text=True, timeout=60, **kw
    )


def calls(state):
    p = state / "calls.log"
    return p.read_text() if p.exists() else ""


def resized_to_zero(state):
    return "managed resize mig1 --size 0" in calls(state)


# --- C1: training exit must stop billing ---------------------------------------------------
def test_startup_success_writes_done_and_resizes_to_zero(sandbox):
    r = run("startup.sh")
    assert r.returncode == 0, r.stderr
    assert "storage cp - gs://bkt/status/DONE_t1" in calls(sandbox)
    assert resized_to_zero(sandbox)
    assert "ckpt_dir=" in (sandbox / "trains.log").read_text()


def test_startup_training_failure_retries_then_fails_and_resizes(sandbox):
    (sandbox / "train_rc").write_text("3")
    r = run("startup.sh", env={**os.environ, "MAX_TRIES": "2"})
    assert r.returncode != 0
    assert (sandbox / "trains.log").read_text().count("train key=") == 2
    assert "status/FAILED_t1" in calls(sandbox) and "cp - gs://bkt/status/DONE_t1" not in calls(sandbox)
    assert resized_to_zero(sandbox)


def test_startup_boot_abort_fails_and_resizes(sandbox):
    (sandbox / "fail.cp").write_text("")
    r = run("startup.sh")
    assert r.returncode != 0
    assert "status/FAILED_t1" in calls(sandbox)
    assert resized_to_zero(sandbox)
    assert "tinyllm.v2.train" not in calls(sandbox)  # I4: a failed copy never trains


def test_startup_done_marker_skips_training_and_resizes(sandbox):
    (sandbox / "done").write_text("")
    assert run("startup.sh").returncode == 0
    assert "tinyllm.v2.train" not in calls(sandbox)
    assert resized_to_zero(sandbox)


def test_startup_preemption_does_not_give_up(sandbox):
    (sandbox / "fail.cp").write_text("")
    r = run("startup.sh", env={**os.environ, "STUB_PREEMPTED": "TRUE"})
    assert r.returncode != 0 and not resized_to_zero(sandbox)


def test_startup_resize_failure_falls_back_to_deleting_self(sandbox):
    (sandbox / "fail.resize").write_text("")
    run("startup.sh")
    assert "compute instances delete vm1" in calls(sandbox)


# --- I4 / I6 / A2 --------------------------------------------------------------------------
def test_startup_requires_prepare_done(sandbox):
    (sandbox / "listing").write_text("gs://bkt/data/v2/fineweb_train.bin\n")
    assert run("startup.sh").returncode != 0
    assert "tinyllm.v2.train" not in calls(sandbox)


def test_startup_asserts_configured_data_files_exist(sandbox):
    (sandbox / "listing").write_text("gs://bkt/data/v2/PREPARE_DONE\ngs://bkt/data/v2/tokenizer.json\n")
    r = run("startup.sh")
    assert r.returncode != 0 and "missing data file" in r.stderr
    assert "tinyllm.v2.train" not in calls(sandbox)


@pytest.mark.parametrize("fail", ["fail.cuda", "fail.nvidia-smi"])
def test_startup_refuses_gpu_less_boot(sandbox, fail):
    (sandbox / fail).write_text("")
    assert run("startup.sh").returncode != 0
    assert "tinyllm.v2.train" not in calls(sandbox)
    assert resized_to_zero(sandbox)


def test_startup_empty_secret_fails(sandbox):
    (sandbox / "secret_value").write_text("")
    assert run("startup.sh").returncode != 0
    assert "tinyllm.v2.train" not in calls(sandbox)


def test_startup_key_reaches_training_but_never_leaks(sandbox, tmp_path):
    r = run("startup.sh")
    assert "key=set" in (sandbox / "trains.log").read_text()
    for blob in (r.stdout, r.stderr, calls(sandbox), (sandbox / "trains.log").read_text(),
                 (tmp_path / "train.log").read_text() if (tmp_path / "train.log").exists() else ""):
        assert "sekrit-value-123" not in blob


# --- I1 / I2 -------------------------------------------------------------------------------
def test_create_template_caller_gpu_type_beats_env_file(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    r = run("create_template.sh", env={**os.environ, "GPU_TYPE": "h100"})
    assert r.returncode == 0, r.stderr
    c = calls(sandbox)
    assert "--machine-type a3-highgpu-1g" in c and "g2-standard-8" not in c


def test_create_template_replaces_old_group_and_template(sandbox):
    (sandbox / "ig.list").write_text("mig1 zones/zone1\n")
    (sandbox / "tpl.list").write_text("tpl1\n")
    (sandbox / "instances").write_text("RUNNING\n")
    assert run("create_template.sh").returncode == 0
    c = calls(sandbox)
    assert c.index("managed delete mig1") < c.index("instance-templates delete tpl1") < c.index("instance-templates create")


def test_create_template_timeout_deletes_stuck_group(sandbox):
    (sandbox / "instances").write_text("PROVISIONING\n")
    r = run("create_template.sh", env={**os.environ, "WAIT_SECS": "2", "POLL_SECS": "1"})
    assert r.returncode != 0
    assert calls(sandbox).rstrip().splitlines()[-1].startswith("compute instance-groups managed delete mig1")


# --- I3: teardown --------------------------------------------------------------------------
CONFIRM = "DELETE proj1\n"


def test_teardown_wrong_confirmation_deletes_nothing(sandbox):
    assert run("teardown.sh", stdin="yes\n").returncode != 0
    assert " delete " not in calls(sandbox)


def test_teardown_clean_run_succeeds(sandbox):
    r = run("teardown.sh", stdin=CONFIRM)
    assert r.returncode == 0, r.stderr
    assert "Billing" in r.stdout


def test_teardown_tolerates_not_found(sandbox):
    (sandbox / "fail.ig").write_text("ERROR: The resource 'mig1' was not found")
    assert run("teardown.sh", stdin=CONFIRM).returncode == 0


def test_teardown_real_delete_failure_stops_before_data(sandbox):
    (sandbox / "ig.list").write_text("mig1 zones/zone1\n")
    (sandbox / "fail.ig").write_text("ERROR: permission denied")
    r = run("teardown.sh", stdin=CONFIRM)
    assert r.returncode != 0
    assert "storage rm" not in calls(sandbox) and "secrets delete" not in calls(sandbox)


def test_teardown_leftover_resource_is_nonzero(sandbox):
    (sandbox / "vm.list").write_text("fb1\n")  # still listed after deletion
    r = run("teardown.sh", stdin=CONFIRM)
    assert r.returncode != 0 and "INCOMPLETE" in r.stderr


def test_teardown_listing_failure_is_not_success(sandbox):
    (sandbox / "fail.list.disk").write_text("")
    assert run("teardown.sh", stdin=CONFIRM).returncode != 0


def test_teardown_covers_fallback_vm_in_any_zone(sandbox):
    (sandbox / "vm.list").write_text("fb1 zones/other-zone\n")
    run("teardown.sh", stdin=CONFIRM)
    assert "instances delete fb1" in calls(sandbox) and "--zone other-zone" in calls(sandbox)


# --- A1 / I7 -------------------------------------------------------------------------------
def test_env_file_is_git_ignored():
    r = subprocess.run(["git", "check-ignore", "scripts/gcp/env"], cwd=REPO, capture_output=True)
    assert r.returncode == 0


def test_runbook_prerequisites_and_stop_billing():
    r = RUNBOOK.read_text()
    for k in ["git push", "managed resize", "--size 0", "compute.googleapis.com",
              "secretmanager.googleapis.com", "storage.googleapis.com", "source scripts/gcp/env",
              "--project", "roles/compute.instanceAdmin.v1", "FALLBACK_VM"]:
        assert k in r, k
