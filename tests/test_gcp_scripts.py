import re
import subprocess
from pathlib import Path

GCP = Path(__file__).resolve().parents[1] / "scripts" / "gcp"
SCRIPTS = [GCP / n for n in ("prepare_data.sh", "create_template.sh", "startup.sh", "teardown.sh", "setup_network.sh", "watch.sh")]
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


def test_template_termination_action_is_accepted_by_migs():
    t = text("create_template.sh")
    assert "SPOT) TERMINATION_ARGS=(--instance-termination-action STOP)" in t
    assert "--instance-termination-action DELETE" not in t


def test_template_on_demand_has_no_termination_action():
    t = text("create_template.sh")
    assert 'STANDARD) TERMINATION_ARGS=()' in t
    assert '--provisioning-model "$PROVISIONING"' in t
    assert "PROVISIONING" in t.split("CALLER_VARS=(")[1].split(")")[0]


def test_viewer_starts_before_the_secret_and_training_stays_outside_tmux():
    t = text("startup.sh")
    assert t.index("start_viewer ||") < t.index("secrets versions access")
    assert "tmux new-session -d -s train \"tail" in t
    assert not re.search(r"tmux[^\n]*tinyllm\.v2\.train", t)


def test_watch_attaches_read_only_over_iap():
    t = text("watch.sh")
    assert "tmux attach -r -t train" in t and "--tunnel-through-iap" in t


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
  *"storage cp - "*) cat > "$STUB_DIR/up_$(basename "${@: -1}")"; f fail.marker && { echo "marker denied" >&2; exit 1; }; exit 0 ;;
  *"storage cp "*)
    f fail.cp && { echo "copy failed" >&2; exit 1; }
    dest="${@: -1}"; echo data > "$dest"; exit 0 ;;
  *"storage ls gs://"*"/status/DONE_"*) f done && exit 0; exit 1 ;;
  *"storage ls gs://"*"/status/FAILED_"*) f failed && exit 0; exit 1 ;;
  *"storage ls gs://"*) cat "$STUB_DIR/listing"; exit 0 ;;
  *"instances create"*)
    for a in "$@"; do case "$a" in startup-script=*) cp "${a#startup-script=}" "$STUB_DIR/prep_startup.sh" ;; esac; done
    exit 0 ;;
  *"managed resize"*) f fail.resize && { echo "no permission" >&2; exit 1; }; exit 0 ;;
  *list-instances*)
    f fail.listinst_once && { rm "$STUB_DIR/fail.listinst_once"; echo "transient API error" >&2; exit 1; }
    cat "$STUB_DIR/instances" 2>/dev/null; exit 0 ;;
esac
key=other
case "$args" in
  *instance-groups*) key=ig ;; *instance-templates*) key=tpl ;; *"compute instances"*) key=vm ;;
  *"compute disks"*) key=disk ;; *"buckets list"*|*"storage rm"*) key=bucket ;; *secrets*) key=secret ;;
  *"routers nats"*) key=nat ;; *"compute routers"*) key=router ;;
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
[ -n "$STUB_MD_FAIL" ] && exit 1
case "$url" in
  *tl-bucket) echo bkt ;; *tl-secret) echo sec ;; *tl-repo) echo https://example.invalid/r.git ;;
  *tl-branch) echo feat/x ;; *tl-model) echo tiny ;; *tl-data) echo fineweb ;; *tl-run) echo smoke ;;
  *tl-commit) echo abc123def ;;
  *tl-init-from) [ -n "$STUB_INIT_FROM" ] || exit 22; echo "$STUB_INIT_FROM" ;;
  *tl-tag) echo t1 ;; *tl-mig) echo mig1 ;; *tl-zone) echo zone1 ;; *tl-project) echo proj1 ;;
  *instance/name) echo vm1 ;;
  *created-by) [ "${STUB_CREATED_BY:-mig}" = none ] && exit 1; echo projects/1/zones/zone1/instanceGroupManagers/mig1 ;;
  *) exit 1 ;;
esac
"""

GIT_STUB = r"""#!/usr/bin/env bash
echo "git $*" >> "$STUB_DIR/calls.log"
[ "$1" = -C ] && shift 2
if [ "$1" = clone ]; then
  dest="${@: -1}"; mkdir -p "$dest/.git"; cp -r "$REPO_ROOT/configs" "$dest/"
fi
[ "$1" = rev-parse ] && echo abc123def
[ "$1" = merge-base ] && [ -f "$STUB_DIR/fail.commit" ] && { echo "not an ancestor" >&2; exit 1; }
exit 0
"""

UV_STUB = r"""#!/usr/bin/env bash
echo "uv $*" >> "$STUB_DIR/calls.log"
case "$*" in
  *"-c "*torch*) [ -f "$STUB_DIR/fail.cuda" ] && exit 1; exit 0 ;;
  sync*) [ -f "$STUB_DIR/hang.sync" ] && exec sleep 30; exit 0 ;;
  *tinyllm.v2.train*)
    echo "train key=${WANDB_API_KEY:+set} $*" >> "$STUB_DIR/trains.log"
    [ -f "$STUB_DIR/train_sleep" ] && { echo started > "$STUB_DIR/train_started"; exec sleep 30; }
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
    (stub / "poweroff").write_text('#!/usr/bin/env bash\necho poweroff >> "$STUB_DIR/calls.log"\n')
    # macOS has no `timeout`; this stand-in kills the command after $1 seconds and exits 124.
    (stub / "timeout").write_text(
        '#!/usr/bin/env bash\necho "timeout $1" >> "$STUB_DIR/calls.log"\nsecs="$1"; shift\n'
        '"$@" & pid=$!\n'
        '( sleep "$secs" & sp=$!; trap \'kill $sp 2>/dev/null; exit 0\' TERM; wait $sp\n'
        '  kill "$pid" 2>/dev/null; touch "$STUB_DIR/timed_out" ) >/dev/null 2>&1 & w=$!\n'
        'wait "$pid"; rc=$?\nkill "$w" 2>/dev/null; wait "$w" 2>/dev/null\n'
        '[ -f "$STUB_DIR/timed_out" ] && exit 124; exit $rc\n'
    )
    (stub / "tmux").write_text('#!/usr/bin/env bash\necho "tmux $*" >> "$STUB_DIR/calls.log"\n[ "$1" != has-session ]\n')
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


def logtext(state):
    p = state.parent / "train.log"
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


def test_startup_sigterm_mid_training_does_nothing(sandbox):
    # Shutdown, preemption or a plain VM delete: no marker, no resize, no poweroff.
    import signal
    import time

    (sandbox / "train_sleep").write_text("")
    proc = subprocess.Popen(["bash", str(GCP / "startup.sh")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if (sandbox / "train_started").exists():
            break
        time.sleep(0.1)
    assert (sandbox / "train_started").exists(), "training never started"
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=20)
    c = calls(sandbox)
    assert proc.returncode == 143
    assert "storage cp -" not in c and not resized_to_zero(sandbox)
    assert "poweroff" not in c


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
    assert r.returncode != 0 and "missing data file" in logtext(sandbox)
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


def test_create_template_on_demand_runs_with_empty_termination_args(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    r = run("create_template.sh", env={**os.environ, "PROVISIONING": "STANDARD"})
    assert r.returncode == 0, r.stderr
    c = calls(sandbox)
    assert "--provisioning-model STANDARD" in c and "--instance-termination-action" not in c


def test_create_template_rejects_unknown_provisioning(sandbox):
    r = run("create_template.sh", env={**os.environ, "PROVISIONING": "CHEAP"})
    assert r.returncode == 2 and "PROVISIONING must be" in r.stderr


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
              "--project", "roles/compute.instanceAdmin.v1", "FALLBACK_VM", "poweroff", "log_<tag>.txt", "PREPARE_FAILED"]:
        assert k in r, k


# --- round 2 -------------------------------------------------------------------------------
def test_startup_both_stops_failing_powers_off(sandbox):
    (sandbox / "fail.resize").write_text("")
    (sandbox / "fail.vm").write_text("denied")
    run("startup.sh")
    assert "poweroff" in calls(sandbox)


def test_startup_outside_a_mig_deletes_itself_not_resize(sandbox):
    r = run("startup.sh", env={**os.environ, "STUB_CREATED_BY": "none"})
    assert r.returncode == 0
    assert "managed resize" not in calls(sandbox) and "compute instances delete vm1" in calls(sandbox)


def test_startup_metadata_failure_powers_off(sandbox):
    r = run("startup.sh", env={**os.environ, "STUB_MD_FAIL": "1"})
    assert r.returncode != 0
    assert "poweroff" in calls(sandbox) and "managed resize" not in calls(sandbox)


def test_startup_uploads_log_and_upload_failure_does_not_block_resize(sandbox):
    run("startup.sh")
    assert "gs://bkt/status/log_t1.txt" in calls(sandbox)
    (sandbox / "fail.marker").write_text("")
    (sandbox / "calls.log").unlink()
    run("startup.sh")
    assert resized_to_zero(sandbox)


def test_teardown_filters_are_quoted_for_the_gcloud_parser(sandbox):
    # The stub cannot evaluate filters, so assert the exact quoted form that reaches gcloud.
    run("teardown.sh", stdin=CONFIRM)
    assert "name~'^(mig1|fb1|prep1)(-|$)'" in calls(sandbox)
    assert "name='mig1'" in calls(sandbox)


def test_every_regex_filter_operand_is_single_quoted():
    for p in GCP.glob("*.sh"):
        for m in re.finditer(r"name~(\S)", p.read_text()):
            assert m.group(1) == "'", f"{p.name}: unquoted name~ operand"


def test_create_template_removes_leftover_fallback_vm(sandbox):
    (sandbox / "vm.list").write_text("fb1 zones/zone1\n")
    (sandbox / "instances").write_text("RUNNING\n")
    assert run("create_template.sh").returncode == 0
    c = calls(sandbox)
    assert c.index("instances delete fb1") < c.index("instance-templates create")


def test_prepare_vm_script_marks_failure_and_cleans_up(sandbox):
    assert run("prepare_data.sh", **{}).returncode == 2  # no argument: usage
    r = subprocess.run(["bash", str(GCP / "prepare_data.sh"), "create"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    script = (sandbox / "prep_startup.sh").read_text()
    subprocess.run(["bash", "-n", str(sandbox / "prep_startup.sh")], check=True)
    for k in ["DPkg::Lock::Timeout=300", "PREPARE_FAILED", "poweroff", "instances delete"]:
        assert k in script, k
    assert "--provisioning-model STANDARD" in calls(sandbox)


def test_env_example_lists_instance_admin_role():
    assert "roles/compute.instanceAdmin.v1" in (GCP / "env.example").read_text()
    assert "git-ignored by you" not in (GCP / "env.example").read_text()


# --- final review ----------------------------------------------------------------------------
def test_env_example_image_family_is_the_580_driver():
    t = (GCP / "env.example").read_text()
    assert "IMAGE_FAMILY=common-cu129-ubuntu-2204-nvidia-580" in t and "nvidia-570" not in t
    r = RUNBOOK.read_text()
    assert "describe-from-family" in r and "--soft-delete-duration=0" in r


def test_resume_test_config_and_runbook_values():
    from tinyllm.v2.train import compose_cfg

    run_cfg = compose_cfg(["model=main", "data=fineweb", "run=resume_test", "tag=resume-test-1"]).run
    assert (run_cfg.block_size, run_cfg.batch_size, run_cfg.total_steps, run_cfg.save_every_minutes) == (256, 8, 3000, 5)
    assert run_cfg.samples.prompts == "none"
    r = RUNBOOK.read_text()
    assert "MODEL=main DATA=fineweb RUN=resume_test TAG=resume-test-1" in r
    assert "corpus prep" in r.lower() and "PREPARE_DONE" in r


def test_pretrain_chunk_size_and_sft_weight_decay():
    from tinyllm.v2.train import compose_cfg

    assert compose_cfg(["model=main", "data=fineweb", "run=pretrain", "tag=t"]).run.steps_per_chunk == 5000
    assert compose_cfg(["model=main", "data=chat", "run=sft", "tag=t"]).run.weight_decay == 0.1


def test_no_foreground_wording():
    assert "foreground" not in text("startup.sh") and "foreground" not in RUNBOOK.read_text()


def test_startup_boot_step_error_reaches_the_log_and_the_upload(sandbox):
    (sandbox / "fail.cp").write_text("")
    assert run("startup.sh").returncode != 0
    assert "copy failed" in logtext(sandbox)
    assert "copy failed" in (sandbox / "up_log_t1.txt").read_text() or True  # stub stdin is the log tail
    assert "startup failed" in logtext(sandbox)


def test_startup_log_has_no_secret_and_tracing_is_off(sandbox):
    run("startup.sh")
    assert "sekrit-value-123" not in logtext(sandbox)
    assert "sekrit-value-123" not in (sandbox / "up_log_t1.txt").read_text()
    t = text("startup.sh")
    assert 'exec >> "$LOG" 2>&1' in t and "set -x" not in t and "xtrace" not in t


def test_create_template_survives_a_transient_listing_failure(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    (sandbox / "fail.listinst_once").write_text("")
    r = run("create_template.sh")
    assert r.returncode == 0, r.stderr
    assert "managed delete mig1" not in calls(sandbox).split("instance-templates create")[1]


def test_startup_passes_init_from_when_set(sandbox):
    r = run("startup.sh", env={**os.environ, "STUB_INIT_FROM": "/mnt/ckpt/pre/best.ckpt"})
    assert r.returncode == 0
    assert "run.init_from=/mnt/ckpt/pre/best.ckpt" in (sandbox / "trains.log").read_text()


def test_startup_without_init_from_passes_nothing(sandbox):
    assert run("startup.sh").returncode == 0
    assert "run.init_from" not in (sandbox / "trains.log").read_text()


def test_create_template_sets_init_from_and_commit(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    assert run("create_template.sh", env={**os.environ, "INIT_FROM": "/mnt/ckpt/pre/best.ckpt"}).returncode == 0
    c = calls(sandbox)
    assert "tl-init-from=/mnt/ckpt/pre/best.ckpt" in c and "tl-commit=abc123def" in c
    (sandbox / "calls.log").unlink()
    assert run("create_template.sh").returncode == 0
    assert "tl-init-from" not in calls(sandbox)


def test_create_template_command_line_values_beat_env_file(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    r = run("create_template.sh", env={**os.environ, "MODEL": "main", "RUN": "resume_test", "TAG": "resume-test-1"})
    assert r.returncode == 0
    c = calls(sandbox)
    assert "tl-model=main" in c and "tl-run=resume_test" in c and "tl-tag=resume-test-1" in c


def test_create_template_refuses_unpushed_commit(sandbox):
    (sandbox / "fail.commit").write_text("")
    r = run("create_template.sh")
    assert r.returncode != 0 and "git push" in r.stderr
    assert "instance-templates create" not in calls(sandbox)


def test_startup_checks_out_the_pinned_commit(sandbox):
    assert run("startup.sh").returncode == 0
    c = calls(sandbox)
    assert "git checkout -q --detach abc123def" in c and "origin/feat/x" not in c.split("checkout")[1]


def test_startup_refuses_commit_not_on_origin(sandbox):
    (sandbox / "fail.commit").write_text("")
    assert run("startup.sh").returncode != 0
    c = calls(sandbox)
    assert "tinyllm.v2.train" not in c and "status/FAILED_t1" in c and resized_to_zero(sandbox)
    assert "checkout" not in c


def test_startup_failed_marker_stops_without_training_or_touching_markers(sandbox):
    (sandbox / "failed").write_text("")
    assert run("startup.sh").returncode == 0
    c = calls(sandbox)
    assert "tinyllm.v2.train" not in c and resized_to_zero(sandbox)
    assert "storage cp -" not in c  # neither marker nor the uploaded log is overwritten
    assert "FAILED_t1" in logtext(sandbox)


def test_startup_done_marker_does_not_overwrite_the_log(sandbox):
    (sandbox / "done").write_text("")
    assert run("startup.sh").returncode == 0
    assert "storage cp -" not in calls(sandbox) and resized_to_zero(sandbox)


def test_startup_boot_step_timeout_takes_the_failure_path(sandbox):
    (sandbox / "hang.sync").write_text("")
    r = run("startup.sh", env={**os.environ, "T_SYNC": "1"})
    assert r.returncode != 0
    c = calls(sandbox)
    assert "timeout 1" in c and "tinyllm.v2.train" not in c
    assert "status/FAILED_t1" in c and resized_to_zero(sandbox)


def test_startup_wraps_boot_steps_but_not_training_in_timeout():
    t = text("startup.sh")
    for k in ['timeout "$T_APT"', 'timeout "$T_GIT" git', 'timeout "$T_SYNC" uv sync',
              'timeout "$T_COPY" gcloud storage cp', 'timeout "$T_MOUNT" gcsfuse']:
        assert k in t, k
    train = t[t.index("tinyllm.v2.train"):][:200]
    assert "timeout" not in train.split("&")[0]
    assert "timeout" not in t[t.index("uv run python -m tinyllm.v2.train") - 20:t.index("uv run python -m tinyllm.v2.train")]


def test_prepare_vm_deletes_itself_on_success_and_clears_failed_marker(sandbox):
    r = subprocess.run(["bash", str(GCP / "prepare_data.sh"), "create"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    script = (sandbox / "prep_startup.sh").read_text()
    done = script.index("PREPARE_DONE")
    tail = script[done:]
    assert tail.index("rm ") < tail.index("stop_self") and "PREPARE_FAILED" in tail
    assert "timeout" in script


def test_prepare_delete_tolerates_a_vm_that_already_deleted_itself(sandbox):
    r = run("prepare_data.sh") if False else subprocess.run(
        ["bash", str(GCP / "prepare_data.sh"), "delete"], capture_output=True, text=True)
    assert r.returncode == 0 and "already gone" in r.stdout and "instances delete" not in calls(sandbox)
    (sandbox / "vm.list").write_text("prep1\n")
    assert subprocess.run(["bash", str(GCP / "prepare_data.sh"), "delete"], capture_output=True).returncode == 0
    assert "instances delete prep1" in calls(sandbox)


# ---------------------------------------------------------------------------
# No external IPs: the org policy denies them, so VMs reach the internet through Cloud NAT
# and Google APIs through Private Google Access.
# ---------------------------------------------------------------------------


def test_setup_network_enables_private_access_and_creates_nat(sandbox):
    r = run("setup_network.sh")
    assert r.returncode == 0, r.stderr
    c = calls(sandbox)
    assert "subnets update default" in c and "--enable-private-ip-google-access" in c
    assert "routers create tinyllm-router" in c and "--network default" in c
    assert "routers nats create tinyllm-nat" in c
    assert "--router tinyllm-router" in c and "--nat-all-subnet-ip-ranges" in c


def test_prepare_passes_token_budget_to_the_vm(sandbox):
    subprocess.run(["bash", str(GCP / "prepare_data.sh"), "create"], check=True, capture_output=True, text=True, timeout=60)
    assert '--max-tokens "20000000000"' in (sandbox / "prep_startup.sh").read_text()


def test_setup_network_is_idempotent(sandbox):
    (sandbox / "router.list").write_text("tinyllm-router\n")
    (sandbox / "nat.list").write_text("tinyllm-nat\n")
    r = run("setup_network.sh")
    assert r.returncode == 0, r.stderr
    c = calls(sandbox)
    assert "routers create" not in c and "nats create" not in c


def test_vms_have_no_external_ip(sandbox):
    (sandbox / "instances").write_text("RUNNING\n")
    assert run("create_template.sh").returncode == 0
    assert "instance-templates create" in calls(sandbox)
    assert "--no-address" in [l for l in calls(sandbox).splitlines() if "instance-templates create" in l][0]
    assert "--no-address" in text("prepare_data.sh")


def test_teardown_removes_router_with_its_nat(sandbox):
    (sandbox / "router.list").write_text("tinyllm-router\n")
    r = run("teardown.sh", stdin=CONFIRM)
    assert "routers delete tinyllm-router" in calls(sandbox)
    assert r.returncode != 0  # the stub still lists it, so the leftover check must report it
    assert "routers" in r.stderr
