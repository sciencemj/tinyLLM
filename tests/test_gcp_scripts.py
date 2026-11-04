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
