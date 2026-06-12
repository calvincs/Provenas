"""The examples/ scripts are documentation that runs — so CI runs them.

Each must exit 0 (they contain their own asserts) and print its closing line.
All are pure stdlib and LLM-free by design.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

EXAMPLES = {
    "demo.py": "",
    "access_audit.py": "instant access auditor",
    "eligibility_screening.py": "changes gated by precedent",
    "service_sidecar.py": "exact answers with proofs over plain HTTP",
}


def _run(name):
    # Prepend the repo root so the examples run from a fresh clone,
    # whether or not the package has been pip-installed.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(ROOT / "examples" / name)],
                          capture_output=True, text=True, timeout=120, cwd=ROOT, env=env)


def test_examples_run_clean():
    for name, marker in EXAMPLES.items():
        r = _run(name)
        assert r.returncode == 0, f"{name} failed:\n{r.stdout}\n{r.stderr}"
        assert marker in r.stdout, f"{name} did not print its closing line"
