"""Real-life integration scenarios — the engine used the way a deployment would use it.

Distilled from a full manual test campaign (CLI smoke, RBAC org lifecycle, HTTP service
under concurrent load, sandbox adversarial battery, crash recovery, multi-handle
consistency). All LLM-free; the NL layer is exercised separately against a live model.
"""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from provenas import domains, qa
from provenas.infer import Rule
from provenas.server import serve
from provenas.store import Store
from provenas.toolsmith import validate_ast, run_sandboxed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- CLI entry points
def _run_cli(module, *args, stdin=""):
    env = dict(os.environ, PYTHONPATH=ROOT, PROVENAS_LLM_HOST="http://127.0.0.1:9")
    return subprocess.run([sys.executable, "-m", module, *args], input=stdin,
                          capture_output=True, text=True, timeout=60, env=env, cwd=ROOT)


def test_help_flag_prints_usage_and_creates_no_db(tmp_path):
    """`provenas --help` must show usage — NOT open a SQLite KB named '--help'."""
    for module in ("provenas", "provenas.server"):
        r = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True,
                           text=True, timeout=30, cwd=str(tmp_path),
                           env=dict(os.environ, PYTHONPATH=ROOT))
        assert r.returncode == 0 and r.stdout.startswith("usage:"), f"{module}: {r.stdout!r}"
    assert not list(tmp_path.iterdir()), "no files (especially no '--help' db) may be created"


def test_cli_scripted_session_offline(tmp_path):
    """A full scripted REPL session with no model: load, ask :why, strict mode, simplify."""
    kb = str(tmp_path / "t.db")
    script = (":load rbac\n:why alice can prod_deploy\n:assert bob has_role intern\n"
              ":declare has_role\n:strict on\n:assert x badrel y\n:strict off\n"
              ":simplify (x + 0) * (2 + 3)\n:kb\n:quit\n")
    r = _run_cli("provenas", kb, stdin=script)
    assert r.returncode == 0
    assert "rule[can-do]" in r.stdout                      # proof tree rendered
    assert "not declared in the schema" in r.stdout        # strict mode enforced
    assert "x * 5" in r.stdout                             # exact simplification


# --------------------------------------------------------------- RBAC org lifecycle
def _can(store, u, a):
    return qa.run_action(store, {"action": "check", "triple": [u, "can", a]})


def test_rbac_org_negation_guard_retract_restart(tmp_path):
    """A company access KB: inheritance, a suspension deny (negation), a tenure guard,
    suspension lifted/reimposed, and a process restart that must preserve every answer."""
    db = str(tmp_path / "acme.db")
    st = Store(db)
    for s, r, o in [("dana", "has_role", "cto"), ("dana", "tenure_days", "2400"),
                    ("fiona", "has_role", "engineer"), ("fiona", "tenure_days", "45"),
                    ("hana", "has_role", "engineer"), ("hana", "tenure_days", "900"),
                    ("hana", "suspended", "true"),
                    ("cto", "inherits", "engineer"), ("engineer", "grants", "deploy_perm"),
                    ("prod_deploy", "requires", "deploy_perm")]:
        st.assert_(s, r, o, source="hr")
    st.add_rule(Rule([("?u", "has_role", "?r"), ("?r", "inherits", "?r2")],
                     ("?u", "has_role", "?r2"), "role-inherit"))
    st.add_rule(Rule([("?u", "has_perm", "?p"), ("?a", "requires", "?p"),
                      ("?u", "~suspended", "true"),
                      ("?u", "tenure_days", "?t"), ("?t", ">=", "90")],
                     ("?u", "can", "?a"), "can-do-gated"))
    st.add_rule(Rule([("?u", "has_role", "?r"), ("?r", "grants", "?p")],
                     ("?u", "has_perm", "?p"), "perm-from-role"))

    assert _can(st, "dana", "prod_deploy")["answer"] is True       # via inheritance
    assert _can(st, "fiona", "prod_deploy")["answer"] is False     # tenure guard
    assert _can(st, "hana", "prod_deploy")["answer"] is False      # suspended (negation)
    st.retract("hana", "suspended", "true")
    assert _can(st, "hana", "prod_deploy")["answer"] is True       # closure recomputed
    st.assert_("hana", "suspended", "true")
    assert _can(st, "hana", "prod_deploy")["answer"] is False
    r = _can(st, "dana", "prod_deploy")
    assert "role-inherit" in r["trace"] and "given fact" in r["trace"]
    st.close()

    st2 = Store(db)                                                # restart
    assert _can(st2, "dana", "prod_deploy")["answer"] is True
    assert _can(st2, "hana", "prod_deploy")["answer"] is False
    st2.close()


# --------------------------------------------------------------- HTTP service
TOKEN = "test-token"


def _req(base, path, body=None, token=TOKEN, raw=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers),
                                    timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def service():
    store, server = serve(":memory:", 0, token=TOKEN, with_llm=False)   # port 0: OS picks
    domains.load(store, "rbac")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield store, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    store.close()


def test_http_malformed_actions_are_client_errors(service):
    """A decision service must answer malformed requests with 400, never 200/kind=error."""
    _, base = service
    for body in ({"action": "frobnicate"}, {"action": "check"},
                 {"action": "query", "pattern": ["?x", "can"]}):
        s, b = _req(base, "/action", body=body)
        assert s == 400 and b["kind"] == "error", (body, s, b)
    s, _ = _req(base, "/action", raw=b"{not json")
    assert s == 400
    s, _ = _req(base, "/health", token="wrong")
    assert s == 401
    s, b = _req(base, "/ask", body={"q": "anything"})
    assert s == 503                                        # no model configured


def test_http_concurrent_mixed_load_consistent(service):
    """Concurrent writers + decision readers: no 5xx, every write visible, closure correct."""
    _, base = service
    statuses, errors = [], []
    lock = threading.Lock()

    def client(cid):
        for i in range(10):
            try:
                if cid % 2 == 0:
                    s, _ = _req(base, "/action", body={"action": "assert",
                                                       "triple": [f"u{cid}_{i}", "has_role", "engineer"]})
                else:
                    s, b = _req(base, "/action", body={"action": "check",
                                                       "triple": ["alice", "can", "prod_deploy"]})
                    if s == 200 and b["answer"] is not True:
                        errors.append(f"decision flipped: {b}")
                with lock:
                    statuses.append(s)
            except Exception as e:                                  # pragma: no cover
                errors.append(repr(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(client, range(8)))
    assert not errors and all(s == 200 for s in statuses)
    s, b = _req(base, "/facts")
    present = {tuple(f) for f in b["facts"]}
    for cid in range(0, 8, 2):
        for i in range(10):
            assert (f"u{cid}_{i}", "has_role", "engineer") in present
    s, b = _req(base, "/action", body={"action": "check", "triple": ["u0_0", "can", "prod_deploy"]})
    assert b["answer"] is True                             # derived through rules, post-load


# --------------------------------------------------------------- sandbox extras
@pytest.mark.parametrize("label,src", [
    ("decorator", "@staticmethod\ndef f(x):\n    return x"),
    ("type-smuggle", "def f(x):\n    return type(x)"),
    ("vars", "def f(x):\n    return vars()"),
    ("breakpoint", "def f(x):\n    breakpoint()\n    return x"),
    ("comprehension-smuggle", "def f(x):\n    return [open(p) for p in x]"),
    ("fstring-attr", "def f(x):\n    return f'{x.__class__}'"),
    ("lambda-smuggle", "def f(x):\n    g = lambda: __import__('os')\n    return g()"),
])
def test_ast_gate_blocks_more_escapes(label, src):
    ok, reason = validate_ast(src, "f")
    assert not ok, f"{label} slipped through: {reason}"


def test_sandbox_contains_output_flood():
    ok, _ = run_sandboxed("def f(x):\n    return 'a' * (10**9)", "f", [[1]], timeout=5)
    assert not ok


# --------------------------------------------------------------- durability
def test_sigkill_mid_write_recovers(tmp_path):
    """kill -9 a writer mid-stream: WAL must recover, integrity clean, KB usable."""
    db = str(tmp_path / "crash.db")
    Store(db).assert_("seed", "rel", "x").close()
    writer = subprocess.Popen([sys.executable, "-c", (
        f"import sys; sys.path.insert(0, {ROOT!r})\n"
        f"from provenas.store import Store\n"
        f"st = Store({db!r})\n"
        f"i = 0\n"
        f"while True:\n"
        f"    st.assert_(f'w{{i}}', 'rel', f'v{{i}}')\n"
        f"    i += 1\n")])
    time.sleep(1.5)
    os.kill(writer.pid, signal.SIGKILL)
    writer.wait()
    raw = sqlite3.connect(db)
    assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    raw.close()
    st = Store(db)
    st.assert_("post", "rel", "crash")
    kg, _ = st.closure()
    assert ("post", "rel", "crash") in kg.triples
    st.close()


def test_two_handles_one_file_stay_consistent(tmp_path):
    """A CLI and a sidecar on the same KB file: revision-keyed closure cache must
    invalidate across handles for asserts, rule adds, and retractions."""
    db = str(tmp_path / "shared.db")
    a, b = Store(db), Store(db)
    a.assert_("alice", "has_role", "engineer")
    a.add_rule(Rule([("?u", "has_role", "engineer")], ("?u", "can", "deploy"), "eng-deploy"))
    assert ("alice", "can", "deploy") in b.closure()[0].triples
    b.assert_("bob", "has_role", "engineer")
    assert ("bob", "can", "deploy") in a.closure()[0].triples
    a.retract("alice", "has_role", "engineer")
    assert ("alice", "can", "deploy") not in b.closure()[0].triples
    a.close()
    b.close()


# --------------------------------------------------------------- LLM context grounding
def test_context_surfaces_entities_added_after_pack_load():
    """The translate-context must include relations/entities asserted AFTER a pack load,
    or the model cannot ground questions about them (found live: 'log_access')."""
    st = Store(":memory:")
    domains.load(st, "rbac")
    st.assert_("bob", "has_role", "auditor")
    st.assert_("auditor", "grants", "read_logs_perm")
    st.assert_("log_access", "requires", "read_logs_perm")
    ctx = qa.context_from_store(st)
    for tok in ("log_access", "auditor", "read_logs_perm"):
        assert tok in ctx, f"context missing post-pack token {tok!r}"
    st.close()
