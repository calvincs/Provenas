"""Real-life example 3 — the DECISION SIDECAR: your services ask over HTTP, answers carry proofs.

The way OPA runs next to an app, Provenas can too: `provenas-serve mykb.db` exposes the exact
fabric on localhost, and any service in any language POSTs a JSON action and gets back the
answer plus its derivation (which you can log, display, or attach to an audit ticket).

This script plays both roles in one process — it starts the server, then queries it exactly
the way your app would (plain HTTP, here via urllib; curl works identically).

Runs offline, no LLM, no third-party packages:   python examples/service_sidecar.py
"""
import json
import threading
import urllib.request

from provenas.server import serve

TOKEN = "demo-token"


def client(method_path, data=None):
    """What YOUR service does — any language, any HTTP client."""
    path = method_path.split(" ", 1)[1]
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(data).encode() if data else None,
                                 headers={"Authorization": "Bearer " + TOKEN})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    print(f"\n>>> {method_path}" + (f"   {json.dumps(data)}" if data else ""))
    print("<<< " + json.dumps(body, indent=2).replace("\n", "\n    "))
    return body


# ---------------------------------------------------------------- start the sidecar
store, server = serve(":memory:", port=0, token=TOKEN, with_llm=False)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d" % server.server_address[1]
print(f"sidecar up at {BASE}  (in production: `provenas-serve mykb.db 8642`)")

# seed a little deployment policy: facts go in over the API; the RULE is installed on the
# store at setup time (rules enter through the admission gate, not raw HTTP)
from provenas.infer import Rule                                                  # noqa: E402
store.add_rule(Rule([("?u", "has_role", "?r"), ("?r", "grants", "?p"), ("?a", "requires", "?p")],
                    ("?u", "can", "?a"), "can"), source="policy")
for s, r, o in [("alice", "has_role", "sre"), ("sre", "grants", "deploy_key"),
                ("prod", "requires", "deploy_key"), ("bo", "has_role", "engineer")]:
    client("POST /action", {"action": "assert", "triple": [s, r, o]})

# ---------------------------------------------------------------- what your services do
client("GET /health")
r = client("POST /action", {"action": "check", "triple": ["alice", "can", "prod"]})
assert r["answer"] is True and "rule[can]" in r["trace"]    # the proof rides along in "trace"

r = client("POST /action", {"action": "check", "triple": ["bo", "can", "prod"]})
assert r["answer"] is False                                 # not derivable -> a clean, honest "no"

client("POST /action", {"action": "query", "pattern": ["?x", "can", "prod"]})
client("GET /facts?contains=alice")
client("GET /rules")

server.shutdown()
store.close()
print("\nDone. One binary, one SQLite file, exact answers with proofs over plain HTTP.")
