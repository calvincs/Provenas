"""server — the exact-answer fabric as a local HTTP service (pure stdlib).

Run Provenas the way OPA runs: a sidecar your services query for decisions, every answer exact
and carrying its derivation. The LLM layer is optional — /action needs no model at all.

  GET  /health                 {ok, facts, rules, llm}
  GET  /facts[?contains=sub]   the stored facts
  GET  /rules                  the active rules
  GET  /schema                 declared relations + strict flag
  POST /action {"action":"query|check|assert", ...}    exact answer + proof trace
  POST /ask    {"q": "natural language question"}      translate (needs a model) -> exact answer

Auth: set PROVENAS_API_TOKEN to require  Authorization: Bearer <token>  on every request.
Status codes are honest HTTP: malformed/unknown action -> 400, bad token -> 401, no model for /ask -> 503,
model failed mid-translate -> 502. A 200 always carries a real decision.
Run:  provenas-serve [kb.db] [port]      (default kb provenas.db, port 8642 / $PROVENAS_PORT)
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from provenas import qa
from provenas.llm import LLM
from provenas.store import Store


def make_handler(store, llm, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                      # quiet stdout; the store keeps the audit log
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self):
            if not token or self.headers.get("Authorization") == "Bearer " + token:
                return True
            self._send(401, {"error": "unauthorized"})
            return False

        def do_GET(self):
            if not self._auth():
                return
            u = urlparse(self.path)
            with store.lock:
                if u.path == "/health":
                    t, r = store.counts()
                    self._send(200, {"ok": True, "facts": t, "rules": r,
                                     "llm": bool(llm and llm.available)})
                elif u.path == "/facts":
                    sub = parse_qs(u.query).get("contains", [""])[0]
                    self._send(200, {"facts": [list(t) for t in sorted(store.triples())
                                               if sub in " ".join(t)]})
                elif u.path == "/rules":
                    self._send(200, {"rules": [{"name": r.name, "head": list(r.head),
                                                "body": [list(a) for a in r.body]}
                                               for r in store.rules()]})
                elif u.path == "/schema":
                    self._send(200, {"schema": store.schema(),
                                     "strict": store.get_meta("strict", "0") == "1"})
                else:
                    self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError):
                return self._send(400, {"error": "request body must be JSON"})
            with store.lock:
                if self.path == "/action":
                    try:
                        r = qa.run_action(store, payload)
                        # a malformed/unknown action is a client error, not a decision
                        self._send(400 if r["kind"] == "error" else 200, r)
                    except Exception as e:
                        self._send(400, {"error": str(e)})
                elif self.path == "/ask":
                    if not (llm and llm.available):
                        return self._send(503, {"error": "no model configured/reachable; use /action"})
                    try:
                        action = llm.translate(str(payload.get("q", "")), qa.context_from_store(store))
                        r = qa.run_action(store, action)
                        r["action"] = action
                        self._send(502 if r["kind"] == "error" else 200, r)
                    except Exception as e:
                        self._send(502, {"error": str(e)})
                else:
                    self._send(404, {"error": "not found"})

    return Handler


def serve(kb="provenas.db", port=8642, host="127.0.0.1", token=None, with_llm=True):
    """Build (store, server) — caller runs server.serve_forever(). Split out for tests."""
    store = Store(kb)
    llm = LLM() if with_llm else None
    if llm:
        llm.ping()
    server = ThreadingHTTPServer((host, port), make_handler(store, llm, token))
    return store, server


USAGE = """usage: provenas-serve [kb.db] [port]

Serves the knowledge base as a local HTTP decision service (default kb
provenas.db, port 8642 / $PROVENAS_PORT, bind $PROVENAS_BIND or 0.0.0.0).
Set PROVENAS_API_TOKEN to require Bearer auth. Endpoints: GET /health
/facts /rules /schema, POST /action /ask.
"""


def main():
    kb = sys.argv[1] if len(sys.argv) > 1 else "provenas.db"
    if kb in ("-h", "--help") or kb.startswith("-"):
        print(USAGE, end="")
        return
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PROVENAS_PORT", "8642"))
    host = os.environ.get("PROVENAS_BIND", "0.0.0.0")
    token = os.environ.get("PROVENAS_API_TOKEN")
    store, server = serve(kb, port, host, token)
    print(f"provenas service @ http://{host}:{port}   (kb: {kb}, auth: {'bearer' if token else 'OFF'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
