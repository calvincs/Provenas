"""llm — the interface tier: a chat model that translates NL <-> structured fabric actions.

The governance spine: the LLM PROPOSES structured operations; the fabric VERIFIES and executes them
exactly. The model never answers a computable question from its own weights — stochastic at the edges,
exact at the core.

Two backends (pure stdlib, no extra deps):
  - "ollama"  — a local OR remote Ollama server (default; native /api/chat, disables reasoning "think").
  - "openai"  — any OpenAI-compatible /v1/chat/completions endpoint (OpenAI, OpenRouter, vLLM, LM Studio,
                llama.cpp server, Together, Groq, ...), authenticated with a Bearer API key.

Configure via environment (or constructor args):
  PROVENAS_LLM_MODEL     model name            (default: qwen3.5:9b)
  PROVENAS_LLM_HOST      base URL              (default: http://localhost:11434; https://api.openai.com for openai)
  PROVENAS_LLM_BACKEND   "ollama" | "openai"   (default: openai if an API key is set, else ollama)
  PROVENAS_LLM_API_KEY   Bearer token          (its presence selects the openai backend by default)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLM:
    def __init__(self, model=None, host=None, backend=None, api_key=None, timeout=180):
        self.model = model or os.environ.get("PROVENAS_LLM_MODEL", "qwen3.5:9b")
        self.api_key = api_key or os.environ.get("PROVENAS_LLM_API_KEY")
        self.backend = (backend or os.environ.get("PROVENAS_LLM_BACKEND")
                        or ("openai" if self.api_key else "ollama")).lower()
        default_host = "https://api.openai.com" if self.backend == "openai" else "http://localhost:11434"
        self.host = (host or os.environ.get("PROVENAS_LLM_HOST") or default_host).rstrip("/")
        self.timeout = timeout
        self.available = None

    def _post(self, path, body, headers=None):
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _chat(self, messages, fmt=None, num_predict=256):
        if self.backend == "openai":                       # OpenAI-compatible /v1/chat/completions
            body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": num_predict}
            if fmt == "json":
                body["response_format"] = {"type": "json_object"}
            headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
            path = "/chat/completions" if self.host.endswith("/v1") else "/v1/chat/completions"
            return self._post(path, body, headers)["choices"][0]["message"]["content"]
        body = {"model": self.model, "stream": False, "think": False, "messages": messages,   # Ollama native
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": num_predict}}
        if fmt:
            body["format"] = fmt
        return self._post("/api/chat", body)["message"]["content"]

    def ping(self):
        """Probe reachability with a SHORT timeout first — startup must not hang for minutes on an
        unroutable host — then one tiny generation with the full timeout (a cold model may need to load)."""
        probe = "/v1/models" if self.backend == "openai" else "/api/version"
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        try:
            urllib.request.urlopen(urllib.request.Request(self.host + probe, headers=headers),
                                   timeout=5).read()
        except urllib.error.HTTPError:
            pass                                           # an HTTP error is still a reachable server
        except Exception:
            self.available = False
            return False
        try:
            self._chat([{"role": "user", "content": "reply with: ok"}], num_predict=8)
            self.available = True
        except Exception:
            self.available = False
        return self.available

    def translate(self, question, context):
        sysmsg = ("Translate the user's question into exactly ONE JSON action for a knowledge-graph "
                  "engine. Output ONLY JSON, no prose. Use \"?x\" to mark the unknown to find.")
        out = self._chat([{"role": "system", "content": sysmsg},
                          {"role": "user", "content": context + "\n\nQuestion: " + question}], fmt="json")
        return normalize_action(_loads(out))

    def propose_rule(self, spec, context):
        from provenas.infer import Rule
        sysmsg = ('You induce ONE logic rule for a knowledge-graph engine from the description and '
                  'examples. Output ONLY JSON: {"name":str,"body":[[s,r,o],...],"head":[s,r,o]}. Share '
                  '"?x"-style variables between body and head. You MAY add guard atoms: '
                  '["?a","!=","?b"] (two variables must differ), comparisons ["?age",">=","18"] '
                  '(also < <= > ==), and ["?x","~rel","?y"] meaning NO fact (?x rel ?y) exists.')
        out = self._chat([{"role": "system", "content": sysmsg},
                          {"role": "user", "content": context + "\n\n" + spec}], fmt="json")
        d = _loads(out)
        body = [_atom(a) for a in d["body"]]
        head = _atom(d["head"])
        for atom in body + [head]:
            if len(atom) != 3 or any(x is None for x in atom):
                raise ValueError(f"malformed rule atom: {atom}")
        return Rule(body, head, d.get("name", "learned"))

    def propose_tool(self, name, spec, examples):
        ex = "\n".join(f"  {name}({', '.join(map(str, a))}) == {r}" for a, r in examples)
        sysmsg = ("Write ONE pure Python function. Rules: NO imports, NO attribute access (no '.'), no "
                  "exec/eval/open/print, no dunder names. Use only plain builtins (abs, min, max, range, "
                  "len, sum, sorted, int, ...) plus basic control flow and self-recursion. Output ONLY the "
                  "function source code — no markdown fences, no prose, no examples.")
        prompt = f"Define a function named `{name}`. {spec}\nIt must satisfy:\n{ex}"
        return _strip_code(self._chat([{"role": "system", "content": sysmsg},
                                       {"role": "user", "content": prompt}], num_predict=400))

    def narrate(self, question, answer, trace):
        sysmsg = ("You relay a knowledge engine's EXACT result in 1-2 plain sentences. Do NOT invent "
                  "facts; restate only the given answer and, if a reason is given, summarize it.")
        msg = f"Question: {question}\nExact answer: {answer}\nDerivation:\n{trace}"
        return self._chat([{"role": "system", "content": sysmsg},
                           {"role": "user", "content": msg}], num_predict=160).strip()


def _loads(s):
    """Tolerant JSON: strip code fences / surrounding prose, take the outermost {...}, drop trailing commas.
    Only if it STILL fails to parse, quote bare ?variables (?a -> "?a") — last, because that regex cannot
    see JSON string boundaries and would corrupt a valid document containing '?' inside a string."""
    import re
    s = s.strip()
    if "```" in s:
        s = s.replace("```json", "```").split("```")[1] if s.count("```") >= 2 else s.replace("```", "")
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i:j + 1]
    s = re.sub(r",(\s*[}\]])", r"\1", s)          # trailing commas
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'(?<!")\?(\w+)', r'"?\1"', s))


def _strip_code(s):
    """Pull the function source out of a possibly fenced / prose-wrapped LLM reply."""
    s = s.strip()
    if "```" in s:
        for part in s.split("```"):
            if "def " in part:
                s = part
                break
        s = s.lstrip()
        if s.lower().startswith("python"):
            s = s[6:]
    return s.strip()


def _atom(a):
    """Normalize a rule atom given as [s,r,o] or {s,r,o} (unwrapping extra [ ... ] nesting) into a 3-tuple."""
    while isinstance(a, list) and len(a) == 1 and isinstance(a[0], (list, dict)):
        a = a[0]
    if isinstance(a, dict):
        return (a.get("s") or a.get("subject"),
                a.get("r") or a.get("relation") or a.get("predicate") or a.get("rel"),
                a.get("o") or a.get("object") or a.get("value"))
    return tuple(a)


def normalize_action(a):
    """Coerce a pattern/triple given as {s,r,o} (or [{...}]) into a flat [s, r, o] list."""
    def to_list(x):
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], dict):
            x = x[0]
        if isinstance(x, dict):
            return [x.get("s") or x.get("subject"), x.get("r") or x.get("relation"),
                    x.get("o") or x.get("object")]
        return x
    if "pattern" in a:
        a["pattern"] = to_list(a["pattern"])
    if "triple" in a:
        a["triple"] = to_list(a["triple"])
    return a
