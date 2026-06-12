# Provenas — a small neuro-symbolic engine

**Talk to a knowledge base in plain English and get answers that are exact, proven, and never
hallucinated** — backed by a fabric that computes symbolically, shows its work, and only ever changes
itself through a test-before-admit gate.

Provenas pairs a **neural / LLM controller** (which *proposes* — translates language, picks tools, drafts
rules and code) with an **exact symbolic fabric** (which *verifies and computes* — facts, rules, inference,
arithmetic, memory). The LLM is the mouth and hands; the fabric is the source of truth. Every answer comes
with a derivation; every change — a fact, a rule, a piece of code — is validated before it is accepted.

> It started as a research project on *where neural nets fail at exact computation* (the cliff plots in
> the table below) and grew, component by component, into the engine described here. The research write-up
> is in [`CONCLUSIONS.md`](CONCLUSIONS.md).

---

## What it's for

Use it where you need an assistant you can **talk to** but whose answers you must be able to **trust and
audit** — no made-up facts, every conclusion backed by a proof:

- **Explainable rule systems** — access control, eligibility/benefits, compliance, product/config
  compatibility, troubleshooting. *"Can Alice deploy to prod?" → yes, with the exact role→permission chain.*
- **A self-extending knowledge base** — teach it new rules from examples, and let it synthesize new tools,
  each one tested before it is allowed in.
- **Grounded NL interface** — natural-language ergonomics with symbolic guarantees: exact answers,
  derivations, and a "not derivable" instead of a confident guess when it doesn't know.

**Practical implications.** It's small and local (a tiny controller + a local LLM via Ollama), so it runs
on one box, offline, auditable end to end — a fit for regulated / safety-critical / edge settings where
"the LLM probably got it right" isn't acceptable. It can grow safely because nothing — fact, rule, or
code — enters the system without passing a test.

**Scale & performance (measured).** The engine is a stratified, semi-naive Datalog evaluator over an
indexed triple store, and the derived closure is **materialized** — computed when facts or rules change,
served from cache on reads. Measured on one box (CPU): 50k base facts → 287k-fact closure in ~1.3s;
200k base facts → 1.15M-fact closure in ~6s; queries against the materialized KB are sub-millisecond.
The envelope (10⁵–10⁶ facts) is enforced by `tests/test_perf.py`.

**Honest limits.** Natural-language answers route through an LLM (≈ seconds of latency, accuracy tracks
model strength — the gates catch bad proposals either way); the code sandbox is defense-in-depth (AST
allowlist + a resource-limited subprocess), *not* OS-level isolation — untrusted multi-tenant use needs
containers/seccomp; and it's a single-process, single-writer system (the HTTP service mode serializes
access; it is not a distributed database).

---

## Quickstart (the CLI)

Natural-language questions need a chat model — a local or remote [Ollama](https://ollama.com) server, or any
OpenAI-compatible API (see [Connecting a model](#connecting-a-model) below). The `:command` interface works
without one.

```bash
pip install -e .                    # installs the `provenas` command (pure stdlib — no heavy deps)
provenas mykb.db                    # opens (or creates) a persistent SQLite knowledge base
provenas-serve mykb.db 8642         # …or run the same KB as a local HTTP decision service
```

Starter packs to `:load`: **family, rbac, diagnostics, eligibility, config** (or start empty and `:assert`
your own facts).

```text
provenas — neuro-symbolic engine   (kb: mykb.db)
  llm: qwen3.5:9b via ollama @ http://localhost:11434  · ready   ·   :help

provenas> :load rbac
  loaded 'rbac'

provenas> Can Alice deploy to production?
  → yes
    [{'action': 'check', 'triple': ['alice', 'can', 'prod_deploy']}]
    (alice can prod_deploy)   ⇐ rule[can-do]
      (alice has_perm deploy_perm)   ⇐ rule[perm-from-role]
        (alice has_role engineer)   (given fact)
        (engineer grants deploy_perm)   (given fact)
      (prod_deploy requires deploy_perm)   (given fact)

provenas> :tool gcd : greatest common divisor of a and b :: gcd(12,8)=4, gcd(7,5)=1
  proposed:
      def gcd(a, b):
          if b == 0:
              return a
          return gcd(b, a % b)
  -> ADMITTED (saved): all 2 examples passed

provenas> :call gcd 48 36
  gcd(48, 36) = 12
```

Type `:help` for the full command list.

**New here? Two fast ways in:**

- **The zero-to-hero tutorial** — open [`docs/tutorial.html`](docs/tutorial.html) in a browser
  (clone the repo and double-click it): ten short chapters from first fact to service mode, with
  real transcripts, a command cheatsheet, and the full action/rule reference.
- **Three real-life examples** in [`examples/`](examples/) — each runs offline in seconds, no model,
  no third-party packages (CI runs them too, so they can't rot):

```bash
python examples/access_audit.py          # "who can touch prod, and WHY" — proofs, what-if, strict schema
python examples/eligibility_screening.py # benefits rules with >=/~unless; a pinned case blocks a bad rule
python examples/service_sidecar.py       # the HTTP decision sidecar, queried like your app would
```

## Connecting a model

The natural-language layer talks to any chat model over one of two protocols — configure it with environment
variables (nothing leaves your machine unless you point it at a remote endpoint):

```bash
# 1. Local Ollama (the default — nothing to set)
ollama pull qwen3.5:9b

# 2. Remote Ollama server
export PROVENAS_LLM_HOST=http://my-server:11434
export PROVENAS_LLM_MODEL=qwen3.5:9b

# 3. Any OpenAI-compatible API (OpenAI, OpenRouter, vLLM, LM Studio, llama.cpp, Together, Groq, …)
export PROVENAS_LLM_BACKEND=openai
export PROVENAS_LLM_HOST=https://api.openai.com     # or your provider's / local server's base URL
export PROVENAS_LLM_API_KEY=sk-...                  # presence alone also selects the openai backend
export PROVENAS_LLM_MODEL=gpt-4o-mini               # any chat model the endpoint serves
```

The fabric is the source of truth regardless of model — a weaker model just proposes worse actions/rules,
which the test-before-admit gates catch. The `:command` interface works with no model at all.

---

## Examples

**Ask, and see the proof.** Free-text questions are translated to an exact query/check; the answer arrives
with a derivation back to the given facts (the RBAC chain above). When the LLM mis-reads a question, the
fabric answers *"not derivable"* rather than inventing one — the guarantee that makes it trustworthy.

**Teach a rule from examples — tested before it's admitted.** You give a description and ± example pairs;
the LLM drafts a rule, the fabric admits it only if it derives every positive and no negative. If it's
wrong, the gate feeds the verdict back and the LLM revises — and if it still fails, it's rejected, never
admitted (a wrong rule can't slip in).
```text
provenas> :load family
provenas> :learn grandparent : a grandparent is a parent of a parent :: tom,ann tom,cy tom,dan | tom,bob
  proposed: grandparent ⇐ (?x parent ?y), (?y parent ?z)
  -> ADMITTED (saved)
provenas> :rules
  grandparent: ?x grandparent ?z  ⇐  (?x parent ?y), (?y parent ?z)
  ...
```
(Rule quality tracks the model: a small model may propose a flawed rule that the gate rejects; a stronger
one — `PROVENAS_LLM_MODEL=qwen3.5:27b-q4_K_M` — gets harder relations like `sibling`. Either way, only a
verified rule is ever committed.)

**Synthesize a tool — safe *and* correct, or it doesn't get in.** The LLM writes Python; the fabric admits
it only after AST validation (no imports / attribute access / dangerous builtins) **and** running it in a
resource-limited sandbox against your examples. An `import os` escape is rejected statically, an infinite
loop is killed by the sandbox, a wrong implementation fails the tests — only `gcd` above survives.

**Compute, and feed it back.** `:simplify (x + 0) * (2 + 3)` → `x * 5` — exact algebraic simplification,
its rewrite rules living in the same KB (`:rewrites` to list them). And a synthesized tool's result can
become a fact: `:call double 21 as answer is` asserts `(answer is 42)`, which later queries can use.

**Real policy rules: negation and comparisons.** Rule bodies aren't limited to positive lookups —
`["?u","~suspended","?x"]` means *no* `suspended` fact exists (stratified negation, so "allowed *unless*
suspended" works and a negation cycle is rejected at admission), and comparison guards like
`["?age",">=","18"]` compare numerically. Both are available to learned rules.

**Pin decisions; rule changes must preserve them.** After any answer, `:case prod-gate` pins it as a
regression case. From then on, every `:learn` re-runs the pinned cases against the candidate ruleset and
**rejects any rule that flips one** — CI for rules. `:disable <rule>` / `:enable <rule>` deactivate a rule
without losing it (the closure updates immediately), and `:declare` + `:strict on` make the store reject
asserts whose relation was never declared (no more typo'd facts).

**Inspect everything.** `:facts`, `:rules`, `:tools`, `:rewrites`, `:cases`, `:schema`,
`:why <s> <r> <o>`, `:log`, `:kb` — the knowledge base, rules, tools, pinned cases, schema, and an audit
log that records every change *and every decision* (query/check + answer) are all open to view.

**Run it as a service.** `provenas-serve mykb.db` exposes the same exact fabric over HTTP — the way OPA
runs as a policy sidecar, but every answer carries its proof. `POST /action` needs no model at all;
`POST /ask` adds the NL layer when one is configured. Set `PROVENAS_API_TOKEN` to require a bearer token.

```bash
curl -s localhost:8642/action -d '{"action":"check","triple":["alice","can","prod_deploy"]}'
# {"kind": "check", "answer": true, "trace": "(alice can prod_deploy)   ⇐ rule[can-do]\n  ..."}
```

Two end-to-end demo scripts reproduce the headline results:
`python -m experiments.rules_toy` (NL → proof across three domains, 9/9) and
`python -m experiments.rule_learning` (learn `grandparent`/`sibling`/`uncle`, gated).

---

## How it works

One shape, repeated: **a controller proposes, exact tools verify and compute, inspectable memory holds the
state.** Three tiers, all local:

- **Interface** — any chat model, via Ollama or an OpenAI-compatible API (`provenas/llm.py`; default
  Qwen on local Ollama): NL → a structured `{query|check|assert}` action, or a drafted rule / tool.
  Calls disable "thinking" on reasoning models and request JSON.
- **Fabric (the source of truth)** — an indexed associative knowledge graph (`kg.py`), a stratified
  semi-naive Datalog engine with negation, comparison guards, and provenance proofs (`infer.py`), a
  learned term-rewriting engine (`rewrite.py`), and the cross-domain reduction engine from the research
  phase (`engine.py`, `typed.py`).
- **Persistence** — SQLite (`store.py`): facts, rules (with activation history), a tool registry,
  pinned regression cases, an optional relation schema, the materialized closure, and an append-only
  audit + decision log.

The governance spine, applied at **every** level — facts, rules, rewrite rules, and code — is the same:

> **The LLM proposes; the fabric verifies and is the source of truth. Nothing is admitted until it passes
> a test.** A fact is asserted; a rule must satisfy ± examples; a rewrite rule must be value-preserving; a
> tool must pass static checks *and* a sandboxed run. The LLM is never trusted as the answer.

A further idea closes the loop back to the neural core: because the symbolic tools are *exact*, they are an
inexhaustible **oracle** that re-labels training data for the small controller — so when the rule set
changes, the net **retrains itself** automatically, no human labels (`experiments/rewrite_engine.py`).

---

## The research behind it (Phases 1–6)

How a "neural net can't reliably add" became "a learned controller drives an exact, self-extending machine."
Each row emits a plot or prints; the findings are written up in [`CONCLUSIONS.md`](CONCLUSIONS.md).

| Phase | Result | Artifact |
|---|---|---|
| 1 · distillation | nets *interpolate* but cliff at the training edge; multiply fails where add degrades gently | `m1_cliff.png`, `m3_stacked_cliffs.png` |
| 1 · parsing vs computing | Tree-LSTM (given structure) computes where a Transformer (given string) can't | `m4_cvd_bars.png` |
| 2 · the hybrid | neural dispatcher → symbolic VM = **exact** computation the pure net never achieves | `dispatcher_depth.png` |
| 3 · unbounded length | local reduction + a conv pointer → **100% exact at every depth**, trained on 1–4 | `scratchpad_depth.png` |
| 4 · the engine | ONE structure-only controller → arithmetic + lists + boolean + fuzzy, exact, depth-gen | `make_it_useful.png` |
| 5 · reasoning | KG + inference + TransE semantics + a solver that **combines tools** (10→100%) | `kg_reasoner.png` |
| 5 · typed engine | comparisons (num→bool) + if/else branching; controller drives typed reduction | `typed_engine.png` |
| 6 · Slice 1 | NL → exact answer **+ proof tree** over SQLite, 9/9 across 3 domains | `python -m experiments.rules_toy` |
| 6 · Slice 2 | learns **composable rules**, each gated by ± examples | `python -m experiments.rule_learning` |
| 6 · Slice 3 | **self-retraining** controller: ruleset evolves → stale dip → auto-recover from the oracle | `rewrite_engine.png` |
| 6 · Slice 4 | **synthesizes sandboxed tools**, admitted only if safe *and* correct | `python -m experiments.tool_synthesis` |

(The full Phase-1 distillation results — validator, stateful, perf, the RASP-L probes — and the cliff
findings are written up in [`CONCLUSIONS.md`](CONCLUSIONS.md).)

---

## Layout

```
provenas/
  cli.py  __main__.py     the CLI (python -m provenas [kb.db])
  server.py               HTTP service mode (provenas-serve): the fabric as a decision sidecar
  llm.py                  LLM interface, Ollama or OpenAI-compatible (translate / propose rule / tool / narrate)
  store.py                SQLite spine: facts, rules, tools, cases, schema, materialized closure, audit log
  qa.py                   the canonical answer path: action -> exact answer + proof
  domains.py              starter knowledge packs (family / rbac / diagnostics / eligibility / config)
  kg.py  infer.py         indexed knowledge graph + stratified semi-naive Datalog (negation,
                          comparison guards) with provenance proofs
  learn.py                rule admission gate (± examples + pinned regression cases before commit)
  rewrite.py              learned term-rewriting engine + value-preserving rule gate
  toolsmith.py            tool synthesis safety gate (AST allowlist + sandboxed subprocess)
  engine.py typed.py …    the cross-domain reduction engine from the research phases
docs/tutorial.html        the zero-to-hero tutorial (self-contained; open in a browser)
examples/                 real-life walkthroughs: access audit, eligibility screening, HTTP sidecar
experiments/              one script per result (the table above)
tests/sanity.py           fast checks for every component (no training, no LLM)
```

## Tests & development

```bash
pip install -e ".[dev]" && pytest        # core + stress + perf suites (pure stdlib, fast)
pip install -e ".[research]" && pytest   # also runs the Phase 1-5 research tests (torch + numpy)
```

- `tests/test_core.py` — the engine and fabric (knowledge graph, inference, store, QA, rewrite, sandbox).
- `tests/test_stress.py` — adversarial inputs: cyclic/recursive rules, malformed actions, and a battery of
  sandbox-escape attempts the tool gate must reject.
- `tests/test_perf.py` — the published performance envelope, enforced (a return to naive evaluation fails).
- `tests/test_research.py` — the distillation phases; auto-skipped if torch/numpy aren't installed.
- `python -m tests.sanity` — the same fast checks as a plain script, no pytest needed.

The research experiments live in `experiments/` (`python -m experiments.<name>`, needs the `research` extra)
and write plots to `artifacts/`. The natural-language features need a local [Ollama](https://ollama.com)
server with a model pulled (default `qwen3.5:9b`; override with `PROVENAS_LLM_MODEL`).
