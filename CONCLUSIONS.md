# Provenas — Conclusions

Provenas began by asking *where small neural networks fail at exact computation*, and ended as a working
**neuro-symbolic engine**: a learned / LLM controller that **proposes**, and an exact symbolic fabric that
**verifies, computes, and remembers**. This document summarizes what the work concluded. Every number is
measured (CPU/GPU on a 4× RTX 4090 box), averaged over seeds where noted.

**The thesis, in one line:** *don't make the network compute — make it drive a machine that does.* A pure
net is good at deciding **which** behavior applies and bad at **being** the computation; pairing a learned
controller with exact symbolic tools gives you both — natural-language ergonomics with exactness, proofs,
and no hallucinated facts.

---

## 1. Where pure neural networks fail (the motivating result)

We mapped the failure modes precisely — the "cliffs" — because knowing the operating envelope is the point:

- **Extrapolation.** Trained on operands in [-50, 50], every op interpolates to ~3×10⁻³ relative error;
  past the training range linear ops (add/subtract) degrade gently while multiply fails completely.
  *Networks interpolate; whether they extrapolate depends on the linearity of the target.*
- **Dynamic range ≠ extrapolation.** Log-magnitude encoding beats raw by ~10⁵× in-distribution for
  multiply, yet **neither extrapolates** — a small log-space error is multiplicatively amplified by the
  `exp` decode.
- **Parsing vs computing.** Handed a parse *tree*, a model computes (≈12% error at depth 1); handed the raw
  *string* it cannot (≈55%, ~100% by depth 2). The gap is the cost of parsing — and both still fall off a
  *computing* cliff with nesting depth.
- **Exact state drifts.** A net tracking a running balance drifts (worse when it re-feeds its own
  predictions); only an external variable stays exact.

The mechanism is identical every time: a cliff is a network being asked to *be* exact, unbounded
computation. **Representation (log / digit / tree) delays the cliff; it never removes it.** What nets *do*
excel at is classifying which branch / type / error applies — 100% on the error head, 99.9% on string-type
classification. That is **control, not computation.**

---

## 2. The resolution: a controller driving exact tools

Let the net do control, not computation. A small model learns to **drive a symbolic machine** that holds
exact values in external memory and executes the operations. The result is a categorical jump: the hybrid
is **exact** where the pure net is *never* exact (100% to depth 5 when trained on the range, vs ~0% for the
pure net). The only remaining limit is how far the *controller* generalizes — a separable problem, which
we solved:

- **Unbounded length.** Reformulating evaluation as **iterative local reduction** (a scratchpad) driven by
  a **translation-equivariant convolutional pointer** (no positional encoding) gives ~100% exact-match at
  every depth — 100% through depth 9, 99.4% at depth 10 — **trained only on depths 1–4.** The length cliff
  is gone. The lever is matching the model's inductive bias (locality, translation-equivariance) to the
  algorithm's structure (a local rewrite rule).
- **Two regimes.** When the per-step decision is a *simple, magnitude-independent rule* ("step while work
  remains"), it extrapolates perfectly — grounding arithmetic in loops over exact memory makes addition and
  multiplication exact at *any* magnitude. When the controller must emit *complex structure*, it is exact
  within its trained range, and the scratchpad reformulation extends that to unbounded length.
- **Discoverable from outcomes.** The reduction policy can be *discovered* by reinforcement learning from
  sparse terminal reward alone (no per-step labels) and still length-generalizes — the symbolic VM doubles
  as an exact training environment.

Either way the **symbolic machine supplies the exactness**; the net only has to get the structure right.

---

## 3. One controller, many domains — the engine

The controller is **structure-only**: it reasons about token *types* and positions, never values, which
makes it domain-agnostic. One learned controller drives exact, depth-generalizing evaluation across
**arithmetic, lists, boolean logic, fuzzy logic, and a typed engine with comparisons (num→bool) and
`if/else` branching** (short-circuit — a dead branch never runs). Because operations are **tools the
controller dispatches to**, the operation set is arbitrary: scientific functions and computer algebra
(sympy `solve`, `diff`, `integrate`) are a tool-table swap, not new machinery. *Control is shared and
learned; computation is per-domain exact tools.*

---

## 4. From computation to reasoning — the knowledge fabric

The same controller-drives-tools pattern extends from computing to **knowledge and reasoning**:

- an **associative knowledge graph** — exact relational memory (assert/retract, pattern query, transitive
  inference, property discovery);
- a **Datalog-style inference engine** that derives facts and returns a **proof tree** for every conclusion;
- **neural semantics** (TransE embeddings) that cluster symbols and predict *adjacent, unstored*
  relationships from learned geometry;
- a **combination solver** — the "aha" — that discovers the smallest mix of {infer, combine, semantic} tools
  to crack a multi-constraint goal (solve-rate climbs 10% → 100% as more tools may combine), with a tiny
  controller that learns to pick the right combination up front.

A calculator yesterday; a small reasoner today — the same fabric.

---

## 5. The integrated system — talk to it, and trust it

The product is a three-tier system: an **LLM interface** (local or remote Ollama, or any OpenAI-compatible
API) that *proposes*; the **exact fabric** that *verifies and is the source of truth*; and **SQLite** that
*persists* facts, rules, tools, and an audit log. You ask in plain English, the LLM emits a structured
action, and the fabric returns the exact answer **with a derivation** — nothing is taken on the model's word.

The governance spine is one principle applied at **every** level — *the LLM proposes; the fabric verifies;
nothing is admitted until it passes a test*:

- **facts / queries** — answered exactly, with a proof; a mis-read question yields "not derivable", never a
  hallucination;
- **rules** — an LLM-proposed rule is admitted only if it derives every positive and no negative example (a
  wrong rule is rejected, revised, and never committed);
- **rewrite rules** — admitted only if value-preserving; and when the ruleset changes, the controller
  **retrains itself from the exact oracle**, with no human labels;
- **tools** — an LLM-authored Python function is admitted only if it passes static AST checks (no imports,
  attribute access, or dangerous builtins) *and* runs correctly in a resource-limited sandbox.

So the system **grows safely**: it learns vetted, composable rules, regenerates its own neural controller as
it evolves, and synthesizes sandboxed tools — all by proposal-and-proof, not by trust.

**Hardening it was engineering, not research — and the same spine absorbed it.** The evaluator became an
indexed, semi-naive, **stratified Datalog engine**: rule bodies gained negation (`~rel` — "unless", with
negation-through-a-cycle rejected at admission) and numeric comparison guards (`?age >= 18`), which is what
real policy actually needs. The derived closure is **materialized** — recomputed when facts or rules change,
served from cache on reads. And the governance idea extended naturally to operations: **pinned regression
cases** (any answer can be pinned; a rule change that would flip a pinned decision is rejected — CI for
rules), rule **activation history** (switch a rule off, see exactly which conclusions evaporate — instant
what-if analysis), an optional **strict relation schema** (a typo'd fact is rejected, not stored), a
**decision log** (every answer, not just every change, is on the record), and an **HTTP service mode** that
runs the fabric as a local policy sidecar — the proof riding along in every response, no LLM required.

---

## 6. Performance, novelty, and honest limits

**Performance.** Against the original Python, a distilled net is ~100× slower per scalar call and only wins
on massive batched GPU throughput (≈43×) — and even then it is approximate. You never swap exact code for a
net to gain speed or accuracy; the net's role is *control at scale*, and the fabric keeps the exactness.

The symbolic engine, meanwhile, scales like the database technique it is. The first implementation was
naive (re-derive everything, full scans): 2,000 facts took 28 s to close. Indexing the triple store,
joining each round only against the newly derived delta (semi-naive evaluation, delta atom first), and
materializing the closure brought that to **0.04 s — a 700× drop** — with 200,000 base facts closing to a
1.15 M-fact KB in ~6 s and cached reads in microseconds. The envelope (10⁵–10⁶ facts on one box) is
enforced by a performance regression suite, so it is a published number, not a hope.

**Novelty — honestly.** Each individual component sits in well-studied territory (length-generalization /
RASP-L, DreamCoder-style library learning, Datalog inference, knowledge-graph embeddings, neuro-symbolic RL,
LLM-as-proposer-with-verification). The contribution is not a new algorithm; it is the **compound** — these
pieces integrated into one small, coherent, end-to-end system with **test-before-admit at every level** and
exactness as an invariant. The integrated whole being useful is the point.

**Limits.** Natural-language answers route through an LLM (seconds of latency, and a weaker model proposes
worse actions — the gates catch them, but accuracy tracks model strength; the gates make a weak model *safe*,
not smart). The code sandbox is defense-in-depth (AST allowlist + a resource-limited subprocess), **not**
OS-level isolation, so untrusted multi-tenant use needs containers / seccomp. And it is a single-process,
single-writer system: the service mode serializes access on one box — it is a sidecar with a measured
10⁵–10⁶-fact envelope, not a distributed database, and it only knows what it has been told (it complements
retrieval; it does not replace it).

---

## Bottom line

A *pure* network is feasible and useful for the classification / routing slice of program behavior, and not
a viable replacement for exact computation — the cliff plots say exactly where its limits are. But a
**learned controller driving an exact symbolic fabric** *is* exact, inspectable, and self-extending: it
answers in plain English, shows its work, learns vetted rules, regenerates its own controller, and grows new
sandboxed tools — with the LLM always proposing and never the source of truth.

**Don't make the net compute; make it drive a machine that does — and let nothing into the machine that
hasn't passed a test.**
