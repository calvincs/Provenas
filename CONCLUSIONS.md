# provenas — Definitive Conclusions

**Question.** Is *neural function distillation* — pointing a pipeline at a Python
callable and learning a small network that mimics its observable behavior —
feasible? We are testing what is *possible* (near-approximation, not lossless
matching), and how the distilled artifact compares to the original Python.

**Verdict in one line:** distillation works *beautifully for classifying which
behavior applies* and *fails for reproducing exact unbounded computation*. The
honest use of the technique is fuzzy routing/validation over behavior — not
replacing a calculator. Every number below is measured (CPU/GPU on aibox, 4× RTX
4090), averaged over seeds where noted.

---

## 1. The feasibility spectrum (the headline)

| Capability | Example | Result | Feasible? |
|---|---|---|---|
| **Classify a string into a category** | `InputValidator.detect_type` | **99.9%** accuracy, **robust** to 6.6× length shift (99.99%) | ✅ shines |
| **Detect *which* error/branch fires** | calculator error head (M2) | **100%** accuracy; learns a smooth P(error) surface | ✅ shines |
| **Compute with structure handed in** | Tree-LSTM on parse trees (M4) | depth-1 ~12% rel-err, usable *near-approximation*; degrades with depth | ⚠️ partial |
| **Compute by parsing a raw string** | Transformer on expr strings (M4) | depth-1 ~55%, collapses by depth 2 | ❌ |
| **Exact unbounded arithmetic** | calculator regression (M1/M3) | best ~3×10⁻³ in a narrow box; cliffs to 100% off-distribution | ❌ |
| **Hold exact numeric state across calls** | BankAccount balance | both neural trackers DRIFT (M0 →~1750, M1 →~380 abs err over 100 ops); a Python variable is exact | ❌ use a store |
| **Compute *exactly* via a hybrid (net drives a symbolic VM)** | dispatcher → stack VM | **100%** exact to depth 5 when trained on the range; pure net is ~0% exact, ever | ✅ the resolution |

The single most important picture: **the network is good at learning *which
behavior applies* and *where errors live*, and bad at *being* the computation.**

---

## 2. Where and why it breaks (the cliffs are the deliverable)

- **Extrapolation cliff (M1, `m1_cliff.png`).** Trained on operands in [-50, 50],
  every op interpolates to ~3×10⁻³ relative error. Past the training range,
  *linear* ops (add/subtract) degrade only to ~0.2 (a ReLU net extrapolates
  affinely), while the *nonlinear* op (multiply) fails completely (→ 1.0).
  Lesson: networks interpolate; whether they extrapolate at all depends on the
  linearity of the target.

- **Dynamic range vs extrapolation (M3, `m3_stacked_cliffs.png`).** Log-magnitude
  encoding beats raw encoding by **~10⁵×** in-distribution for multiply (median
  rel-err ~0.07 vs ~10⁴), because `log(a·b)=log(a)+log(b)` linearizes magnitude.
  But **neither encoding extrapolates** — both cliff to ~100% past the training
  range, because a small error in log-space is *multiplicatively* amplified by the
  `exp` decode. The float32 precision floor (~6×10⁻⁸) sits far below what either
  model reaches; the "precision cliff" is a floor, not a knee.

- **Soft boundary vs hard rule (M2, `m2_error_surface.png`).** The net replaces
  Python's infinitely-thin discontinuity at `b=0` with a *smooth band* of
  P(ZeroDivisionError)≈1 (half-width ~0.7). The band's sharpness is set by the
  *encoding*: a sign-aware encoding yields a near-exact zero-detector; a smooth one
  yields a fuzzy band. You get classification of behavior, not the exact rule.

- **Parsing vs computing (M4, `m4_depth_scaling.png`).** Handing the model the
  parse *tree* (D) vs the raw *string* (C) is the whole experiment. D computes
  (12% error at depth 1, a real near-approximation) where C cannot (55% at depth
  1, ~100% by depth 2). **The D–C gap is the cost of parsing** — enormous at low
  depth. But D itself falls off a *computing* cliff with nesting depth, and by
  depths 5–7 (held out) both collapse to ~100%. Error *detection*, by contrast,
  stays high for the tree model (≈99%) — structure helps the structural task far
  more than the arithmetic one.

The recurring mechanism: every cliff is the network being asked to *be* exact,
unbounded computation. Representation (log/digit/tree) delays the cliff; it never
removes it.

---

## 3. Performance: distilled net vs the original Python (`perf_throughput.png`)

| Mode | throughput | vs Python |
|---|---:|---:|
| Python `run_op` (loop) | 2.83M calls/s | 1× |
| Neural, CPU, batch=1 | 27k calls/s | **100× slower** |
| Neural, CPU, batch=4096 | 2.64M calls/s | ~1× |
| Neural, GPU, batch=65536 | **123M calls/s** | **43× faster** |

The static Python function wins decisively for single/scalar calls — the net's
per-call dispatch overhead makes it 100× slower. The net only wins on **massive
batched GPU throughput** (43×), and even then it is *approximate*, not exact. So
you would never swap a calculator for a net to gain speed *or* accuracy. The net's
advantage is throughput on a firehose of inputs where an approximate, *uniform*
answer is acceptable — i.e. classification/routing at scale, not arithmetic.

---

## 4. The resolution: a neural dispatcher over a symbolic machine

Everything above says the same thing — let the net do *control*, not computation —
so we built exactly that. A small Transformer learns to *compile* an expression
(over positional value-symbols `N0,N1,…`) into a stack program (RPN); a real Python
stack executes it with the exact values held in an external list. The net
manipulates **symbols**, the list is **typed memory**, the stack machine is **loops +
logic execution**, and the op-codes are **tooling** — the net never does arithmetic.

Result (`dispatcher_depth.png`) — the hybrid is **EXACT**, not approximate:

| nesting depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| trained on depths 1–4 | 1.00 | 1.00 | 1.00 | 0.98 | 0.34 | 0.09 | 0.02 |
| trained on depths 1–7 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.93 | 0.71 |
| M4 pure Tree-LSTM | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 |

Two things matter. (1) **In-distribution the hybrid is exact** (100% to depth 5) — a
categorical jump over the pure net, which is *never* exact. The arithmetic cliff is
gone; exactness comes from the symbolic VM, and the net only has to get the *program
structure* right. (2) **The remaining cliff is the net's, not the machine's:** trained
only to depth 4 it doesn't generalize deeper (0.34 at depth 5), but trained on the
range it handles them (0.99) — so the limit is the Transformer's sequence/length
generalization (a known, separately-fixable problem: scratchpads, relative positions,
RPN curricula), *not* an inability to compute. The depth-7 dip (0.71) is the longest
programs straining a tiny model.

This is the plan's predicted "honest endgame," realized: the moment it works you have
*a learned dispatcher driving an interpreter* — a neural/classical hybrid, not a pure
net. That is the price of exact, inspectable, unbounded computation, and it is the
right price to pay.

## 5. The honest landing

1. **Distill the control flow, keep the computation.** Networks reliably learn
   *which* branch / type / error-state applies (M2 100%, validator 99.9%). They do
   not learn to *be* the arithmetic. The productive pattern is a learned dispatcher
   over behavior with exact computation kept symbolic — which is also where M4's
   "hand in the structure" result points.
2. **Approximation is the deal, and it has a shape.** Within a trained
   distribution, near-approximation is genuinely achievable (M1 ~0.3%, M4 depth-1
   ~12%). The value of the project is *mapping the cliff* — extrapolation,
   dynamic-range, depth, charset — so you know the operating envelope.
3. **Feed structure; it pays.** Every time we hand the network the function's
   structure (types, operation identity, parse tree, error labels) it does better.
   M4 is the sharpest case: the tree model computes where the string model can't.
   Taken to its limit this becomes "a learned controller over a data structure" —
   a neural/classical hybrid, not a pure net (the plan's §4a thesis).

---

## 6. Scope tested vs. open

Covered, with evidence, across all three primitives the plan claims Python logic
decomposes into:
- **deterministic numeric** (regression — the pessimistic bound),
- **discrete classification** (the optimistic sweet spot),
- **stateful transition** (`state_drift.png`): a BankAccount running balance. The
  plan predicted the pure recurrent net (M1) would drift; we found that **both**
  neural approaches drift — and that state-as-input (M0) drifts *worse* (~1750 abs
  error by 100 ops) than the recurrent net (~380), because re-feeding a feedforward
  prediction compounds errors, especially when the balance scale dwarfs the per-step
  update. Only **M2** — holding the balance in an actual Python variable — is exact
  (flat at zero). Same thesis again: a net can't *be* exact numeric state;
  externalize it to a classical store.

Plus an error/branch-detection head, a reusable thin pipeline, and a neural-vs-Python
performance comparison.

Open (stretch goals): differentiable get/set/delete memory (NTM/DNC-style), the
symbolic hybrid dispatcher (path F), and digit-sequence arithmetic (path E) — each a
deeper version of "feed in more structure," which every result here says is the right
direction.

**Bottom line:** a *pure* network is feasible and useful for the
classification/routing/error-detection slice of Python logic, and not a viable
replacement for exact computation. But a **neural dispatcher driving a symbolic
memory/VM** *is* exact — the net learns the control flow, the classical machine
supplies the exactness — and that hybrid, not the pure net, is what makes distilling
real program behavior feasible. The cliff plots say exactly where each piece's limits
are, and the dispatcher result says how to get past them.

---

## 7. Phase 2 — the richer machine (in progress)

The dispatcher opens the full program-induction vision (a net driving a typed,
dynamic memory with loops and tools). We're probing four directions, each a focused
feasibility experiment building on the symbolic-machine pattern; findings appended as
they land.

1. **Symbol grounding (3 = 1+1+1)** — `grounding_count.py`. Learn number *semantics*
   by composition: addition-by-counting (a learned loop over an exact memory counter).
   If the net learns the loop rather than memorizing step counts, addition is exact at
   ANY magnitude — the inverse of M1's cliff, because it counts instead of approximating.
   **Result: 100% exact for b up to 2000, trained only on b<=20** (`grounding_count.png`).
   The learned loop generalizes perfectly; grounding addition in counting + exact memory
   removes the magnitude cliff entirely.
2. **Variables + get/set/delete memory** — a register/variable machine: `x = 3; y = x + x`
   compiled to LOAD/STORE/op over a keyed store (the plan's M2 with explicit lifecycle).
   **Result: 100% exact for 2-6 statements (trained range), 0% at 7+** (`variables_depth.png`).
   The net learns symbol binding and the keyed store makes execution exact in-distribution;
   program-length generalization is the same controller bottleneck (→ direction 4).
3. **Loops + control flow** — bounded loops/conditionals; the controller drives the VM
   step-by-step until a halt condition (the deepest "logic execution").
   **Result: multiply-by-repeated-addition is 100% exact for n up to 500 (trained n<=20)**
   (`grounding_mult.png`) — the same learned repeat-until loop as counting, now doing real
   accumulation; multiplication grounded in repeated addition, exact at any magnitude.
4. **Generalize to any depth — SOLVED.** First, the cliff is purely the training
   distribution (register machine trained on 2-12 is exact across 2-12, cliffs only at 13).
   Then the real fix: reformulate evaluation as **iterative local reduction** (a scratchpad)
   — a model points to the next op to reduce, the symbolic VM reduces it exactly, repeat.
   With **(a)** a genuinely LOCAL reduction rule (immediate operator-neighbors, not a global
   "deepest-paren" scan) and **(b)** a **translation-equivariant convolutional pointer** (no
   positional encoding, so it applies the same fixed-radius function everywhere), per-step
   accuracy is **100% at every depth** and end-to-end exact-match is **100% through depth 9,
   99.4% at depth 10 — trained only on depths 1-4** (`scratchpad_depth.png`). The length cliff
   is GONE. The lesson: **match the model's inductive bias (locality, translation-equivariance)
   to the algorithm's structure (a local rewrite rule)** and the hybrid generalizes unbounded.
   (Ablation: one-shot dispatcher cliffs at depth 5 ~34%; scratchpad + global rule + Transformer
   decays to ~7% by depth 10; scratchpad + local rule + Transformer holds per-step ~85%;
   scratchpad + local rule + **conv** holds per-step **100%** — the inductive bias is the lever.)

**Phase 2 synthesis.** Two regimes emerge. (a) When the controller's per-step decision is a
*simple, magnitude-independent rule* — counting/multiply's "step while work remains" — it
**extrapolates perfectly**, so grounding arithmetic in loops over exact memory gives EXACT
results at *any* magnitude (no cliff). (b) When the controller must emit *complex structure*
— parse a deep expression, compile a long program — it is exact *within* its trained length
range but does not extrapolate beyond it; curriculum slides the cliff to wherever you train,
and reformulating as a local-reduction **scratchpad with a translation-equivariant model
removes it entirely — exact at unbounded length**. Either way the **symbolic
memory/VM supplies the exactness**; the only variable is how far the learned controller
generalizes. The hybrid is feasible, exact, and inspectable — the pure net is none of these
for computation. That is the definitive conclusion: **don't make the net compute; make it
drive a machine that does.**

**Beyond + - * / — scientific computing & algebra (`tools_demo.py`).** Because operations are
*tools* the controller dispatches to, the operation set is arbitrary. One tool table spans
arithmetic, scientific functions (`sqrt, sin, cos, exp, log, hypot` via `math`), and symbolic
**algebra** (`diff, solve, factor, integrate` via `sympy`) — all computing exactly
(`solve(x^2-4)=[-2,2]`, `diff(x^3+2x)=3x^2+2`, `integrate(3x^2+2)=x^3+2x`). The net's job —
pick which tool + operands — is identical regardless of the tool's complexity (M2 classified
behavior at 100%; M4/the dispatcher picked ops exactly). So scientific computing and computer
algebra are **free extensions of the same hybrid — just a richer VM**, and the scratchpad
already generalizes the control to unbounded length. The natural next step is a trained
dispatcher over this richer grammar.

---

## 8. Novelty probes (Phase 3) — and an honest verdict

A literature check first: length generalization is a large, active area. The **RASP-L
conjecture** ([Zhou et al., ICLR 2024](https://arxiv.org/pdf/2310.16028)) characterizes which
tasks monolithic transformers length-generalize on (counting/sorting yes; **addition, parity
no**); and scratchpad-for-length-gen is **mixed / often negative** in practice
([Anil et al. 2022](https://arxiv.org/pdf/2207.04901)). So our earlier results largely
*reproduce/cleanly-frame* known theory. We then targeted two genuinely-open threads where the
controlled hybrid is an edge:

- **Probe A — hybrid vs the RASP-L barrier (`probe_a_parity.png`).** PARITY is RASP-L-hard: a
  monolithic transformer collapses to chance (~0.50) past the training length; decomposed into a
  **controller + exact 1-bit memory** (a learned finite-state transducer), the SAME task is
  **100% at every length (5→80, trained ≤20)**. *Honest novelty: modest — this is the classic
  "recurrence/FST beats transformer on parity," cleanly framed; the fact is known.*

- **Probe B (pilot) — discover the policy from OUTCOME reward (`probe_b_discover.png`).** With NO
  per-step supervision (only sparse end-of-episode reward + the VM as exact environment),
  REINFORCE **discovered** the counting policy and it length-generalizes **100% to b=500**.
  *Honest novelty: the easy case is a trivial RL task; the genuine frontier is discovering a
  non-trivial symbolic algorithm (e.g. the reduction policy) from outcomes — which we then
  attempted directly.*

- **Probe B (hard) — discover the REDUCTION policy from outcome reward (`probe_b_hard.png`).** A
  conv policy in the reduction VM, trained by REINFORCE on depths 1-4 with ONLY terminal reward
  (+1 iff the final value is exact), no step labels. It **discovered a length-generalizing policy**:
  100% at depths 1-4, and ~99 / 95 / 89% at depths 5 / 7 / 10 — 2.5× beyond training, with no
  per-step supervision. (Our forecast was a likely negative; it was a clear positive — the
  many-correct-orders structure + conv locality made credit assignment tractable.) It trails the
  supervised scratchpad (100% at depth 10) by ~11 points — a measurable *trace-supervision gap*.

**Honest verdict (post-probes).** Probe B-hard *worked* — better than our forecast (we expected a
likely negative). But a literature check places it in mainstream territory: outcome-reward RL for
reasoning ([RLVR](https://www.emergentmind.com/topics/reinforcement-learning-for-llm-reasoning)) is
the dominant LLM-reasoning paradigm; length generalization via inductive bias is known; and
neuro-symbolic RL ([Neural Reward Machines](https://arxiv.org/pdf/2408.08677), etc.) is active. So
our result is a clean, surprisingly-strong *controlled instance* of established ideas — a nice
demonstration plus a measurable trace-supervision gap — but **not a novel contribution**. The
intellectually honest bottom line: this path is an outstanding way to *understand* the frontier
(and it even beat our own pessimistic forecast), but it does not yield a genuinely novel research
result.

**Phase 4 (deeper machine / domains / interpretability).** Item 1 (loops) iteration-generalizes 200×;
Item 2 (lists) confirms the controller is domain-agnostic (structure-only) — both clean instances,
not novel. **Item 4 (interpretability) is the crispest finding of the whole project:** by
reverse-engineering the local conv policies we found that the RL-discovered reduction policy is
precedence-correct but learns a *different, rightmost-leaning tie-break* than the supervised leftmost
one (`item4_interpret.png`). The reward pins down precedence (what correctness requires) and leaves
tie-break order free; RL fills it non-canonically. This cleanly decomposes a learned symbolic policy
into reward-CONSTRAINED vs reward-FREE parts — a tidy controlled illustration of **reward
underspecification** (Agarwal et al. 2019; reward-hacking lit), which is a known phenomenon. Crisp and
interesting, still not a novel contribution. **The verdict is now thoroughly evidenced across many
angles: this work is powerful, general, and honest — and not novel.**

**Final state (Phase 4 complete).** All five roadmap items are done. The capstone is
`provenas/engine.py`: one **structure-only** controller drives EXACT evaluation across arithmetic,
lists, and boolean domains, depth-generalizing (`make_it_useful.png`) — a genuinely useful, reusable
realization of the whole thesis (**control is shared & learned; computation is per-domain exact
tools**). Item 5 was a clean honest *negative* (the reduction task is too well-structured to need a
curriculum). Novelty across the whole Phase-4 sweep: none (the cross-domain library idea is DreamCoder,
Ellis et al. 2021). Value: a complete, honest, end-to-end working understanding of neuro-symbolic
computation — from "a net can't add" to "a learned controller drives a symbolic machine to compute
exactly, at unbounded depth, over an arbitrary, swappable set of domains."

## Phase 5 — reasoning & knowledge (in progress)

The engine grows from exact COMPUTATION toward knowledge + REASONING by compounding new Legos onto the
same controller-drives-tools fabric.

- **Fuzzy logic** (`make_it_useful.png`) — a new logic costs only a tool table (OR=max, AND=min over
  [0,1]); the same structure-only controller evaluates it exactly, depth-generalizing. Four domains now
  ride one controller.
- **Associative knowledge graph** (`provenas/kg.py`) — exact relational memory: assert/retract,
  mini-Datalog pattern query, transitive inheritance inference, groups, property discovery. The storage core.
- **The reasoner** (`provenas/{kgvm,infer,semantic,solver}.py`, `kg_reasoner.png`) — the whole fabric on a
  small animal world:
  - `infer` forward-chains rules to derive inherited facts (whale ⊢ has_part backbone; 87 derived), finds
    connecting relation paths, and induces 2-hop rules from examples.
  - `semantic` (TransE) gives the neural half its grip: it clusters the taxonomy, returns "what is like X,"
    and predicts *adjacent, unstored* relationships — e.g. proposing `(lion, can, walk)`, never asserted,
    from the learned geometry.
  - `solver` is the **aha**: for a multi-constraint goal it discovers by trial-and-error the smallest
    COMBINATION of {infer, combine, semantic} that cracks it. Solve-rate climbs **10% → 48% → 89% → 100%**
    as more tools may combine — putting two or three ideas together is what unlocks the hard problems
    (flagship: `backbone ∧ lives_in water ∧ like shark → {dolphin, salmon, shark, whale}`, all three tools).
  - a tiny **controller** learns goal-features → the right combination, turning the search into one-shot
    dispatch (**1.00 vs 5.25** attempts, 48/48 test goals) — a neural controller driving symbolic reasoning,
    the arithmetic-engine pattern one level up.
  - **learning:** solved derivations are baked back into the KG and the winning combination is named as a
    reusable macro — the system grows new structure from what it solved.

- **Typed engine — comparisons + if/else + mixed types** (`provenas/typed.py`, `typed_engine.png`). Values
  carry a type (num|bool) and operators a signature, so one expression crosses types: arithmetic,
  comparison (num,num→**bool**), boolean, and `if c then a else b` that BRANCHES with short-circuit (the
  dead branch is never evaluated — `if 5-5==0 then 0 else 10/(5-5)` = 0, no error). The lazy local reducer
  is exact (== Python oracle, 4000 exprs); and the **same structure-only conv controller** (token types
  only, never values) drives the whole reduction, generalizing past its training depth (trained ≤4;
  end-to-end exact 100% through depth 5, ~92% at depth 7). Branching and types, added to the fabric
  without changing its shape.

The fabric is the point: one shape — a (learned) controller dispatching over swappable, exact symbolic
tools + inspectable memory — now spans arithmetic, lists, boolean, fuzzy, relational knowledge +
reasoning, **and** a typed engine with comparisons and if/else branching. A calculator yesterday; a small
reasoner today.

## Phase 6 — the integrated system (in progress)

Slice 1 stands up the 3-tier system (Qwen on Ollama · exact fabric · SQLite) end to end: a natural-language
question → Qwen emits a structured action → the fabric forward-chains rules and returns the exact answer
WITH a proof tree → Qwen narrates it. One pipeline, **9/9 exact across three different domains** (kinship,
access-control, diagnostics); facts and rules persist in SQLite; an NL "learn a fact" step flips a decision
and survives a reopen. The governance spine — the LLM proposes, the fabric verifies and is the source of
truth — held even when Qwen mis-translated (the fabric answered "not derivable", never hallucinated). The
fabric is now reachable in plain English, with every answer backed by an exact, inspectable derivation.

Slice 2 makes it self-evolving: Qwen proposes new RULES (not just facts), and a test-before-admit gate
validates each against positive/negative examples on a KB copy before it is committed to SQLite. Taught
`grandparent` / `sibling` / `uncle` from descriptions, the gate rejected both an over-generating naive rule
and Qwen's own direction-inverted proposal (a wrong rule never reached the store), then admitted the vetted
ones — and `uncle` composed on the just-learned `sibling`. The knowledge base grows by proposal-and-proof,
not by trust.

Slice 3 closes the loop back to the neural core: a learned term-rewriting engine where the structure-only
controller is REGENERATED from the exact oracle whenever the ruleset changes. A value-preserving soundness
gate admits new rewrite rules (and rejects an unsound one); when constant folding is added, the stale
controller degrades (99.6% → 81.9%, worst at depth) and is automatically retrained from oracle-labeled data
back to 99.1%, generalizing past its training depth. Because the symbolic tools are exact, they are an
inexhaustible teacher for the small net — the net updates itself as the fabric evolves, no human labels.
This is the whole thesis in one loop: a learned controller and exact symbolic tools, each making the other
better — the controller routes, the tools compute and *teach*.

Slice 4 lets the system grow new TOOLS: an LLM authors a Python function and a layered safety gate admits it
only if it is both safe (AST allowlist — no imports, attribute access, or dangerous builtins) and correct
(run in a resource-limited subprocess against examples). The gate rejected an import-escape and an
attribute-escape (static), an infinite loop (sandbox), and a wrong implementation (tests); Qwen then
synthesized gcd, is_prime, and collatz_steps — each admitted only after passing, persisted to a SQLite tool
registry, and callable.

With slices 1-4, Phase 6 is complete: a small, honest neuro-symbolic system where natural language drives an
exact, inspectable fabric — answering with proofs, learning vetted composable rules, regenerating its own
neural controller from the exact oracle, and synthesizing sandboxed tools — with TEST-BEFORE-ADMIT at every
level (facts, rules, rewrite rules, and code) and the LLM always proposing, never the source of truth. From
"a net can't add" to a learned controller driving an exact, self-extending machine that a person can talk to
and inspect end to end.
