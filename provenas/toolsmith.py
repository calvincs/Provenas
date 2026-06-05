"""toolsmith — Slice 4: synthesize new Python TOOLS, admit only after sandboxed testing.

The most powerful and riskiest capability: an LLM authors a brand-new tool (a Python function), and the
fabric admits it ONLY after a layered gate:
  1. STATIC AST validation against an allowlist — no imports, no attribute access ('.'), no dunder names,
     no dangerous builtins; only safe builtins + locals + self-recursion.
  2. SANDBOXED execution — a separate Python process with a wall-clock timeout AND rlimits (CPU, memory),
     a minimal environment, run against I/O examples (test-before-admit, extended to code).
Only code that is both safe AND correct on every example is admitted (and persisted to SQLite).

This is defense in depth for a single-box toy. A real deployment needs OS-level isolation (containers /
seccomp / gVisor) — the AST allowlist and a resource-limited subprocess are necessary, not sufficient.
"""
from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys

SAFE_BUILTINS = {
    "abs", "min", "max", "sum", "len", "range", "int", "float", "bool", "str", "list", "tuple",
    "dict", "set", "frozenset", "sorted", "enumerate", "zip", "map", "filter", "reversed", "round",
    "pow", "divmod", "all", "any", "ord", "chr",
}


# ----------------------------------------------------------------- 1. static validation
def _targets(t, names):
    if isinstance(t, ast.Name):
        names.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _targets(e, names)
    elif isinstance(t, ast.Starred):
        _targets(t.value, names)


def validate_ast(src, func_name):
    """Return (ok, reason). Accept only a single safe pure function named func_name."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or not funcs or funcs[0].name != func_name:
        return False, f"must define exactly one function named '{func_name}'"
    fn = funcs[0]

    allowed = set(SAFE_BUILTINS) | {func_name}
    a = fn.args
    for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
        allowed.add(arg.arg)
    if a.vararg:
        allowed.add(a.vararg.arg)
    if a.kwarg:
        allowed.add(a.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _targets(t, allowed)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For)):
            _targets(node.target, allowed)
        elif isinstance(node, ast.comprehension):
            _targets(node.target, allowed)

    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed"
        if isinstance(node, ast.Attribute):
            return False, "attribute access ('.') is not allowed"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global/nonlocal not allowed"
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"dunder name '{node.id}' not allowed"
            if isinstance(node.ctx, ast.Load) and node.id not in allowed:
                return False, f"forbidden/unknown name '{node.id}'"
    return True, "ok"


# ----------------------------------------------------------------- 2. sandboxed run
def _limits():                                            # POSIX, applied in the child before exec
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))


def run_sandboxed(src, func_name, inputs, timeout=3):
    """Run func(*args) for each args in `inputs` in a locked-down subprocess. Return (ok, outputs|reason)."""
    harness = (src + "\n"
               "import json, sys\n"
               "_in = json.loads(sys.stdin.read())\n"
               "print(json.dumps([" + func_name + "(*a) for a in _in]))\n")
    try:
        p = subprocess.run([sys.executable, "-I", "-c", harness], input=json.dumps(inputs),
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return False, "timeout (killed by the sandbox)"
    except Exception as e:
        return False, f"sandbox error: {e}"
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["non-zero exit"])[-1]
        return False, tail
    try:
        return True, json.loads(p.stdout.strip())
    except Exception:
        return False, "tool produced non-JSON output"


# ----------------------------------------------------------------- load an admitted tool
def load_tool(src, name):
    """Exec a VALIDATED tool with builtins restricted to the safe set, return the callable."""
    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    ns = {"__builtins__": safe}
    exec(compile(src, f"<tool:{name}>", "exec"), ns)
    return ns[name]


# ----------------------------------------------------------------- admission gate
def admit_tool(store, name, src, examples, source="qwen", timeout=3):
    """Layered gate: AST -> sandbox -> examples. Persist to SQLite only if all pass.
    examples: list of (args_tuple, expected). Returns (ok, stage, detail)."""
    ok, reason = validate_ast(src, name)
    if not ok:
        return False, "ast", reason
    ok, out = run_sandboxed(src, name, [list(a) for a, _ in examples], timeout=timeout)
    if not ok:
        return False, "sandbox", out
    fails = [(a, exp, got) for (a, exp), got in zip(examples, out) if got != exp]
    if fails:
        a, exp, got = fails[0]
        return False, "tests", f"{len(fails)}/{len(examples)} failed, e.g. {name}{tuple(a)} -> {got} != {exp}"
    if store is not None:
        store.add_tool(name, src, examples, source=source)
        store.log("admit_tool", f"{name} ({len(examples)} examples)")
    return True, "admitted", f"all {len(examples)} examples passed"
