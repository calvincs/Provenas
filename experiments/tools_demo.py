"""Part 2: operations are TOOLS the controller dispatches to.

So "beyond + - * /" — scientific functions and even symbolic ALGEBRA — slot into
the hybrid for free: each is one entry in the VM's tool table, and correctness comes
from the tool, not the net. The controller's job (pick which tool + which operands)
is identical no matter how complex the tool is — and M2 classified behavior at 100%,
M4/the dispatcher picked ops/programs exactly. This script shows the *tool* side: one
table spanning arithmetic -> scientific -> symbolic algebra, all computing exactly.
"""
from __future__ import annotations

import math

# Adding an operation to the hybrid VM is literally one dict entry. The net never
# computes any of these — it only learns to dispatch to them.
NUMERIC_TOOLS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "pow": lambda a, b: a ** b, "mod": lambda a, b: a % b,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "exp": math.exp, "log": math.log, "hypot": math.hypot,
}


def demo_numeric():
    T = NUMERIC_TOOLS
    print("Numeric / scientific tools (VM computes exactly via math):")
    print("  sqrt(sin(1)^2 + cos(1)^2) =",
          T["sqrt"](T["sin"](1.0) ** 2 + T["cos"](1.0) ** 2))
    print("  exp(log(42))             =", T["exp"](T["log"](42.0)))
    print("  hypot(3, 4)              =", T["hypot"](3.0, 4.0))
    print("  pow(2, 10) mod 1000      =", T["mod"](T["pow"](2, 10), 1000))


def demo_algebra():
    try:
        import sympy as sp
    except ImportError:
        print("\nAlgebra: sympy not installed (the tool would be sympy; dispatch is the same).")
        return
    x = sp.symbols("x")
    SYMBOLIC_TOOLS = {
        "diff": lambda e: sp.diff(e, x),
        "solve": lambda e: sp.solve(e, x),
        "expand": lambda e: sp.expand(e),
        "factor": lambda e: sp.factor(e),
        "integrate": lambda e: sp.integrate(e, x),
    }
    T = SYMBOLIC_TOOLS
    print("\nSymbolic algebra tools (VM computes exactly via sympy):")
    print("  diff(x^3 + 2x)      =", T["diff"](x ** 3 + 2 * x))
    print("  solve(x^2 - 4 = 0)  =", T["solve"](x ** 2 - 4))
    print("  expand((x+1)^3)     =", T["expand"]((x + 1) ** 3))
    print("  factor(x^2 - 4)     =", T["factor"](x ** 2 - 4))
    print("  integrate(3x^2 + 2) =", T["integrate"](3 * x ** 2 + 2))


if __name__ == "__main__":
    demo_numeric()
    demo_algebra()
    print("\nThe controller's role (dispatch: which tool + operands) is identical for ALL of")
    print("these. Exactness/correctness is the TOOL's job, so scientific computing and")
    print("symbolic algebra are free extensions of the same hybrid — just a richer VM.")
