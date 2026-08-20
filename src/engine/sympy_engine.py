"""Deterministic symbolic mathematics engine for Saad.AI.

This module intentionally contains no Streamlit or network dependencies so it can be
unit-tested independently of the UI and provider integrations.
"""

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

def run_sympy(problem: str) -> dict:
    """
    Symbolic computation engine. Returns verified result dict.
    Falls back silently on any error.
    ELIF ORDER (important — must check ODE before Solve):
      Derivative → Integral → Limit → ODE → NR → Solve → Matrix → Mod
    """
    p = problem.lower().strip()
    x = sp.Symbol('x')
    tfms = standard_transformations + (implicit_multiplication_application,)
    ld = {
        "x": x,
        "e": sp.E, "E": sp.E,
        "pi": sp.pi, "PI": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "inf": sp.oo, "oo": sp.oo
    }

    def clean(s):
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"\^", "**", s)
        return s

    def to_coeff(s):
        """Convert string to sympy integer/rational — never float (floats break dsolve)."""
        s = s.replace(" ", "")
        if s in ("", "+"): return sp.Integer(1)
        if s == "-": return sp.Integer(-1)
        try:
            f = float(s)
            return sp.Integer(int(f)) if f == int(f) else sp.Rational(s)
        except Exception:
            return sp.Integer(1)

    try:
        # ── 1. Derivative ────────────────────────────────────────────
        if any(k in p for k in ["derivative", "differentiate", "d/dx", "diff"]):
            raw = p
            for kw in ["derivative of", "differentiate", "diff of", "d/dx of", "d/dx"]:
                if kw in p:
                    raw = p.split(kw, 1)[-1].strip()
                    break
            raw = re.sub(r"\s*(dx|with\s*respect\s*to\s*x).*$", "", raw).strip()
            expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            result = sp.diff(expr, x)
            return {"type": "Derivative", "result": str(result), "latex": sp.latex(result)}

        # ── 2. Integral ──────────────────────────────────────────────
        elif any(k in p for k in ["integral", "integrate", "antiderivative"]):
            raw = p
            for kw in ["integral of", "integrate", "antiderivative of"]:
                if kw in p:
                    raw = p.split(kw, 1)[-1].strip()
                    break
            raw = re.sub(r"\s*dx.*$", "", raw).strip()
            expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            result = sp.integrate(expr, x)
            return {"type": "Integral", "result": str(result), "latex": sp.latex(result)}

        # ── 3. Limit ─────────────────────────────────────────────────
        elif "limit" in p:
            match = re.search(
                r"limit\s+of\s+([\w\s\(\)\+\-\*/\^\.\,]+?)"
                r"\s+as\s+x\s*(?:->|→|approaches)\s*([\w\.\+\-]+)", p
            )
            if match:
                raw_expr = clean(match.group(1))
                pt = match.group(2).strip()
                point = sp.oo if pt in ("inf", "infinity", "oo") else sp.sympify(pt)
                expr = parse_expr(raw_expr, transformations=tfms, local_dict=ld)
                result = sp.limit(expr, x, point)
                return {"type": "Limit", "result": str(result), "latex": sp.latex(result)}

        # ── 4. ODE — MUST be before Solve (many ODE problems start with "solve") ──
        elif any(k in p for k in ["d²y", "d^2y", "d2y", "dy/dx",
                                   "differential equation", "second order", "first order"]):
            y_fn = sp.Function('y')
            ode_sol = None
            try:
                # Extract RHS — everything after = and before "with"
                rhs_m = re.search(r"=\s*(.+?)(?:\s+with|\s*$)", p)
                rhs_str = clean(rhs_m.group(1).strip()) if rhs_m else "0"
                rhs_expr = parse_expr(rhs_str, transformations=tfms, local_dict=ld)
                lhs = p.split("=")[0]

                # Second order: d²y/dx² + a*dy/dx + b*y = rhs
                if any(k in p for k in ["d²y", "d^2y", "d2y", "second order"]):
                    # Match coefficient of dy/dx (must be dy/dx not just dy to avoid d²y match)
                    c1_m = re.search(r"([+\-]\s*\d*\.?\d*)\s*dy/dx", lhs)
                    # Match coefficient of standalone y (word boundary)
                    c0_m = re.search(r"([+\-]\s*\d+\.?\d*)\s*y\b", lhs)
                    a1 = to_coeff(c1_m.group(1)) if c1_m else sp.Integer(0)
                    a0 = to_coeff(c0_m.group(1)) if c0_m else sp.Integer(0)
                    ode_eq = sp.Eq(
                        y_fn(x).diff(x, 2) + a1*y_fn(x).diff(x) + a0*y_fn(x),
                        rhs_expr
                    )
                    ode_sol = sp.dsolve(ode_eq, y_fn(x))

                # First order: dy/dx + a*y = rhs
                elif "dy/dx" in p:
                    c0_m = re.search(r"([+\-]\s*\d+\.?\d*)\s*y\b", lhs)
                    a0 = to_coeff(c0_m.group(1)) if c0_m else sp.Integer(0)
                    ode_eq = sp.Eq(y_fn(x).diff(x) + a0*y_fn(x), rhs_expr)
                    ode_sol = sp.dsolve(ode_eq, y_fn(x))

            except Exception:
                ode_sol = None  # safe fallback to AI

            if ode_sol is not None:
                return {
                    "type": "ODE",
                    "result": f"General solution: y = {str(ode_sol.rhs)}",
                    "latex": f"y = {sp.latex(ode_sol.rhs)}"
                }

        # ── 5. Newton-Raphson ────────────────────────────────────────
        elif any(k in p for k in ["newton", "newton-raphson", "newton raphson"]):
            # Extract: equation (between of/to/for and "= 0"), x0, iterations
            eq_m   = re.search(r"(?:of|to|for)\s+(.+?)\s*=\s*0", p)
            x0_m   = re.search(r"x\s*0\s*[=:]\s*([\d\.]+)", p)
            iter_m = re.search(r"(\d+)\s*iter", p)

            if eq_m and x0_m:
                raw_eq  = clean(eq_m.group(1).strip())
                x0_val  = float(x0_m.group(1))
                n_iter  = int(iter_m.group(1)) if iter_m else 3

                expr_nr = parse_expr(raw_eq, transformations=tfms, local_dict=ld)
                f_sym   = sp.lambdify(x, expr_nr, modules="math")
                df_sym  = sp.lambdify(x, sp.diff(expr_nr, x), modules="math")

                # Run all iterations with full precision — never round intermediate values
                iterations = []
                xn = x0_val
                for i in range(n_iter):
                    fxn  = f_sym(xn)
                    dfxn = df_sym(xn)
                    if abs(dfxn) < 1e-15:
                        break  # avoid division by zero
                    xn1 = xn - fxn / dfxn
                    iterations.append({
                        "n": i, "xn": round(xn, 8),
                        "fxn": round(fxn, 8), "dfxn": round(dfxn, 8),
                        "xn1": round(xn1, 8)
                    })
                    xn = xn1  # use FULL precision for next iteration

                final_x  = iterations[-1]["xn1"] if iterations else x0_val
                final_fx = round(f_sym(final_x), 10)
                iter_str = "\n".join([
                    f"  x{it['n']+1} = {it['xn']} - ({it['fxn']}) / ({it['dfxn']}) = {it['xn1']}"
                    for it in iterations
                ])
                result_str = (
                    f"f(x) = {str(expr_nr)}, f'(x) = {str(sp.diff(expr_nr, x))}\n"
                    f"x0 = {x0_val}, iterations = {n_iter}\n"
                    f"VERIFIED ITERATIONS (AI MUST use these exact values):\n"
                    f"{iter_str}\n"
                    f"Final answer: x{n_iter} = {final_x}\n"
                    f"Verification: f({final_x}) = {final_fx} ≈ 0"
                )
                return {
                    "type": "Newton-Raphson",
                    "result": result_str,
                    "latex": f"x_{{{n_iter}}} = {final_x}"
                }

        # ── 6. Numerical Analysis Methods ───────────────────────────
        elif any(k in p for k in ["bisection", "secant method", "false position",
                                   "regula falsi", "gauss elimination", "gauss elim",
                                   "lu decomposition", "lu decomp",
                                   "lagrange interpolation", "lagrange interp",
                                   "newton divided", "divided difference",
                                   "trapezoidal", "trapezoid rule",
                                   "simpson", "euler method", "euler's method",
                                   "runge-kutta", "runge kutta", "rk4"]):
            import math as _math

            # ── Helper: extract f(x) expression ──────────────────────
            def get_expr():
                # Stop before common natural-language delimiters so prompts such as
                # "bisection of x^3 - x on [1,2]" do not parse the interval as math.
                eq_m = re.search(
                    r"(?:of|to|for|function)\s+(.+?)(?=\s*(?:=\s*0|,|\bon\b|\bfrom\b|\bbetween\b|\[|$))",
                    p,
                )
                if eq_m:
                    raw = clean(eq_m.group(1).strip())
                    return parse_expr(raw, transformations=tfms, local_dict=ld)
                return None

            # ── Helper: extract bounds a, b ───────────────────────────
            def get_bounds():
                nums = re.findall(r"[-]?\d+\.?\d*", p)
                floats = [float(n) for n in nums]
                # x0 value
                x0_m = re.search(r"x\s*0\s*[=:]\s*([-]?\d+\.?\d*)", p)
                x1_m = re.search(r"x\s*1\s*[=:]\s*([-]?\d+\.?\d*)", p)
                # interval [a,b]
                ab_m = re.search(r"\[\s*([-]?\d+\.?\d*)\s*,\s*([-]?\d+\.?\d*)\s*\]", p)
                return floats, x0_m, x1_m, ab_m

            # ── Helper: extract iterations ────────────────────────────
            def get_iters(default=5):
                m = re.search(r"(\d+)\s*iter", p)
                return int(m.group(1)) if m else default

            # ── Helper: extract step size h ───────────────────────────
            def get_h():
                m = re.search(r"h\s*[=:]\s*([\d\.]+)", p)
                return float(m.group(1)) if m else 0.1

            # ── Helper: extract ODE rhs f(x,y) ───────────────────────
            def get_ode_rhs():
                # dy/dx = f(x,y) → extract rhs
                m = re.search(r"dy/dx\s*=\s*(.+?)(?:\s*,|\s*with|\s*y\s*\(|$)", p)
                return m.group(1).strip() if m else None

            floats, x0_m, x1_m, ab_m = get_bounds()
            n_iter = max(1, get_iters())

            # ════════════════════════════════════════════════════════
            # BISECTION METHOD
            # ════════════════════════════════════════════════════════
            if "bisection" in p:
                expr_b = get_expr()
                if expr_b is not None and ab_m:
                    a_val = float(ab_m.group(1))
                    b_val = float(ab_m.group(2))
                    f_b = sp.lambdify(x, expr_b, modules="math")
                    fa, fb = f_b(a_val), f_b(b_val)
                    if fa == 0:
                        return {"type": "Bisection", "result": f"Root = {a_val}", "latex": f"x = {a_val}"}
                    if fb == 0:
                        return {"type": "Bisection", "result": f"Root = {b_val}", "latex": f"x = {b_val}"}
                    if fa * fb > 0:
                        return {
                            "type": "Bisection",
                            "result": f"Cannot apply bisection: f({a_val}) and f({b_val}) have the same sign.",
                            "latex": "\\text{Invalid bracket}",
                        }
                    steps = []
                    a_n, b_n = a_val, b_val
                    for i in range(n_iter):
                        c = (a_n + b_n) / 2
                        fc = f_b(c)
                        steps.append({"iter": i+1, "a": round(a_n,8),
                                      "b": round(b_n,8), "c": round(c,8),
                                      "fc": round(fc,8)})
                        if f_b(a_n) * fc < 0: b_n = c
                        else: a_n = c
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: a={s['a']}, b={s['b']}, c={s['c']}, f(c)={s['fc']}"
                        for s in steps])
                    final_c = steps[-1]["c"]
                    return {
                        "type": "Bisection",
                        "result": (f"f(x) = {str(expr_b)}\n"
                                   f"Interval [{a_val},{b_val}], {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {final_c}"),
                        "latex": f"x \\approx {final_c}"
                    }

            # ════════════════════════════════════════════════════════
            # SECANT METHOD
            # ════════════════════════════════════════════════════════
            elif "secant" in p:
                expr_s = get_expr()
                if expr_s is not None and x0_m and x1_m:
                    x0_v = float(x0_m.group(1))
                    x1_v = float(x1_m.group(1))
                    f_s = sp.lambdify(x, expr_s, modules="math")
                    steps = []
                    xp, xc = x0_v, x1_v
                    for i in range(n_iter):
                        fxp, fxc = f_s(xp), f_s(xc)
                        if abs(fxc - fxp) < 1e-15: break
                        xn_val = xc - fxc*(xc-xp)/(fxc-fxp)
                        steps.append({"iter": i+1, "x": round(xn_val, 8),
                                      "fx": round(f_s(xn_val), 8)})
                        xp, xc = xc, xn_val
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: x={s['x']}, f(x)={s['fx']}"
                        for s in steps])
                    return {
                        "type": "Secant",
                        "result": (f"f(x) = {str(expr_s)}\n"
                                   f"x0={x0_v}, x1={x1_v}, {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {steps[-1]['x']}"),
                        "latex": f"x \\approx {steps[-1]['x']}"
                    }

            # ════════════════════════════════════════════════════════
            # FALSE POSITION (Regula Falsi)
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["false position", "regula falsi"]):
                expr_fp = get_expr()
                if expr_fp is not None and ab_m:
                    a_val = float(ab_m.group(1))
                    b_val = float(ab_m.group(2))
                    f_fp = sp.lambdify(x, expr_fp, modules="math")
                    steps = []
                    a_n, b_n = a_val, b_val
                    for i in range(n_iter):
                        fa, fb = f_fp(a_n), f_fp(b_n)
                        c = (a_n*fb - b_n*fa) / (fb - fa)
                        fc = f_fp(c)
                        steps.append({"iter": i+1, "a": round(a_n,8),
                                      "b": round(b_n,8), "c": round(c,8),
                                      "fc": round(fc,8)})
                        if fa * fc < 0: b_n = c
                        else: a_n = c
                    step_str = "\n".join([
                        f"  Iter {s['iter']}: a={s['a']}, b={s['b']}, c={s['c']}, f(c)={s['fc']}"
                        for s in steps])
                    return {
                        "type": "FalsePosition",
                        "result": (f"f(x) = {str(expr_fp)}\n"
                                   f"Interval [{a_val},{b_val}], {n_iter} iterations\n"
                                   f"VERIFIED ITERATIONS:\n{step_str}\n"
                                   f"Root ≈ {steps[-1]['c']}"),
                        "latex": f"x \\approx {steps[-1]['c']}"
                    }

            # ════════════════════════════════════════════════════════
            # GAUSS ELIMINATION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["gauss elimination", "gauss elim"]):
                # Extract matrix from problem — look for [[...]] pattern
                mat_m = re.search(r"\[\s*\[(.+?)\]\s*\]", p)
                rhs_m = re.search(r"(?:rhs|=|b)\s*[=:]?\s*\[([^\]]+)\]", p)
                if mat_m and rhs_m:
                    rows = re.findall(r"\[([^\]]+)\]", p)
                    mat_data = [[sp.Rational(v) for v in re.split(r"[,\s]+", r.strip()) if v]
                                for r in rows[:-1]]
                    rhs_data = [sp.Rational(v) for v in re.split(r"[,\s]+", rows[-1].strip()) if v]
                    A = sp.Matrix(mat_data)
                    b_vec = sp.Matrix(rhs_data)
                    sol = A.solve(b_vec)
                    sol_str = ", ".join([f"x{i+1}={sol[i]}" for i in range(len(sol))])
                    return {
                        "type": "GaussElimination",
                        "result": f"VERIFIED SOLUTION: {sol_str}",
                        "latex": sol_str
                    }

            # ════════════════════════════════════════════════════════
            # LU DECOMPOSITION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["lu decomposition", "lu decomp"]):
                rows = re.findall(r"\[([^\]]+)\]", p)
                if rows:
                    mat_data = [[sp.Rational(v) for v in re.split(r"[,\s]+", r.strip()) if v]
                                for r in rows]
                    M = sp.Matrix(mat_data)
                    L, U, _ = M.LUdecomposition()
                    return {
                        "type": "LUDecomposition",
                        "result": (f"VERIFIED:\nL = {str(L)}\nU = {str(U)}"),
                        "latex": f"L={sp.latex(L)}, U={sp.latex(U)}"
                    }

            # ════════════════════════════════════════════════════════
            # LAGRANGE INTERPOLATION
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["lagrange interpolation", "lagrange interp"]):
                # Extract data points (x0,y0),(x1,y1),...
                pts = re.findall(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", p)
                if pts:
                    data = [(sp.Rational(px), sp.Rational(py)) for px, py in pts]
                    poly = sp.interpolate(data, x)
                    poly_exp = sp.expand(poly)
                    return {
                        "type": "LagrangeInterpolation",
                        "result": f"VERIFIED polynomial: {str(poly_exp)}",
                        "latex": sp.latex(poly_exp)
                    }

            # ════════════════════════════════════════════════════════
            # NEWTON DIVIDED DIFFERENCE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["newton divided", "divided difference"]):
                pts = re.findall(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", p)
                if pts:
                    xs_v = [float(px) for px, py in pts]
                    ys_v = [float(py) for px, py in pts]
                    n_p = len(xs_v)
                    dd = [[0.0]*n_p for _ in range(n_p)]
                    for i in range(n_p): dd[i][0] = ys_v[i]
                    for j in range(1, n_p):
                        for i in range(n_p - j):
                            dd[i][j] = (dd[i+1][j-1]-dd[i][j-1])/(xs_v[i+j]-xs_v[i])
                    coeffs = [round(dd[0][j], 8) for j in range(n_p)]
                    # Build polynomial
                    poly_s = sp.interpolate(list(zip(xs_v, ys_v)), x)
                    return {
                        "type": "NewtonDividedDiff",
                        "result": (f"VERIFIED divided differences: {coeffs}\n"
                                   f"Polynomial: {str(sp.expand(poly_s))}"),
                        "latex": sp.latex(sp.expand(poly_s))
                    }

            # ════════════════════════════════════════════════════════
            # TRAPEZOIDAL RULE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["trapezoidal", "trapezoid rule"]):
                expr_t = get_expr()
                ab_m2 = re.search(r"\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", p)
                n_m = re.search(r"n\s*[=:]\s*(\d+)", p)
                if expr_t is not None and ab_m2 and n_m:
                    a_v = float(ab_m2.group(1))
                    b_v = float(ab_m2.group(2))
                    n_v = int(n_m.group(1))
                    f_t = sp.lambdify(x, expr_t, modules="math")
                    h = (b_v - a_v) / n_v
                    s = f_t(a_v) + f_t(b_v)
                    pts_str = [f"f({round(a_v,4)})={round(f_t(a_v),6)}", ]
                    for i in range(1, n_v):
                        xi = a_v + i*h
                        pts_str.append(f"f({round(xi,4)})={round(f_t(xi),6)}")
                    pts_str.append(f"f({round(b_v,4)})={round(f_t(b_v),6)}")
                    result_val = round(h/2 * (f_t(a_v)+f_t(b_v) + 2*sum(f_t(a_v+i*h) for i in range(1,n_v))), 8)
                    return {
                        "type": "Trapezoidal",
                        "result": (f"f(x)={str(expr_t)}, [{a_v},{b_v}], n={n_v}, h={round(h,6)}\n"
                                   f"Function values: {', '.join(pts_str)}\n"
                                   f"VERIFIED result: {result_val}"),
                        "latex": f"\\int_{{{a_v}}}^{{{b_v}}} \\approx {result_val}"
                    }

            # ════════════════════════════════════════════════════════
            # SIMPSON'S 1/3 RULE
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["simpson"]):
                expr_si = get_expr()
                ab_m3 = re.search(r"\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", p)
                n_m2 = re.search(r"n\s*[=:]\s*(\d+)", p)
                if expr_si is not None and ab_m3 and n_m2:
                    a_v = float(ab_m3.group(1))
                    b_v = float(ab_m3.group(2))
                    n_v = int(n_m2.group(1))
                    if n_v % 2 != 0: n_v += 1  # must be even
                    f_si = sp.lambdify(x, expr_si, modules="math")
                    h = (b_v - a_v) / n_v
                    s = f_si(a_v) + f_si(b_v)
                    for i in range(1, n_v):
                        s += (4 if i % 2 != 0 else 2) * f_si(a_v + i*h)
                    result_val = round(h/3 * s, 8)
                    return {
                        "type": "Simpsons",
                        "result": (f"f(x)={str(expr_si)}, [{a_v},{b_v}], n={n_v}, h={round(h,6)}\n"
                                   f"VERIFIED result: {result_val}"),
                        "latex": f"\\int_{{{a_v}}}^{{{b_v}}} \\approx {result_val}"
                    }

            # ════════════════════════════════════════════════════════
            # EULER'S METHOD
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["euler method", "euler's method"]):
                rhs_str = get_ode_rhs()
                x0_em = re.search(r"x\s*0?\s*[=:]\s*([-\d\.]+)", p)
                y0_em = re.search(r"y\s*[=\(]\s*0?\s*\)?\s*[=:]\s*([-\d\.]+)", p)
                h_val = get_h()
                n_steps = get_iters(default=5)
                if rhs_str and x0_em and y0_em:
                    x_s, y_s = sp.symbols('x y')
                    rhs_expr_ode = parse_expr(clean(rhs_str), transformations=tfms,
                                              local_dict={**ld, "y": y_s})
                    f_ode = sp.lambdify((x_s, y_s), rhs_expr_ode, modules="math")
                    xn_e, yn_e = float(x0_em.group(1)), float(y0_em.group(1))
                    steps = []
                    for i in range(n_steps):
                        yn1 = yn_e + h_val * f_ode(xn_e, yn_e)
                        xn_e += h_val
                        steps.append({"n": i+1, "x": round(xn_e,6), "y": round(yn1,8)})
                        yn_e = yn1
                    step_str = "\n".join([f"  Step {s['n']}: x={s['x']}, y={s['y']}" for s in steps])
                    final_y_e = steps[-1]['y']
                    return {
                        "type": "EulersMethod",
                        "result": (f"dy/dx = {rhs_str}, h={h_val}, {n_steps} steps\n"
                                   f"VERIFIED ITERATIONS (USE THESE EXACT VALUES):\n{step_str}\n"
                                   f"FINAL ANSWER: y({steps[-1]['x']}) = {final_y_e} — USE THIS EXACTLY"),
                        "latex": f"y_{{{n_steps}}} = {final_y_e}"
                    }

            # ════════════════════════════════════════════════════════
            # RUNGE-KUTTA RK4
            # ════════════════════════════════════════════════════════
            elif any(k in p for k in ["runge-kutta", "runge kutta", "rk4"]):
                rhs_str = get_ode_rhs()
                x0_rk = re.search(r"x\s*0?\s*[=:]\s*([-\d\.]+)", p)
                y0_rk = re.search(r"y\s*[=\(]\s*0?\s*\)?\s*[=:]\s*([-\d\.]+)", p)
                h_val = get_h()
                n_steps = get_iters(default=3)
                if rhs_str and x0_rk and y0_rk:
                    x_s, y_s = sp.symbols('x y')
                    rhs_expr_rk = parse_expr(clean(rhs_str), transformations=tfms,
                                             local_dict={**ld, "y": y_s})
                    f_rk = sp.lambdify((x_s, y_s), rhs_expr_rk, modules="math")
                    xn_r, yn_r = float(x0_rk.group(1)), float(y0_rk.group(1))
                    steps = []
                    for i in range(n_steps):
                        k1 = h_val * f_rk(xn_r, yn_r)
                        k2 = h_val * f_rk(xn_r+h_val/2, yn_r+k1/2)
                        k3 = h_val * f_rk(xn_r+h_val/2, yn_r+k2/2)
                        k4 = h_val * f_rk(xn_r+h_val, yn_r+k3)
                        yn1 = yn_r + (k1+2*k2+2*k3+k4)/6
                        xn_r += h_val
                        steps.append({"n": i+1, "x": round(xn_r,6),
                                      "k1": round(k1,8), "k2": round(k2,8),
                                      "k3": round(k3,8), "k4": round(k4,8),
                                      "y": round(yn1,8)})
                        yn_r = yn1
                    step_str = "\n".join([
                        f"  Step {s['n']}: x={s['x']}, k1={s['k1']}, k2={s['k2']}, k3={s['k3']}, k4={s['k4']}, y={s['y']}"
                        for s in steps])
                    final_y = steps[-1]['y']
                    return {
                        "type": "RungeKutta4",
                        "result": (f"dy/dx={rhs_str}, h={h_val}, {n_steps} steps\n"
                                   f"VERIFIED ITERATIONS (USE THESE EXACT k VALUES):\n{step_str}\n"
                                   f"FINAL ANSWER: y({round(xn_r,4)}) = {final_y} — USE THIS EXACTLY"),
                        "latex": f"y_{{{n_steps}}} = {final_y}"
                    }

        # ── 7. Theory of Numbers — SymPy computes exactly ───────────────
        elif any(k in p for k in [
                "gcd", "greatest common divisor", "hcf",
                "lcm", "least common multiple",
                "prime factor", "factoriz", "factori",
                "is prime", "isprime", "prime or not", "check prime",
                "totient", "euler's totient",
                "number of divisor", "sum of divisor", "divisors of",
                "mobius", "möbius",
                "diophantine",
                "chinese remainder", "crt",
                "fermat", "wilson",
                "quadratic residu", "legendre",
                "primitive root",
                "linear congruence", "congruence", "≡",
                "euler's theorem", "euler theorem",
        ]) or re.search(r'\bis\s+\d+\s+prime\b', p):
            import math as _math

            def _nums(text):
                return [int(n) for n in re.findall(r'\b\d+\b', text)]

            x_d, y_d = sp.symbols('x y')

            try:
                # ── GCD (Extended Euclidean) ──────────────────────────────
                if any(k in p for k in ["gcd","greatest common divisor","hcf"]):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a,b = nums[0],nums[1]
                        g = int(sp.gcd(a,b))
                        # Extended Euclidean
                        old_r,r = a,b; old_s,s = 1,0; old_t,t2 = 0,1
                        while r:
                            q=old_r//r; old_r,r=r,old_r-q*r
                            old_s,s=s,old_s-q*s; old_t,t2=t2,old_t-q*t2
                        result_str = (f"gcd({a},{b}) = {old_r}\n"
                                      f"Extended Euclidean: {a}×({old_s}) + {b}×({old_t}) = {old_r}\n"
                                      f"Verify: {a*old_s + b*old_t} = {old_r} ✅")
                        return {"type":"GCD","result":result_str,
                                "latex":f"\\gcd({a},{b})={old_r}"}

                # ── LCM ───────────────────────────────────────────────────
                elif any(k in p for k in ["lcm","least common multiple"]):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        l = int(sp.lcm(nums[0],nums[1]))
                        g = int(sp.gcd(nums[0],nums[1]))
                        return {"type":"LCM",
                                "result":f"lcm({nums[0]},{nums[1]}) = {l}, gcd = {g}",
                                "latex":f"\\text{{lcm}}({nums[0]},{nums[1]})={l}"}

                # ── PRIME FACTORIZATION ───────────────────────────────────
                elif any(k in p for k in ["prime factor","factoriz","factori"]):
                    nums = _nums(p)
                    if nums:
                        f_dict = sp.factorint(nums[0])
                        f_str = " × ".join([f"{pp}^{e}" if e>1 else str(pp) for pp,e in f_dict.items()])
                        return {"type":"PrimeFactorization",
                                "result":f"{nums[0]} = {f_str}",
                                "latex":f"{nums[0]} = {f_str}"}

                # ── IS PRIME ──────────────────────────────────────────────
                elif any(k in p for k in ["is prime","isprime","prime or not","check prime"]) or re.search(r'\bis\s+\d+\s+prime\b',p):
                    nums = _nums(p)
                    if nums:
                        n_val = nums[0]
                        is_p = sp.isprime(n_val)
                        ans = "PRIME" if is_p else "COMPOSITE (NOT PRIME)"
                        return {"type":"PrimeCheck",
                                "result":f"{n_val} is {ans}",
                                "latex":f"{n_val}\\text{{ is }}{ans}"}

                # ── EULER TOTIENT ─────────────────────────────────────────
                elif any(k in p for k in ["totient","euler's totient"]):
                    nums = _nums(p)
                    if nums:
                        phi = int(sp.totient(nums[0]))
                        f_dict = sp.factorint(nums[0])
                        return {"type":"EulerTotient",
                                "result":f"φ({nums[0]}) = {phi}, factorization = {f_dict}",
                                "latex":f"\\phi({nums[0]})={phi}"}

                # ── DIVISOR FUNCTIONS τ, σ ────────────────────────────────
                elif any(k in p for k in ["number of divisor","sum of divisor","divisors of","tau","sigma"]):
                    nums = _nums(p)
                    if nums:
                        divs = sp.divisors(nums[0])
                        return {"type":"DivisorFunctions",
                                "result":f"divisors({nums[0]}) = {divs}, τ = {len(divs)}, σ = {sum(divs)}",
                                "latex":f"\\tau({nums[0]})={len(divs)}, \\sigma({nums[0]})={sum(divs)}"}

                # ── MOBIUS FUNCTION ───────────────────────────────────────
                elif any(k in p for k in ["mobius","möbius"]):
                    nums = _nums(p)
                    if nums:
                        mu = sp.mobius(nums[0])
                        return {"type":"Mobius",
                                "result":f"μ({nums[0]}) = {mu}",
                                "latex":f"\\mu({nums[0]})={mu}"}

                # ── DIOPHANTINE EQUATION ──────────────────────────────────
                elif "diophantine" in p:
                    m = re.search(r"(\d+)\s*x\s*[+\-]\s*(\d+)\s*y\s*=\s*(\d+)",p)
                    if m:
                        a_d,b_d,c_d = int(m.group(1)),int(m.group(2)),int(m.group(3))
                        g = int(sp.gcd(a_d,b_d))
                        if c_d%g == 0:
                            sol = sp.diophantine(sp.Eq(a_d*x_d+b_d*y_d, c_d))
                            return {"type":"Diophantine",
                                    "result":f"{a_d}x+{b_d}y={c_d}: general solution={sol}, gcd={g}",
                                    "latex":str(sol)}
                        else:
                            return {"type":"Diophantine",
                                    "result":f"No integer solution: gcd({a_d},{b_d})={g} does not divide {c_d}",
                                    "latex":"\\text{No solution}"}

                # ── CRT — MUST be before congruence ───────────────────────
                elif any(k in p for k in ["chinese remainder","crt"]):
                    pairs = re.findall(r'x\s*[≡=]\s*(\d+)\s*(?:mod|modulo|\(mod)\s*(\d+)',p)
                    if len(pairs) >= 2:
                        remainders = [int(r) for r,m in pairs]
                        moduli = [int(m) for r,m in pairs]
                        sol = sp.ntheory.modular.crt(moduli, remainders)
                        verify = [f"{sol[0]}%{m}={sol[0]%m}" for m in moduli]
                        return {"type":"CRT",
                                "result":f"x ≡ {sol[0]} (mod {sol[1]}), verify: {verify}",
                                "latex":f"x\\equiv {sol[0]}\\pmod{{{sol[1]}}}"}

                # ── FERMAT'S LITTLE THEOREM ───────────────────────────────
                elif "fermat" in p:
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a_f,p_f = nums[0],nums[1]
                        if sp.isprime(p_f):
                            r = pow(a_f,p_f-1,p_f)
                            return {"type":"FermatTheorem",
                                    "result":f"{a_f}^({p_f}-1) mod {p_f} = {r} ≡ 1 (mod {p_f})",
                                    "latex":f"{a_f}^{{{p_f-1}}}\\equiv 1\\pmod{{{p_f}}}"}

                # ── EULER'S THEOREM ───────────────────────────────────────
                elif "euler" in p and ("theorem" in p or "theorem" in p):
                    nums = _nums(p)
                    if len(nums) >= 2:
                        a_e,n_e = nums[0],nums[1]
                        phi = int(sp.totient(n_e))
                        r = pow(a_e,phi,n_e)
                        return {"type":"EulerTheorem",
                                "result":f"φ({n_e})={phi}, {a_e}^{phi} mod {n_e} = {r} ≡ 1 (mod {n_e})",
                                "latex":f"{a_e}^{{\\phi({n_e})}}\\equiv 1\\pmod{{{n_e}}}"}

                # ── WILSON'S THEOREM ──────────────────────────────────────
                elif "wilson" in p:
                    nums = _nums(p)
                    if nums:
                        p_w = nums[0]
                        val = _math.factorial(p_w-1)%p_w
                        return {"type":"WilsonTheorem",
                                "result":f"({p_w}-1)! mod {p_w} = {val} ≡ -1 (mod {p_w})",
                                "latex":f"({p_w}-1)!\\equiv -1\\pmod{{{p_w}}}"}

                # ── LINEAR CONGRUENCE ─────────────────────────────────────
                elif any(k in p for k in ["congruence","linear congruence"]) or re.search(r'\d+\s*x\s*[≡=]',p):
                    m = re.search(r"(\d+)\s*x\s*[≡=]\s*(\d+)\s*(?:\(mod|mod|modulo)\s*(\d+)",p)
                    if m:
                        a_c,b_c,n_c = int(m.group(1)),int(m.group(2)),int(m.group(3))
                        g = int(sp.gcd(a_c,n_c))
                        if b_c%g != 0:
                            return {"type":"LinearCongruence",
                                    "result":f"No solution: gcd({a_c},{n_c})={g} ∤ {b_c}",
                                    "latex":"\\text{No solution}"}
                        sols = [i for i in range(n_c) if (a_c*i)%n_c==b_c%n_c]
                        return {"type":"LinearCongruence",
                                "result":f"{a_c}x ≡ {b_c} (mod {n_c}): x ≡ {sols} (mod {n_c}), {g} solution(s)",
                                "latex":f"x\\equiv {sols[0]}\\pmod{{{n_c//g}}}"}

                # ── QUADRATIC RESIDUES ────────────────────────────────────
                elif any(k in p for k in ["quadratic residu","quadratic non"]):
                    nums = _nums(p)
                    if nums:
                        p_q = nums[0]
                        qr = sorted(set([pow(i,2,p_q) for i in range(1,p_q)]))
                        qnr = [i for i in range(1,p_q) if i not in qr]
                        return {"type":"QuadraticResidues",
                                "result":f"QR mod {p_q} = {qr}, QNR mod {p_q} = {qnr}",
                                "latex":f"QR\\pmod{{{p_q}}}={qr}"}

                # ── LEGENDRE SYMBOL ───────────────────────────────────────
                elif "legendre" in p:
                    m = re.search(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)",p)
                    if m:
                        a_l,p_l = int(m.group(1)),int(m.group(2))
                        val = 1 if pow(a_l,(p_l-1)//2,p_l)==1 else (-1 if a_l%p_l!=0 else 0)
                        meaning = "QR (quadratic residue)" if val==1 else ("QNR (non-residue)" if val==-1 else "0 (divisible)")
                        return {"type":"LegendreSymbol",
                                "result":f"({a_l}/{p_l}) = {val} → {a_l} is {meaning} mod {p_l}",
                                "latex":f"\\left(\\frac{{{a_l}}}{{{p_l}}}\\right)={val}"}

                # ── PRIMITIVE ROOT ────────────────────────────────────────
                elif "primitive root" in p:
                    nums = _nums(p)
                    if nums:
                        pr = sp.ntheory.primitive_root(nums[0])
                        return {"type":"PrimitiveRoot",
                                "result":f"primitive_root({nums[0]}) = {pr}",
                                "latex":f"g={pr}"}

            except Exception:
                pass  # safe fallback to AI

        # ── 7. Solve equation ────────────────────────────────────────
        elif any(k in p for k in ["solve", "roots", "find x"]):
            raw = re.sub(r"(solve|find x|roots of|roots|the equation)", "", p)
            raw = raw.strip().strip(":").strip()
            if "=" in raw:
                lhs_s, rhs_s = raw.split("=", 1)
                lhs_e = parse_expr(clean(lhs_s), transformations=tfms, local_dict=ld)
                rhs_e = parse_expr(clean(rhs_s), transformations=tfms, local_dict=ld)
                expr  = lhs_e - rhs_e
            else:
                expr = parse_expr(clean(raw), transformations=tfms, local_dict=ld)
            if x in expr.free_symbols:
                sol = sp.solve(expr, x)
                sol_latex = ", ".join([sp.latex(s) for s in sol])
                return {
                    "type": "Equation",
                    "result": str(sol),
                    "latex": r"x \in \{" + sol_latex + r"\}"
                }

        # ── 8. Real Analysis II — SymPy for computations, AI for theory ──
        elif any(k in p for k in [
                # Sets & Real Numbers
                "supremum", "infimum", "least upper bound", "greatest lower bound",
                "lub", "glb", "archimedean", "bounded set", "completeness",
                "cartesian product", "density of rational", "real number system",
                "field propert", "order propert",
                # Sequences
                "sequence", "cauchy sequence", "bounded sequence",
                "monotone sequence", "subsequence", "bolzano", "weierstrass",
                # Series
                "ratio test", "root test", "integral test", "comparison test",
                "alternating series", "leibniz test", "absolute convergence",
                "conditional convergence", "cauchy criterion",
                "pointwise convergence", "uniform convergence",
                "weierstrass m-test", "m-test",
                # Limits & Continuity
                "epsilon delta", "epsilon-delta", "uniform continuity",
                "intermediate value", "extreme value theorem",
                # Differentiation theorems
                "mean value theorem", "rolle", "taylor's theorem",
                "lhopital", "l'hopital",
                # Riemann Integration
                "riemann sum", "riemann integral", "upper sum", "lower sum",
                "darboux", "integrability", "fundamental theorem of calculus",
        ]):
            try:
                n_s = sp.Symbol('n', positive=True)

                # ── Sequence limit ────────────────────────────────────────
                if any(k in p for k in ["sequence","limit of sequence"]):
                    # Extract expression after "of" or "for"
                    m = re.search(r"(?:of|for|lim)\s+(.+?)\s*(?:as|when|$)", p)
                    if m:
                        raw = clean(m.group(1))
                        try:
                            expr_s = parse_expr(raw, transformations=tfms,
                                               local_dict={**ld, "n": n_s})
                            lim_val = sp.limit(expr_s, n_s, sp.oo)
                            return {
                                "type": "SequenceLimit",
                                "result": f"lim({m.group(1)}) as n→∞ = {lim_val}",
                                "latex": f"\\lim_{{n\\to\\infty}} = {sp.latex(lim_val)}"
                            }
                        except Exception:
                            pass

                # ── Series sum ────────────────────────────────────────────
                elif any(k in p for k in ["series","sum of"]):
                    m = re.search(r"(?:sum\s+of|series\s+(?:of\s+)?|convergence\s+of)\s*(.+?)(?:\s+from|\s+using|\s+by|$)", p)
                    if m:
                        raw = m.group(1).strip()
                        raw = re.sub(r"^series\s+sum\s+of\s+", "", raw)
                        raw = re.sub(r"^series\s+of\s+", "", raw)
                        raw = re.sub(r"^sum\s+of\s+", "", raw)
                        raw = re.sub(r"^of\s+", "", raw)
                        raw = clean(raw)
                        try:
                            expr_ser = parse_expr(raw, transformations=tfms,
                                                  local_dict={**ld, "n": n_s})
                            s_val = sp.summation(expr_ser, (n_s, 1, sp.oo))
                            converges = s_val.is_finite
                            return {
                                "type": "SeriesConvergence",
                                "result": (f"Series sum = {s_val}, "
                                           f"Converges: {converges}"),
                                "latex": f"\\sum_{{n=1}}^{{\\infty}} = {sp.latex(s_val)}"
                            }
                        except Exception:
                            pass

                # ── Taylor Series ─────────────────────────────────────────
                elif any(k in p for k in ["taylor", "maclaurin"]):
                    funcs = {
                        "sin": sp.sin(x), "cos": sp.cos(x),
                        "exp": sp.exp(x), "e^x": sp.exp(x),
                        "ln": sp.log(1+x), "log": sp.log(1+x),
                        "tan": sp.tan(x)
                    }
                    for fname, fexpr in funcs.items():
                        if fname in p:
                            n_terms = 6
                            ts = sp.series(fexpr, x, 0, n_terms)
                            return {
                                "type": "TaylorSeries",
                                "result": f"Taylor series of {fname}: {ts}",
                                "latex": sp.latex(ts)
                            }

                # ── L'Hopital ─────────────────────────────────────────────
                elif any(k in p for k in ["lhopital","l'hopital"]):
                    m = re.search(r"(?:of|for)\s+(.+?)\s*(?:as|at|when)\s*x\s*[→→=]\s*([\d\.]+|inf)", p)
                    if m:
                        raw = clean(m.group(1))
                        pt_str = m.group(2)
                        pt = sp.oo if pt_str in ("inf","infinity") else sp.sympify(pt_str)
                        try:
                            expr_lh = parse_expr(raw, transformations=tfms, local_dict=ld)
                            lim_val = sp.limit(expr_lh, x, pt)
                            return {
                                "type": "LHopital",
                                "result": f"lim({m.group(1)}) as x→{pt_str} = {lim_val}",
                                "latex": f"\\lim_{{x\\to {pt_str}}} = {sp.latex(lim_val)}"
                            }
                        except Exception:
                            pass

                # ── Riemann Integral ──────────────────────────────────────
                elif any(k in p for k in ["riemann","riemann integral","riemann sum"]):
                    # Try to extract definite integral
                    m = re.search(r"(?:of|for)\s+(.+?)\s+(?:from|on)\s+([\d\.]+)\s+to\s+([\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1))
                        a_v = sp.sympify(m.group(2))
                        b_v = sp.sympify(m.group(3))
                        try:
                            expr_r = parse_expr(raw, transformations=tfms, local_dict=ld)
                            result_r = sp.integrate(expr_r, (x, a_v, b_v))
                            return {
                                "type": "RiemannIntegral",
                                "result": f"∫({m.group(1)}) from {a_v} to {b_v} = {result_r}",
                                "latex": f"\\int_{{{a_v}}}^{{{b_v}}} = {sp.latex(result_r)}"
                            }
                        except Exception:
                            pass

            except Exception:
                pass  # safe fallback to AI for all theory/proof questions

        # ── 9. Differential Geometry — SymPy for computations ──────────
        elif any(k in p for k in [
                "curvature", "torsion", "tangent vector", "normal vector",
                "binormal", "serret-frenet", "frenet", "osculating",
                "arc length", "space curve", "plane curve", "helix", "helices",
                "evolute", "involute", "rectifying plane",
                "first fundamental form", "second fundamental form",
                "fundamental form", "gaussian curvature", "mean curvature",
                "principal curvature", "geodesic",
                "parametric surface", "christoffel", "covariant derivative",
                "contravariant", "metric tensor",
        ]):
            t_s = sp.Symbol('t')
            u_s, v_s = sp.symbols('u v')
            try:

                # ── Curvature of plane curve y=f(x) ──────────────────────
                if "curvature" in p and not any(k in p for k in ["gaussian","mean","space","torsion"]):
                    # Extract function and point
                    m = re.search(r"(?:of|for)\s+y\s*=\s*(.+?)(?:\s+at|\s*$)", p)
                    pt_m = re.search(r"at\s+x\s*[=:]\s*([-\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1).strip())
                        expr_c = parse_expr(raw, transformations=tfms, local_dict=ld)
                        dy  = sp.diff(expr_c, x)
                        d2y = sp.diff(expr_c, x, 2)
                        kappa_expr = sp.Abs(d2y) / (1 + dy**2)**sp.Rational(3,2)
                        if pt_m:
                            pt_val = float(pt_m.group(1))
                            kappa_val = sp.simplify(kappa_expr.subs(x, pt_val))
                            return {
                                "type": "Curvature",
                                "result": f"κ at x={pt_val}: y'={dy.subs(x,pt_val)}, y''={d2y.subs(x,pt_val)}, κ={kappa_val}",
                                "latex": f"\\kappa = {sp.latex(kappa_val)}"
                            }
                        else:
                            return {
                                "type": "Curvature",
                                "result": f"κ(x) = {sp.simplify(kappa_expr)}",
                                "latex": f"\\kappa = {sp.latex(sp.simplify(kappa_expr))}"
                            }

                # ── Arc Length ────────────────────────────────────────────
                elif "arc length" in p:
                    m = re.search(r"(?:of|for)\s+y\s*=\s*(.+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)", p)
                    if m:
                        raw = clean(m.group(1).strip())
                        a_v = sp.sympify(m.group(2))
                        b_v = sp.sympify(m.group(3))
                        expr_al = parse_expr(raw, transformations=tfms, local_dict=ld)
                        dy = sp.diff(expr_al, x)
                        integrand = sp.sqrt(1 + dy**2)
                        L = sp.integrate(integrand, (x, a_v, b_v))
                        L_simplified = sp.simplify(L)
                        return {
                            "type": "ArcLength",
                            "result": f"L = ∫√(1+y'²)dx from {a_v} to {b_v} = {L_simplified}",
                            "latex": f"L = {sp.latex(L_simplified)}"
                        }

                # ── Space Curve: Curvature + Torsion ─────────────────────
                elif any(k in p for k in ["space curve","torsion","frenet","serret"]):
                    # Extract parametric curve r(t) = (x(t), y(t), z(t))
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    pt_m = re.search(r"at\s+t\s*[=:]\s*([-\d\.]+)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "t": t_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "t": t_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "t": t_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        dr = r_vec.diff(t_s)
                        d2r = dr.diff(t_s)
                        d3r = d2r.diff(t_s)
                        speed = sp.sqrt(dr.dot(dr))
                        cross = dr.cross(d2r)
                        kappa = sp.simplify(sp.sqrt(cross.dot(cross)) / speed**3)
                        torsion_val = sp.simplify(cross.dot(d3r) / cross.dot(cross))
                        t_val = float(pt_m.group(1)) if pt_m else 0
                        k_at = sp.simplify(kappa.subs(t_s, t_val))
                        tau_at = sp.simplify(torsion_val.subs(t_s, t_val))
                        T_vec = sp.simplify(dr / speed)
                        return {
                            "type": "FrenetSerret",
                            "result": (f"r(t)={pts[0]}, at t={t_val}:\n"
                                      f"κ = {k_at}, τ = {tau_at}\n"
                                      f"T = {T_vec.subs(t_s,t_val).T}"),
                            "latex": f"\\kappa={sp.latex(k_at)}, \\tau={sp.latex(tau_at)}"
                        }

                # ── First Fundamental Form ────────────────────────────────
                elif "first fundamental form" in p:
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        ru = r_vec.diff(u_s)
                        rv = r_vec.diff(v_s)
                        E = sp.simplify(ru.dot(ru))
                        F = sp.simplify(ru.dot(rv))
                        G = sp.simplify(rv.dot(rv))
                        return {
                            "type": "FirstFundamentalForm",
                            "result": f"E={E}, F={F}, G={G}, ds²={E}du²+{2*F}dudv+{G}dv²",
                            "latex": f"E={sp.latex(E)}, F={sp.latex(F)}, G={sp.latex(G)}"
                        }

                # ── Gaussian + Mean Curvature ─────────────────────────────
                elif any(k in p for k in ["gaussian curvature","mean curvature"]):
                    pts = re.findall(r"\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)", p)
                    if pts:
                        rx = parse_expr(clean(pts[0][0]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        ry = parse_expr(clean(pts[0][1]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        rz = parse_expr(clean(pts[0][2]), transformations=tfms, local_dict={**ld, "u": u_s, "v": v_s})
                        r_vec = sp.Matrix([rx, ry, rz])
                        ru = r_vec.diff(u_s); rv = r_vec.diff(v_s)
                        E = sp.simplify(ru.dot(ru)); F = sp.simplify(ru.dot(rv)); G = sp.simplify(rv.dot(rv))
                        n = ru.cross(rv); N = sp.simplify(n / sp.sqrt(n.dot(n)))
                        L = sp.simplify(N.dot(ru.diff(u_s)))
                        M = sp.simplify(N.dot(ru.diff(v_s)))
                        Nv = sp.simplify(N.dot(rv.diff(v_s)))
                        K = sp.simplify((L*Nv - M**2)/(E*G - F**2))
                        H = sp.simplify((E*Nv - 2*F*M + G*L)/(2*(E*G - F**2)))
                        return {
                            "type": "GaussianCurvature",
                            "result": f"K (Gaussian) = {K}, H (Mean) = {H}",
                            "latex": f"K={sp.latex(K)}, H={sp.latex(H)}"
                        }

            except Exception:
                pass  # safe fallback to AI

        # ── 10. Hydro Mechanics — SymPy for computations, AI for theory ──
        elif any(k in p for k in [
                "continuity equation", "equation of continuity",
                "streamline", "stream function", "stream line",
                "velocity potential", "irrotational", "rotational motion",
                "lagrangian", "eulerian", "vortex", "vorticity",
                "path line", "streak line",
                "bernoulli", "euler's equation", "euler equation of motion",
                "torricelli", "flow rate", "discharge",
                "reynolds number", "reynolds",
                "hydrostatic pressure", "pressure at depth",
                "hydrostatic", "buoyancy", "archimedes",
                "laminar flow", "turbulent flow", "viscous flow",
                "incompressible fluid", "steady flow",
                "navier-stokes", "navier stokes",
                "stokes stream function", "complex velocity potential",
                "source", "sink", "doublet",
                "milne thomson", "blasius theorem",
                "dimensional analysis", "buckingham pi",
        ]):
            try:
                x_h, y_h = sp.symbols('x y')
                g_val = sp.Rational(981, 100)  # 9.81

                # ── Continuity: find v2 from A1v1=A2v2 ───────────────
                if any(k in p for k in ["continuity equation","equation of continuity"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 3:
                        A1_v,v1_v,A2_v = nums[0],nums[1],nums[2]
                        v2_v = round(A1_v*v1_v/A2_v, 6)
                        Q_v  = round(A1_v*v1_v, 6)
                        return {
                            "type": "ContinuityEq",
                            "result": (f"A1={A1_v}, v1={v1_v}, A2={A2_v}\n"
                                      f"v2 = A1*v1/A2 = {v2_v} m/s\n"
                                      f"Flow rate Q = A1*v1 = {Q_v} m³/s"),
                            "latex": f"v_2 = {v2_v}\\text{{ m/s}}"
                        }

                # ── Reynolds Number ────────────────────────────────────
                elif any(k in p for k in ["reynolds number","reynolds"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 4:
                        rho_v,v_v,D_v,mu_v = nums[0],nums[1],nums[2],nums[3]
                        Re = round(rho_v*v_v*D_v/mu_v, 2)
                        flow = "Turbulent (Re>4000)" if Re>4000 else ("Transitional (2300<Re<4000)" if Re>2300 else "Laminar (Re<2300)")
                        return {
                            "type": "ReynoldsNumber",
                            "result": f"Re = ρvD/μ = {rho_v}×{v_v}×{D_v}/{mu_v} = {Re} → {flow}",
                            "latex": f"Re = {Re}"
                        }

                # ── Bernoulli: find P2 ─────────────────────────────────
                elif "bernoulli" in p:
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 5:
                        P1_v,v1_v,h1_v,v2_v,h2_v = nums[0],nums[1],nums[2],nums[3],nums[4]
                        rho_v = 1000  # default water
                        P2_v = round(P1_v + 0.5*rho_v*(v1_v**2-v2_v**2) + rho_v*9.81*(h1_v-h2_v), 4)
                        return {
                            "type": "Bernoulli",
                            "result": (f"P1+½ρv1²+ρgh1 = P2+½ρv2²+ρgh2\n"
                                      f"P2 = {P2_v} Pa"),
                            "latex": f"P_2 = {P2_v}\\text{{ Pa}}"
                        }

                # ── Torricelli: v = √(2gh) ─────────────────────────────
                elif "torricelli" in p:
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if nums:
                        h_v = nums[0]
                        v_torr = round((2*9.81*h_v)**0.5, 6)
                        return {
                            "type": "Torricelli",
                            "result": f"v = √(2gh) = √(2×9.81×{h_v}) = {v_torr} m/s",
                            "latex": f"v = {v_torr}\\text{{ m/s}}"
                        }

                # ── Hydrostatic Pressure ───────────────────────────────
                elif any(k in p for k in ["hydrostatic pressure","pressure at depth"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if nums:
                        h_v = nums[0]
                        rho_v = 1000
                        P_gauge = round(rho_v*9.81*h_v, 4)
                        P_abs = round(101325 + P_gauge, 4)
                        return {
                            "type": "HydrostaticPressure",
                            "result": (f"At depth h={h_v}m:\n"
                                      f"Gauge pressure = ρgh = {P_gauge} Pa\n"
                                      f"Absolute pressure = P0+ρgh = {P_abs} Pa"),
                            "latex": f"P = P_0 + \\rho g h = {P_abs}\\text{{ Pa}}"
                        }

                # ── Flow Rate ──────────────────────────────────────────
                elif any(k in p for k in ["flow rate","discharge"]):
                    nums = [float(n) for n in re.findall(r"[-]?\d+\.?\d*", p)]
                    if len(nums) >= 2:
                        A_v, v_v = nums[0], nums[1]
                        Q_v = round(A_v*v_v, 8)
                        return {
                            "type": "FlowRate",
                            "result": f"Q = A×v = {A_v}×{v_v} = {Q_v} m³/s",
                            "latex": f"Q = {Q_v}\\text{{ m³/s}}"
                        }

                # ── Velocity Potential ─────────────────────────────────
                elif "velocity potential" in p:
                    m = re.search(r"(?:phi|φ|potential)\s*=\s*(.+?)(?:\s|$)", p)
                    if m:
                        raw = clean(m.group(1))
                        phi_expr = parse_expr(raw, transformations=tfms, local_dict={**ld,"x":x_h,"y":y_h})
                        u_comp = sp.diff(phi_expr, x_h)
                        v_comp = sp.diff(phi_expr, y_h)
                        lap = sp.diff(phi_expr,x_h,2) + sp.diff(phi_expr,y_h,2)
                        return {
                            "type": "VelocityPotential",
                            "result": (f"φ={str(phi_expr)}, u=∂φ/∂x={u_comp}, v=∂φ/∂y={v_comp}\n"
                                      f"∇²φ={sp.simplify(lap)} (irrotational: {sp.simplify(lap)==0})"),
                            "latex": f"\\nabla^2\\phi = {sp.latex(sp.simplify(lap))}"
                        }

            except Exception:
                pass  # safe fallback to AI

        # ── 11. Matrix / Eigenvalues — deterministic SymPy adapter ────────
        elif any(k in p for k in ["matrix", "determinant", "eigenvalue",
                                   "eigenvector", "det(", "inverse matrix", "rank"]):
            rows = re.findall(r"\[([^\[\]]+)\]", p)
            if rows:
                try:
                    matrix_data = [
                        [sp.Rational(value.strip()) for value in row.split(",")]
                        for row in rows
                    ]
                    if len({len(row) for row in matrix_data}) != 1:
                        raise ValueError("Matrix rows have different lengths")
                    matrix = sp.Matrix(matrix_data)

                    if "determinant" in p or "det(" in p:
                        value = sp.factor(matrix.det())
                        return {"type": "MatrixDeterminant", "result": f"det(A) = {value}", "latex": f"\\det(A)={sp.latex(value)}"}
                    if "eigenvector" in p:
                        value = matrix.eigenvects()
                        return {"type": "Eigenvectors", "result": f"Eigenvectors: {value}", "latex": sp.latex(value)}
                    if "eigenvalue" in p:
                        value = matrix.eigenvals()
                        return {"type": "Eigenvalues", "result": f"Eigenvalues: {value}", "latex": sp.latex(value)}
                    if "inverse" in p:
                        value = matrix.inv()
                        return {"type": "MatrixInverse", "result": f"A^(-1) = {value}", "latex": sp.latex(value)}
                    if "transpose" in p:
                        value = matrix.T
                        return {"type": "MatrixTranspose", "result": f"A^T = {value}", "latex": sp.latex(value)}
                    if "rank" in p:
                        value = matrix.rank()
                        return {"type": "MatrixRank", "result": f"rank(A) = {value}", "latex": f"\\operatorname{{rank}}(A)={value}"}
                    return {"type": "Matrix", "result": f"A = {matrix}", "latex": sp.latex(matrix)}
                except Exception:
                    pass

        # ── 12. Modular arithmetic — deterministic adapter ───────────────
        elif "mod" in p or "congruence" in p:
            congruence = re.search(
                r"([+-]?\d+)\s*x\s*(?:≡|=)\s*([+-]?\d+)\s*\(?(?:mod|modulo)\s*([+-]?\d+)\)?",
                p,
            )
            if congruence:
                a_val, b_val, modulus = (int(value) for value in congruence.groups())
                gcd_value = math.gcd(a_val, modulus)
                if b_val % gcd_value != 0:
                    return {
                        "type": "LinearCongruence",
                        "result": f"No solution because gcd({a_val},{modulus})={gcd_value} does not divide {b_val}.",
                        "latex": "\\text{No solution}",
                    }
                solutions = [x_val for x_val in range(modulus) if (a_val * x_val - b_val) % modulus == 0]
                return {
                    "type": "LinearCongruence",
                    "result": f"{a_val}x ≡ {b_val} (mod {modulus}); solutions: {solutions}",
                    "latex": "x \\equiv " + ", \\".join(str(x_val) for x_val in solutions) + f" \\pmod{{{modulus}}}",
                }

            remainder = re.search(r"([+-]?\d+)\s+mod\s+([+-]?\d+)", p)
            if remainder:
                left, right = (int(value) for value in remainder.groups())
                value = left % right
                return {"type": "Modulo", "result": f"{left} mod {right} = {value}", "latex": f"{left} \\bmod {right} = {value}"}

    except Exception:
        pass  # Silently fall back — AI handles it

    return {"type": "general", "result": None, "latex": ""}

