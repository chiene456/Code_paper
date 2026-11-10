import time
import numpy as np
import pandas as pd
from . import algorithms as alg


def residual(P, x, lam):
    """TOL(k) := || x_k - J_{λ_k}^A (x_k - λ_k B x_k) ||."""
    return P["norm"](x - P["J"](x - lam * P["B"](x), lam))


def run(P, state0, step, tol, max_iter):
    s, k = state0, 0
    t0 = time.perf_counter()
    res = residual(P, s["x"], s["lam"])
    hist, times = [res], [0.0]
    while res > tol and k < max_iter:
        k += 1
        s = step(s, k)
        res = residual(P, s["x"], s["lam"])
        hist.append(res)
        times.append(time.perf_counter() - t0)
    return dict(iters=k, cpu=times[-1], tol=res, hist=np.array(hist),
                times=np.array(times), x=s["x"], converged=res <= tol)


def build_method(name, P, params, f, F):
    p = params[name]
    if name == "ours":
        return alg.make_ours(P, p, f)
    if name == "alg1":
        return alg.make_alg1(P, p)
    if name == "alg2":
        return alg.make_alg2(P, p)
    if name == "alg3":
        return alg.make_alg3(P, p, F)
    if name == "alg4":
        return alg.make_alg4(P, p, f)
    raise ValueError(name)


def run_example(P, params, f, F, tol, max_iter, methods=None, verbose=True):
    methods = methods or list(params)
    out = {}
    for m in methods:
        state0, step = build_method(m, P, params, f, F)
        out[m] = run(P, state0, step, tol, max_iter)
        if verbose:
            r = out[m]
            print(f"  {m:5s} iters={r['iters']:7d}  cpu={r['cpu']:9.4f}s  TOL={r['tol']:.3e}"
                  f"{'' if r['converged'] else '  (CHƯA hội tụ)'}")
    return out


def results_table(results, labels):
    rows = [dict(Method=labels[m], Iterations=r["iters"],
                 **{"CPU time (s)": r["cpu"], "TOL(k)": r["tol"]})
            for m, r in results.items()]
    return pd.DataFrame(rows)
