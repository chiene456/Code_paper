"""Chạy các ví dụ.
    python run_all.py                               # Ex 4.1-4.4, err = 1e-5
    python run_all.py --examples 4.1 4.2            # chọn ví dụ
    python run_all.py --tol 1e-8 --x0-case 2t_16
"""
import argparse
import os

import numpy as np

import config as C
from src import problems
from src.plotting import plot_convergence
from src.runner import results_table, run_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", nargs="+", default=list(problems.BUILDERS))
    ap.add_argument("--tol", type=float, default=C.TOL)
    ap.add_argument("--x0-case", default="t2_10", help="điểm khởi tạo cho Ex 4.1")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    os.makedirs(f"{a.outdir}/figures", exist_ok=True)
    os.makedirs(f"{a.outdir}/tables", exist_ok=True)
    for ex in a.examples:
        kw = dict(x0_case=a.x0_case) if ex == "4.1" else dict(seed=C.SEED)
        P = problems.BUILDERS[ex](**kw)
        print(f"\n=== Example {ex}: {P['name']}  (ell = {P['ell']:.4g}, tol = {a.tol:g}) ===")
        res = run_example(P, C.PARAMS, C.f_visc, C.F_alg3, a.tol,
                          a.max_iter or C.MAX_ITER[ex])
        tab = results_table(res, C.LABELS)
        tab.to_csv(f"{a.outdir}/tables/example_{ex}.csv", index=False)
        with open(f"{a.outdir}/tables/example_{ex}.md", "w") as fh:
            fh.write(tab.to_markdown(index=False, floatfmt=".4g"))
        plot_convergence(res, C.LABELS, f"Example {ex} — {P['name']}", a.tol,
                         f"{a.outdir}/figures/example_{ex}.png")


if __name__ == "__main__":
    main()
