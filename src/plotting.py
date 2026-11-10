import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})


STYLES = {
    "alg1": ("#800080", "X", 9),
    "alg2": ("#009900", "o", 9),
    "alg3": ("#0000FF", "d", 9),
    "alg4": ("#FF8C00", "s", 9),
    "ours": ("#FF0000", "p", 10),
}
ORDER = ["alg1", "alg2", "alg3", "alg4", "ours"]


def plot_convergence(results, labels, title=None, tol=None, path=None,
                     floor=1e-5, ylim=None):

    fig = plt.figure(figsize=(14, 6))
    methods = [m for m in ORDER if m in results]

    for pos, xkey in ((1, "iters"), (2, "time")):
        plt.subplot(1, 2, pos)
        for m in methods:
            r = results[m]
            tols = np.maximum(r["hist"][1:], floor)              # k = 1, 2, ...
            x = np.arange(1, len(tols) + 1) if xkey == "iters" else r["times"][1:]
            color, marker, ms = STYLES[m]
            plt.semilogy(x, tols, label=labels[m], color=color, marker=marker,
                         markevery=max(1, len(tols) // 15), linewidth=2.5,
                         markersize=ms, markerfacecolor=color,
                         markeredgecolor=color, markeredgewidth=1.5,
                         zorder=10 if m == "ours" else 5)
        if tol is not None:
            plt.axhline(tol, color="gray", lw=1.0, ls="--")
        if ylim is not None:
            plt.ylim(1e8,1e-5)
        plt.xlabel("Iterations ($k$)" if xkey == "iters" else "Time ($s$)")
        plt.ylabel("TOL($k$)")
        plt.legend(loc="upper right", frameon=True, edgecolor="black")
        plt.grid(True, which="major", color="#cccccc", linestyle="--")

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ tại: {path}")
    return fig