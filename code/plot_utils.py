"""
Figures for Question 5.

Kept apart from main_andres.py so that question5() reads as what it does
(compute the guarantee, draw two figures) rather than as sixty lines of
matplotlib.  Nothing here imports main_andres: every quantity a figure needs
is passed in, so the two modules stay independent and these functions can be
called on a saved history without running gradient descent again:

    import numpy as np, plot_utils
    h = np.load("../results/q4_gd_history.npz")
    ...

The backend is set by the caller (main_andres.py selects "Agg" before
importing this module).
"""

import numpy as np
import matplotlib.pyplot as plt

# categorical slots 1 and 2, fixed order (validated for light-mode line charts)
OBSERVED_COLOR = "#2a78d6"
THEORY_COLOR = "#eb6834"
RULE_COLOR = "#52514e"


def plot_convergence(objectives, grad_norms, bound, c, results_dir):
    """Q5 deliverable: f_lambda(theta_k) and ||grad f_lambda(theta_k)|| vs k.

    `bound` is Corollary 4.32 evaluated at the same k, and is drawn on the
    gradient panel.  Over the run's own range it is visually flat, which is
    the comparison the question asks for.
    """
    k = np.arange(objectives.size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(k, objectives, color=OBSERVED_COLOR, lw=2)
    axes[0].set(xlabel="Iteration $k$", ylabel=r"$f_\lambda(\theta_k)$",
                title="Objective function")

    axes[1].semilogy(k, grad_norms, color=OBSERVED_COLOR, lw=2,
                     label="observed", zorder=3)
    axes[1].semilogy(k, bound, color=THEORY_COLOR, lw=1.8, ls="--",
                     label=r"theoretical bound, $\alpha=%g/L$" % c)
    axes[1].set_ylim(grad_norms[-1] / 3.0, bound[0] * 4.0)
    axes[1].set(xlabel="Iteration $k$",
                ylabel=r"$\|\nabla f_\lambda(\theta_k)\|$",
                title="Gradient norm: run vs guarantee")
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)

    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(results_dir / "q5_convergence.pdf")
    plt.close(fig)


def plot_theory_horizon(grad_norms, prefactor, rho, c, tol, results_dir):
    """Second figure: the same guarantee carried out to where it lands.

    We run a second graph with log x-axis too to be able to clearly see the
    comparison between theoretical and experimental results up to the end.
    The y range is clipped to the band between the tolerance and the bound's
    starting value: past the crossing the bound falls away steeply and only
    the crossing point is of interest.
    """
    target = tol * grad_norms[0]
    kk = np.logspace(0, 11.3, 500)
    K = grad_norms.size - 1
    n_needed = 2.0 * np.log(tol) / np.log(rho)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.loglog(np.arange(1, grad_norms.size), grad_norms[1:],
              color=OBSERVED_COLOR, lw=2, label="observed run", zorder=3)
    ax.loglog(kk, prefactor * rho ** (kk / 2.0), color=THEORY_COLOR, lw=1.8,
              ls="--", label=r"theoretical bound, $\alpha=%g/L$" % c)
    ax.axhline(target, color=RULE_COLOR, ls=":", lw=1.2, zorder=1)

    # the bound's crossing sits near the right edge, so its label is anchored
    # leftward to keep it inside the axes
    for xpos, col, lab, ha, dx in (
            (K, OBSERVED_COLOR, "run stops\n%s iters" % f"{K:,}", "center", 0),
            (n_needed, THEORY_COLOR,
             "bound gets there\nat %.2g iters" % n_needed, "right", -6)):
        ax.plot([xpos], [target], "o", color=col, ms=6, mec="white", mew=1.2,
                zorder=4)
        ax.annotate(lab, xy=(xpos, target), xytext=(dx, -26),
                    textcoords="offset points", color=col, fontsize=8,
                    ha=ha, fontweight="bold")

    ax.set_ylim(target / 12.0, prefactor * 3.0)
    ax.set_xlim(1, 2e11)
    ax.annotate(r"stopping rule  $\|\nabla f_\lambda(\theta_k)\|\leq 10^{-3}"
                r"\|\nabla f_\lambda(\theta_0)\|$",
                xy=(1.0, target), xytext=(3, 6), textcoords="offset points",
                color=RULE_COLOR, fontsize=8)
    ax.set(xlabel="Iteration $k$ (log scale)",
           ylabel=r"$\|\nabla f_\lambda(\theta_k)\|$",
           title="Convergence over the full horizon")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(results_dir / "q5_theory_horizon.pdf")
    plt.close(fig)
