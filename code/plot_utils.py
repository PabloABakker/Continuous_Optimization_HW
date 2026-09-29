"""
Figures and file output for Questions 3, 4 and 5, in that order.

Kept apart from main.py so that question3(), question4() and question5()
read as what they do -- check the gradient, run the method, compare with the
guarantee -- rather than as sixty lines of matplotlib and twenty of file
writing.  Nothing here imports
main: every quantity is passed in, so the two modules stay independent and
the figures can be redrawn from a saved history without running gradient
descent again:

    import numpy as np, plot_utils
    d = np.loadtxt("../results/q4_gd_history.csv", delimiter=",", skiprows=1)
    ...

The backend is set by the caller (main.py selects "Agg" before
importing this module).
"""

import numpy as np
import matplotlib.pyplot as plt

# categorical slots 1 and 2, fixed order (validated for light-mode line charts)
OBSERVED_COLOR = "#2a78d6"
THEORY_COLOR = "#eb6834"
RULE_COLOR = "#52514e"


def plot_gradient_check(t, err, mask, floor, results_dir):
    """
    Q3 deliverable: the Taylor remainder against t, in log-log coordinates.

    `mask` is the window the slope was fitted on, shaded here so the figure
    shows which points the number came from.  The O(t^2) reference is scaled
    to sit on the data, so what matters is that the two are parallel, not
    that they coincide.
    """
    if not mask.any():
        raise ValueError("empty fit window: every remainder is below the "
                         "round-off floor, so the slope could not be measured")
    C = np.median(err[mask] / t[mask] ** 2)

    # the fit uses err > floor, but the curve is drawn over every t, and a
    # log axis cannot show a remainder that cancelled to exactly zero
    pos = err > 0.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t[pos], err[pos], label="Taylor remainder")
    ax.loglog(t, C * t ** 2, "--", label=r"$O(t^2)$ reference")
    ax.axhline(floor, color="gray", ls=":", lw=1, label="round-off floor")
    ax.axvspan(t[mask][0], t[mask][-1], color="gray", alpha=0.12,
               label="fit window")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|f_\lambda(\theta+tv)-f_\lambda(\theta)"
                  r"-t\langle v,\nabla f_\lambda(\theta)\rangle|$")
    ax.set_title("Gradient check")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "q3_gradient_check.pdf")
    plt.close(fig)


def save_q4_results(results_dir, k, objectives, grad_norms, theta0,
                    reason, elapsed, *, step, c, sigma_max, L, lam, seed,
                    tol, max_time):
    """Write the three files Question 4 asks for.

    q4_gd_history.csv is the per-iteration record the question requires:
    one row per iterate k = 0 .. K, the initial point included.
    q4_stopping_reason.txt carries the stopping reason and the summary
    values the write-up cites, so they need not be dug out of the first and
    last rows of an 11572-line file.

    The configuration is keyword-only, so the call site names each constant.
    """

    np.savetxt(results_dir / "q4_gd_history.csv",
               np.column_stack((np.arange(k + 1), objectives, grad_norms)),
               delimiter=",", fmt=["%d", "%.18e", "%.18e"],
               header="k,objective,gradient_norm", comments="")

    with open(results_dir / "q4_stopping_reason.txt", "w") as fh:
        fh.write(f"Stopping reason: {reason}\n")
        fh.write(f"Iterations: {k}\n")
        fh.write(f"Elapsed time: {elapsed:.2f} s   (limit {max_time:.0f} s)\n")
        fh.write(f"Step size: alpha = {c}/L = {step:.12e}\n")
        fh.write(f"L = {L:.12e}   sigma_max(X) = {sigma_max:.12e}   "
                 f"lambda = {lam}   seed = {seed}\n")
        fh.write("(step size chosen from step_size_experiment.py; "
                 "see q4_stepsize_selection(disregard)/)\n\n")
        fh.write(f"stopping rule: ||g_k|| <= {tol:g} * ||g_0||\n\n")
        fh.write(f"Objective       f(theta_0)     = {objectives[0]:.6e}\n")
        fh.write(f"                f(theta_final) = {objectives[-1]:.6e}"
                 f"     (ratio {objectives[0] / objectives[-1]:.3e})\n")
        fh.write(f"Gradient norm   ||g_0||        = {grad_norms[0]:.6e}\n")
        fh.write(f"                ||g_final||    = {grad_norms[-1]:.6e}"
                 f"     (ratio {grad_norms[-1] / grad_norms[0]:.3e})\n")


def plot_convergence(objectives, grad_norms, bound, c, results_dir):
    """
    Q5 deliverable: f_lambda(theta_k) and ||grad f_lambda(theta_k)|| vs k.

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
    # a diverged run ends in inf or nan, which would make the limits unusable
    finite = grad_norms[np.isfinite(grad_norms) & (grad_norms > 0.0)]
    axes[1].set_ylim(finite.min() / 3.0, bound[0] * 4.0)
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
    """
    Second figure: the same guarantee carried out to where it lands.

    We run a second graph with log x-axis too to be able to clearly see the
    comparison between theoretical and experimental results up to the end.
    The y range is clipped to the band between the tolerance and the bound's
    starting value: past the crossing the bound falls away steeply and only
    the crossing point is of interest.
    """
    target = tol * grad_norms[0]
    kk = np.logspace(0, 11.3, 500)
    K = grad_norms.size - 1
    # k where the bound meets the stopping rule: solve prefactor*rho^(k/2) = target.
    # Not 2 ln(tol)/ln(rho): the bound starts above ||g_0||, so it falls further.
    n_needed = 2.0 * np.log(target / prefactor) / np.log(rho)

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
