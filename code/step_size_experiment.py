"""
Step-size selection for Question 4. Ran before submission to get CHOSEN_C. 
Not run by teachers, or imported/called by main.py.

Writes into ../results/q4_stepsize_selection(disregard)/ , a folder kept apart because
"python main.py" does not regenerate these two files:
        q4_step_size_experiment.txt
        q4_step_size_comparison.pdf


Two experiments :

  1) Short runs (500 iterations) from 3 random chosen theta_0.  
    This was motivated by the concern on if the starting point was affecting 
    the ranking of the candidate step size. We can conclude from this experiment
    that it is not the case for this function. This allows us to then run a single
    (longer) run on experiment 2, but also to observe that larger steps yield faster
    convergence in practice - the opposite of what we could have concluded from the
    guarantee given in Theorem 4.31.

  2) One theta_0, every candidate run to the Q4 stopping rule
     ||grad f(theta_k)|| <= TOL * ||grad f(theta_0)||.  
     This one was motivated by the concern of slower convergence around the optimum
     due to a larger step which would have yield well in experiment 1 - far from the
     optimum. It invalidated our concerns and showed us something even more interesting,
     convergence speed scales linearly with the step size, even further than the 2/L 
     theoretical bound for garanteed convergence.
     
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_data, signs, run_gd, LAM, TOL, SEED, RESULTS_DIR

# create folder to seperate with main's results
OUT_DIR = RESULTS_DIR / "q4_stepsize_selection(disregard)"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# candidates within theoretical range
SWEEP_C = [0.5, 1.0, 1.5, 1.9]
# Probes beyond the guarantee (test how conservative the bound is)
PROBE_C = [2.5, 4.0, 7.0, 10.0, 15.0] + [float(c) for c in range(25, 201, 25)]          # updated

# experiment 1 parameters
SHORT_ITERS = 100
SHORT_SEEDS = [42, 0, 7]        # test seeds 

# experiment 2 parameters
LONG_MAX_ITER = 60000
LONG_MAX_TIME = 300.0

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# ------------ Experiment 1 ------------
def short_runs(X, s, L, all_c):
    orderings = []
    for seed in SHORT_SEEDS:
        theta0 = np.random.default_rng(seed).standard_normal(X.shape[0])

        # no tolerance is passed, so each run does the full SHORT_ITERS steps
        # and gn[-1] is the gradient norm at that iteration
        ratio = {}
        for c in all_c:
            _, _, gn, _, _ = run_gd(X, s, L, c, theta0, max_iter=SHORT_ITERS)
            ratio[c] = gn[-1] / gn[0]

        # best (smallest ratio) first; a diverged run gives nan and sorts last
        order = sorted(all_c, key=lambda c: (np.isnan(ratio[c]), ratio[c]))
        orderings.append(tuple(order))

    identical = len(set(orderings)) == 1
    return orderings, identical


# ------------ Experiment 2 ------------
def long_runs(X, s, L, all_c):
    theta0 = np.random.default_rng(SEED).standard_normal(X.shape[0])
    results = {}
    for c in all_c:
        _, obj, gn, reason, secs = run_gd(
            X, s, L, c, theta0, tol=TOL,
            max_iter=LONG_MAX_ITER, max_time=LONG_MAX_TIME)
        iters = gn.size - 1 if reason == "gradient_tolerance" else None

        ratio = gn / gn[0]
        increases = int(np.sum(np.diff(ratio) > 0))
        results[c] = dict(objectives=obj, grad_norms=gn, reason=reason,
                          seconds=secs, iters=iters, increases=increases)

    return results


# ------------ Results ------------
def make_plot(results, all_c):
    fig, ax = plt.subplots(figsize=(8, 5))

    for j, (c, color) in enumerate(zip(all_c, COLORS)):
        gn = results[c]["grad_norms"]
        ratio = gn / gn[0]
        k = np.arange(ratio.size)
        ax.semilogy(k, ratio, color=color, lw=2, label=f"$c = {c:g}$")
        ax.plot(k[-1], ratio[-1], "o", color=color, ms=5,
                mec="white", mew=1.2, zorder=3)
        ax.annotate(f"{c:g}", xy=(k[-1], ratio[-1]),
                    xytext=(3, 8 + 9 * (j % 2)), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", clip_on=False)
    ax.axhline(TOL, color="#52514e", ls="--", lw=1, zorder=1)
    #
    ax.set_ylim(bottom=3.5e-4)
    ax.annotate(f"stopping rule:  $\\|g_k\\| \\leq {TOL:g}\\,\\|g_0\\|$",
                xy=(0.0, TOL), xycoords=("axes fraction", "data"),
                xytext=(6, -14), textcoords="offset points",
                color="#52514e", fontsize=9, ha="left")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|\nabla f_\lambda(\theta_k)\| \,/\, \|\nabla f_\lambda(\theta_0)\|$")
    ax.set_title(r"Gradient-norm decay for $\alpha = c/L$   (seed %d)" % SEED)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", title="step $\\alpha = c/L$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q4_step_size_comparison.pdf")
    plt.close(fig)


def write_report(L, orderings, identical, results, all_c):
    """
    1: the ranking of the candidates does not depend on the starting point.
    2: the iteration count to the stopping rule, and its 1/c scaling.
    """
    with open(OUT_DIR / "q4_step_size_experiment.txt", "w") as fh:
        fh.write("Step-size selection for Question 4\n\n")
        fh.write(f"lambda = {LAM}\nL = {L:.12e}\n2/L = {2.0 / L:.12e}\n")
        fh.write(f"stopping rule: ||g_k|| <= {TOL:g} * ||g_0||\n")
        fh.write(f"candidates: {SWEEP_C} within 2/L, {PROBE_C} beyond it\n\n")

        fh.write(f"Experiment 1 ({SHORT_ITERS} iterations, seeds {SHORT_SEEDS})\n")
        fh.write("Ranked by ||g_k||/||g_0||\n\n")
        for seed, order in zip(SHORT_SEEDS, orderings):
            fh.write(f"     seed {seed:>2}:  "
                     + "  <  ".join(f"{c:g}" for c in order) + "\n")
        fh.write(f"\n   ranking identical across all seeds: {identical}\n\n\n")

        fh.write(f"Experiment 2 (seed {SEED})\n\n")
        fh.write("        c   iterations   c * iterations      f_final   increases\n")
        products = []
        for c in all_c:
            it = results[c]["iters"]
            if it is None:
                fh.write(f"     {c:4.1f}   not reached\n")
                continue
            products.append(c * it)
            fh.write(f"     {c:4.1f}   {it:10d}   {c * it:14.0f}   "
                     f"{results[c]['objectives'][-1]:10.4e}   "
                     f"{results[c]['increases']:9d}\n")
        reached = [c for c in all_c if results[c]["iters"] is not None]

        if reached:
            within = [c for c in reached if c in SWEEP_C]
            if within:
                bw = min(within, key=lambda c: results[c]["iters"])
                fh.write(f"   fastest within the 2/L guarantee: c = {bw} "
                         f"({results[bw]['iters']} iterations)  <- CHOSEN_C\n")



# ------------ Run ------------
def main():
    X, y, _, _ = load_data()
    s = signs(y)
    L = float(np.linalg.eigvalsh(X @ X.T)[-1]) + LAM
    all_c = SWEEP_C + PROBE_C

    orderings, identical = short_runs(X, s, L, all_c)
    results = long_runs(X, s, L, all_c)
    make_plot(results, all_c)
    write_report(L, orderings, identical, results, all_c)
    


if __name__ == "__main__":
    main()
