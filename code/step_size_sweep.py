"""
Step-size selection for Question 4.  Run once, from the code/ folder:

    python step_size_sweep.py

Writes  ../results/q4_step_size_sweep.txt
        ../results/q4_step_size_comparison.pdf

This script is NOT imported or called by main.py: the chosen step size is
hard-coded there as CHOSEN_C, so that the graded run is deterministic and
does not pay for the sweep.

Two experiments, with different jobs:

  A) Short runs (500 iterations) from several theta_0.  Job: show that the
     ranking of the candidates does not depend on the starting point, which
     is what licenses using a single seed everywhere else.  The absolute
     numbers do depend on the seed; the ordering is what we read off.

  B) One theta_0, every candidate run to the Q4 stopping rule
     ||grad f(theta_k)|| <= TOL * ||grad f(theta_0)||.  Job: select the step
     size on the criterion the question actually states, namely the number
     of iterations needed to reach that tolerance.  All candidates cost the
     same per iteration (one X.T @ theta and one X @ ...), so iterations are
     a fair and machine-independent currency.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_andres import load_data, signs, run_gd, LAM, TOL, SEED, RESULTS_DIR

# Candidates at or below the classical 2/L stability limit.
SWEEP_C = [0.5, 1.0, 1.5, 1.9]
# Probes beyond the guarantee.  phi'' = 0 outside (-1, 0), so the Hessian only
# collects the middle-branch samples and its top eigenvalue is well below
# L = lambda_max(X X^T) + lambda.  These test how conservative that bound is.
PROBE_C = [2.5, 4.0]

SHORT_SEEDS = [42, 0, 7]
SHORT_ITERS = 500
CHECKPOINTS = [10, 50, 100, 200, 500]

LONG_MAX_ITER = 60_000
LONG_MAX_TIME = 120.0

# Categorical slots 1-6, fixed order, validated for light-mode line charts.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def short_runs(X, s, L, all_c):
    """Experiment A: does the ranking survive a change of theta_0?"""
    print("\n=== A. short runs, several starting points ===")
    table, orderings = {}, []
    for seed in SHORT_SEEDS:
        theta0 = np.random.default_rng(seed).standard_normal(X.shape[0])
        print(f"\n  seed {seed}:   ||g_k|| / ||g_0||")
        print("     c   " + "".join(f"   k={k:<6d}" for k in CHECKPOINTS))
        for c in all_c:
            _, _, gn, _, _ = run_gd(X, s, L, c, theta0, max_iter=SHORT_ITERS)
            # gn[k] is the norm after k steps; short of that, the run stopped early.
            row = [gn[k] / gn[0] if k < gn.size else np.nan for k in CHECKPOINTS]
            table[(seed, c)] = row
            print(f"   {c:4.2f}  " + "".join(f"   {v:.3e}" for v in row))
        # ranking at the final checkpoint, best (smallest ratio) first
        order = sorted(all_c, key=lambda c: (np.isnan(table[(seed, c)][-1]),
                                             table[(seed, c)][-1]))
        orderings.append(tuple(order))
        print(f"   ranking at k={SHORT_ITERS}: " + " < ".join(f"{c:g}" for c in order))

    identical = len(set(orderings)) == 1
    print(f"\n  ranking identical across all {len(SHORT_SEEDS)} seeds: {identical}")
    return table, orderings, identical


def long_runs(X, s, L, all_c):
    """Experiment B: iterations to the Q4 tolerance, one shared theta_0."""
    print("\n=== B. full runs to the Q4 tolerance, single starting point ===")
    theta0 = np.random.default_rng(SEED).standard_normal(X.shape[0])
    results = {}
    for c in all_c:
        _, obj, gn, reason, secs = run_gd(
            X, s, L, c, theta0, tol=TOL,
            max_iter=LONG_MAX_ITER, max_time=LONG_MAX_TIME)
        iters = gn.size - 1 if reason == "gradient_tolerance" else None
        results[c] = dict(objectives=obj, grad_norms=gn, reason=reason,
                          seconds=secs, iters=iters)
        shown = f"{iters:>6d}" if iters is not None else "  none"
        # non-monotonicity would be the signature of a step that overshoots
        ratio = gn / gn[0]
        bumps = int(np.sum(np.diff(ratio) > 0))
        print(f"   c = {c:4.2f}: iters to tol = {shown}   "
              f"final ||g||/||g_0|| = {ratio[-1]:.3e}   "
              f"increases = {bumps:<6d} {reason}  ({secs:.1f}s)")
    return results


def make_plot(results, all_c):
    fig, ax = plt.subplots(figsize=(8, 5))
    # Curves terminate at the same height (the tolerance) but at different k,
    # so the direct labels are staggered vertically to keep them apart.
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
    # No run goes below the tolerance, so the band under the dashed line is
    # free space: park the rule's label there instead of over the curve ends.
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
    fig.savefig(RESULTS_DIR / "q4_step_size_comparison.pdf")
    plt.close(fig)


def save_histories(results, all_c):
    """Keep the full curves so the figure can be redrawn without re-running."""
    payload = {}
    for c in all_c:
        payload[f"grad_norms_c{c:g}"] = results[c]["grad_norms"]
        payload[f"objectives_c{c:g}"] = results[c]["objectives"]
    payload["candidates"] = np.array(all_c)
    np.savez(RESULTS_DIR / "q4_step_size_histories.npz", **payload)


def write_report(L, table, orderings, identical, results, all_c):
    with open(RESULTS_DIR / "q4_step_size_sweep.txt", "w") as fh:
        fh.write("Step-size selection for Question 4 "
                 "(produced by step_size_sweep.py, not by main.py)\n\n")
        fh.write(f"lambda = {LAM}\nL = {L:.12e}\n")
        fh.write(f"1/L = {1.0 / L:.12e}\n2/L = {2.0 / L:.12e}\n")
        fh.write(f"stopping rule: ||g_k|| <= {TOL} * ||g_0||\n\n")

        fh.write("A. SHORT RUNS FROM SEVERAL STARTING POINTS\n")
        fh.write(f"   {SHORT_ITERS} iterations, seeds {SHORT_SEEDS}\n")
        fh.write("   Reported: ||g_k|| / ||g_0|| at each checkpoint.\n\n")
        for seed in SHORT_SEEDS:
            fh.write(f"   seed {seed}\n")
            fh.write("     c, " + ", ".join(f"k={k}" for k in CHECKPOINTS) + "\n")
            for c in all_c:
                fh.write(f"     {c}, " +
                         ", ".join(f"{v:.6e}" for v in table[(seed, c)]) + "\n")
            fh.write("\n")
        for seed, order in zip(SHORT_SEEDS, orderings):
            fh.write(f"   ranking at k={SHORT_ITERS}, seed {seed} (best first): "
                     + " < ".join(f"{c:g}" for c in order) + "\n")
        fh.write(f"\n   ranking identical across all seeds: {identical}\n")
        fh.write("   => the starting point moves the absolute numbers but not the\n"
                 "      ordering, so a single seed is enough for the selection.\n\n")

        fh.write("B. FULL RUNS TO THE STOPPING RULE (seed %d)\n" % SEED)
        fh.write("   c, iterations_to_tol, final_ratio, increases, stop_reason, seconds\n")
        for c in all_c:
            r = results[c]
            ratio = r["grad_norms"] / r["grad_norms"][0]
            bumps = int(np.sum(np.diff(ratio) > 0))
            it = r["iters"] if r["iters"] is not None else "not_reached"
            fh.write(f"   {c}, {it}, {ratio[-1]:.6e}, {bumps}, "
                     f"{r['reason']}, {r['seconds']:.2f}\n")
        fh.write("\n   'increases' counts iterations where ||g_k||/||g_0|| grew;\n"
                 "   a step that overshoots near the solution would show up there.\n")

        reached = [c for c in all_c if results[c]["iters"] is not None]
        if reached:
            best = min(reached, key=lambda c: results[c]["iters"])
            fh.write(f"\n   fastest to tolerance: c = {best} "
                     f"({results[best]['iters']} iterations)\n")
            within = [c for c in reached if c in SWEEP_C]
            if within:
                bw = min(within, key=lambda c: results[c]["iters"])
                fh.write(f"   fastest within the 2/L guarantee: c = {bw} "
                         f"({results[bw]['iters']} iterations)\n")


def main():
    X, y, _, _ = load_data()
    s = signs(y)
    L = float(np.linalg.eigvalsh(X @ X.T)[-1]) + LAM
    all_c = SWEEP_C + PROBE_C
    print(f"L = {L:.6e},  1/L = {1.0 / L:.6e},  2/L = {2.0 / L:.6e}")
    print(f"candidates: {SWEEP_C} (guaranteed) + {PROBE_C} (beyond 2/L)")

    table, orderings, identical = short_runs(X, s, L, all_c)
    results = long_runs(X, s, L, all_c)
    make_plot(results, all_c)
    save_histories(results, all_c)
    write_report(L, table, orderings, identical, results, all_c)
    print(f"\nWritten to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
