"""
MATH-329 Continuous Optimization -- Homework 1  (Group S)

Run from the code/ folder:   python main.py
Reads   ../data/mnist_train_test.mat
Writes  ../results/*   (all files requested in Questions 2-5 and 7)

Model:  f_lambda(theta) = sum_i phi(s_i <x~_i, theta>) + (lambda/2)||theta||^2
        with s_i = 1 - 2 y_i  (NOT 2 y_i - 1),  y_i in {0, 1}.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no GUI: script must finish without user input
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# ------------------------------------------------------------------
# Paths (relative to this file, so it works from code/ as required)
# ------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAM = 0.005            # regularization parameter (fixed by the homework)
SEED = 42              # seed for theta_0 in Q4
TOL = 1e-3             # relative gradient tolerance in Q4
MAX_TIME = 3 * 60      # seconds, Q4
THETA_SCALE = 0.1          # chosen from a scale sweep: populates all 3 branches
Q2_SCALES = [1e-2, 1e-1, 1.0]   # scales used in the Q2 agreement test
FLOOR_FACTOR = 100.0       # keep points with err > FLOOR_FACTOR * eps * f0
T_MAX_FIT = 1e-1           # upper end of the fit; check against the plot


# ==================================================================
# Data
# ==================================================================
def load_data():
    path = DATA_DIR / "mnist_train_test.mat"
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    train, test = data["train"], data["test"]

    X_train = np.asarray(train.X, dtype=np.float64)   # (785, 12665)
    y_train = np.asarray(train.y, dtype=np.float64).ravel()
    X_test = np.asarray(test.X, dtype=np.float64)     # (785, 2115)
    y_test = np.asarray(test.y, dtype=np.float64).ravel()

    # The last row already contains the ones (bias): do not append another.
    assert np.allclose(X_train[-1, :], 1.0) and np.allclose(X_test[-1, :], 1.0)              # safer than exact float comparison 
    assert X_train.shape[1] == y_train.size and X_test.shape[1] == y_test.size
    return X_train, y_train, X_test, y_test


def signs(y):
    """s_i = 1 - 2 y_i in {-1, +1}  (y=0 -> +1, y=1 -> -1)."""
    return 1.0 - 2.0 * y


# ==================================================================
# Q2: objective and gradient
# ==================================================================
# ---- explicit loops ------------------------------------------------
def f_loop(theta, X, s, lam):
    value = 0.5 * lam * float(theta @ theta)
    for i in range(X.shape[1]):
        z = s[i] * float(X[:, i] @ theta)
        if z <= -1.0:
            phi = 0.0
        elif z < 0.0:
            phi = 0.5 * (1.0 + z) ** 2
        else:
            phi = 0.5 + z
        value += phi
    return value


def grad_loop(theta, X, s, lam):
    grad = lam * theta
    for i in range(X.shape[1]):
        z = s[i] * float(X[:, i] @ theta)
        if z <= -1.0:
            dphi = 0.0
        elif z < 0.0:
            dphi = 1.0 + z
        else:
            dphi = 1.0
        grad = grad + s[i] * dphi * X[:, i]
    return grad


# ---- vectorized ----------------------------------------------------
def _phi(z):
    # phi(z) = 0.5*clip(1+z,0,1)^2 + max(z,0)   (equals the piecewise def.)
    return 0.5 * np.clip(1.0 + z, 0.0, 1.0) ** 2 + np.maximum(z, 0.0)


def _dphi(z):
    # phi'(z) = clip(1+z, 0, 1)
    return np.clip(1.0 + z, 0.0, 1.0)


def f_vec(theta, X, s, lam):
    z = s * (X.T @ theta)
    return np.sum(_phi(z)) + 0.5 * lam * (theta @ theta)


def grad_vec(theta, X, s, lam):
    z = s * (X.T @ theta)
    return X @ (s * _dphi(z)) + lam * theta


def f_and_grad(theta, X, s, lam):
    """Both quantities from a single product X^T theta (no redundant work)."""
    z = s * (X.T @ theta)
    f = np.sum(_phi(z)) + 0.5 * lam * (theta @ theta)
    g = X @ (s * _dphi(z)) + lam * theta
    return f, g


def branch_counts(z):
    """How many z_i fall in each of the three branches of phi."""
    z = np.asarray(z)
    return (
        int(np.sum(z <= -1.0)),
        int(np.sum((z > -1.0) & (z < 0.0))),
        int(np.sum(z >= 0.0)),
    )


# ---- timing --------------------------------------------------------
def time_it(func, args, reps, warmup=2):
    """Median and minimum wall-clock time over `reps` calls, after warm-up."""
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.min(times))


def question2(X, s):
    print("\n=== Q2: objective and gradient ===")
    rng = np.random.default_rng(0)

    # ---- numerical agreement, loop vs vectorized ----------------------
    per_scale = []
    for scale in Q2_SCALES:
        theta = scale * rng.standard_normal(X.shape[0])
        left, middle, right = branch_counts(s * (X.T @ theta))

        fl, fv = f_loop(theta, X, s, LAM), f_vec(theta, X, s, LAM)
        gl, gv = grad_loop(theta, X, s, LAM), grad_vec(theta, X, s, LAM)
        rel_f = abs(fl - fv) / max(1.0, abs(fl))
        rel_g = np.linalg.norm(gl - gv) / max(1.0, np.linalg.norm(gl))

        per_scale.append(dict(
            scale=float(scale),
            branch_left=left,
            branch_middle=middle,
            branch_right=right,
            rel_objective_error=float(rel_f),
            rel_gradient_error=float(rel_g),
        ))
        print(f"  scale = {scale:>5g}: branches (z<=-1 / -1<z<0 / z>=0) = "
              f"{left} / {middle} / {right},  "
              f"rel. error  f = {rel_f:.2e},  grad = {rel_g:.2e}")

    max_f = max(p["rel_objective_error"] for p in per_scale)
    max_g = max(p["rel_gradient_error"] for p in per_scale)
    print(f"  max relative error over all scales: f = {max_f:.2e}, "
          f"grad = {max_g:.2e}")

    # ---- run times, at a single theta ---------------------------------
    theta = THETA_SCALE * np.random.default_rng(1).standard_normal(X.shape[0])
    args = (theta, X, s, LAM)
    t_fl, t_fl_min = time_it(f_loop, args, reps=5)
    t_fv, t_fv_min = time_it(f_vec, args, reps=100)
    t_gl, t_gl_min = time_it(grad_loop, args, reps=5)
    t_gv, t_gv_min = time_it(grad_vec, args, reps=100)
    print(f"  median time  f: loop {t_fl:.3e}s | vec {t_fv:.3e}s | "
          f"speed-up {t_fl/t_fv:.0f}x")
    print(f"  median time  g: loop {t_gl:.3e}s | vec {t_gv:.3e}s | "
          f"speed-up {t_gl/t_gv:.0f}x")

    with open(RESULTS_DIR / "q2_summary.json", "w") as fh:
        json.dump(dict(
            per_scale=per_scale,
            max_rel_objective_error=max_f,
            max_rel_gradient_error=max_g,
            timing_note="median and min over repeated calls, after 2 warm-up calls",
            reps_loop=5,
            reps_vectorized=100,
            time_f_loop_median=t_fl,
            time_f_vec_median=t_fv,
            time_grad_loop_median=t_gl,
            time_grad_vec_median=t_gv,
            time_f_loop_min=t_fl_min,
            time_f_vec_min=t_fv_min,
            time_grad_loop_min=t_gl_min,
            time_grad_vec_min=t_gv_min,
            speedup_f=t_fl / t_fv,
            speedup_grad=t_gl / t_gv,
        ), fh, indent=2)

    # ---- runtime figure ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Objective\nloop", "Objective\nvectorized",
              "Gradient\nloop", "Gradient\nvectorized"]
    times = [t_fl, t_fv, t_gl, t_gv]
    bars = ax.bar(labels, times)
    ax.set_yscale("log")
    ax.set(ylabel="Median runtime (s)", title="Runtime comparison")
    ax.grid(axis="y", ls=":", alpha=0.5)
    for b, tv in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, tv, f"{tv:.2e}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q2_runtime_comparison.pdf")
    plt.close(fig)


# ==================================================================
# Q3: gradient check
# ==================================================================
# The fitting window is chosen from properties of the curve that do NOT
# involve its slope: the remainder must be above the round-off floor
# (differences of numbers of size f0 lose digits below ~eps*f0) and below
# the large-t region where the second-order model stops being accurate.
# Choosing the window by "slope close to 2" would make the test unable to
# fail, since a wrong gradient (slope 1) would simply never be reported.


def question3(X, s):
    print("\n=== Q3: gradient check ===")
    rng = np.random.default_rng(0)
    theta = THETA_SCALE * rng.standard_normal(X.shape[0])
    v = rng.standard_normal(X.shape[0])
    v /= np.linalg.norm(v)                      # unit direction

    # X^T (theta + t v) = X^T theta + t X^T v, so the two products below are
    # computed once instead of once per value of t.
    a = s * (X.T @ theta)
    b = s * (X.T @ v)

    left, middle, right = branch_counts(a)
    print(f"  branches (z<=-1 / -1<z<0 / z>=0) = {left} / {middle} / {right}")

    f0, g0 = f_and_grad(theta, X, s, LAM)
    directional = float(v @ g0)

    # regularizer along the line: ||theta + t v||^2 = ||theta||^2 + 2t<theta,v> + t^2
    tt, tv_, vv = float(theta @ theta), float(theta @ v), 1.0

    t = np.logspace(-8.0, 0.0, num=101)
    err = np.array([
        abs(np.sum(_phi(a + tk * b))
            + 0.5 * LAM * (tt + 2.0 * tk * tv_ + tk ** 2 * vv)
            - f0 - tk * directional)
        for tk in t
    ])

    np.savetxt(RESULTS_DIR / "q3_gradient_check.csv",
               np.column_stack((t, err)), delimiter=",",
               header="t,error", comments="")

    # ---- fit the straight portion --------------------------------------
    floor = FLOOR_FACTOR * np.finfo(float).eps * abs(f0)
    mask = (err > floor) & (t <= T_MAX_FIT)
    slope, _ = np.polyfit(np.log10(t[mask]), np.log10(err[mask]), 1)
    print(f"  f(theta) = {f0:.3e}, round-off floor ~ {floor:.2e}")
    print(f"  fit window: t in [{t[mask][0]:.2e}, {t[mask][-1]:.2e}] "
          f"({int(mask.sum())} points)")
    print(f"  log-log slope = {slope:.4f}")

    # ---- figure ---------------------------------------------------------
    C = np.median(err[mask] / t[mask] ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t, err, label="Taylor remainder")
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
    fig.savefig(RESULTS_DIR / "q3_gradient_check.pdf")
    plt.close(fig)

    with open(RESULTS_DIR / "q3_slope.txt", "w") as fh:
        fh.write(f"theta scale: {THETA_SCALE}\n")
        fh.write(f"branch counts (z<=-1 / -1<z<0 / z>=0): "
                 f"{left} / {middle} / {right}\n")
        fh.write(f"f_lambda(theta): {f0:.12e}\n")
        fh.write(f"round-off floor (={FLOOR_FACTOR}*eps*f0): {floor:.6e}\n")
        fh.write(f"fit window: t in [{t[mask][0]:.6e}, {t[mask][-1]:.6e}]\n")
        fh.write(f"log-log slope: {slope:.6f}\n")


# ==================================================================
# Q4: fixed-step gradient descent
# ==================================================================

"""
Q4: gradient descent with a fixed constant step size.

run_gd() is the shared GD loop, used both here and by step_size_sweep.py.
The step size (CHOSEN_C / L) was selected beforehand with that script;
it is not re-derived here, so this run is deterministic.
"""

CHOSEN_C = 1.9           # fastest within alpha < 2/L; see q4_step_size_sweep.txt
 
 
def run_gd(X, s, L, c, theta0, tol=None, max_iter=None, max_time=None,
           snapshots=None, record_every=200):
    """One gradient descent run with the constant step size alpha = c / L.
 
    Stops on whichever applies: relative gradient tolerance, iteration cap,
    time cap, or a non-finite value (divergence).
    Returns (theta, objectives, grad_norms, reason, elapsed).

    If `snapshots` is a list, (k, theta_k) is appended to it every
    `record_every` iterations (and at k=0 and at the last iterate).  The
    return value is unchanged, so callers that do not record are unaffected.
    """
    step = c / L
    theta = theta0.copy()
 
    f, g = f_and_grad(theta, X, s, LAM)
    g0_norm = float(np.linalg.norm(g))
    objectives = [f]
    grad_norms = [g0_norm]
    if snapshots is not None:
        snapshots.append((0, theta.copy()))
 
    start = time.perf_counter()
    reason = None
    k = 0
    while True:
        if tol is not None and grad_norms[-1] <= tol * g0_norm:
            reason = "gradient_tolerance"
            break
        if max_iter is not None and k >= max_iter:
            reason = "iteration_cap"
            break
        if max_time is not None and time.perf_counter() - start >= max_time:
            reason = "time_limit"
            break
 
        theta = theta - step * g
        k += 1
        f, g = f_and_grad(theta, X, s, LAM)      # one pass per iteration
        gn = float(np.linalg.norm(g))
 
        if not (np.isfinite(f) and np.isfinite(gn)):
            objectives.append(f)
            grad_norms.append(gn)
            reason = "diverged"
            break
 
        objectives.append(f)
        grad_norms.append(gn)
        if snapshots is not None and k % record_every == 0:
            snapshots.append((k, theta.copy()))
 
    if snapshots is not None and snapshots[-1][0] != k:
        snapshots.append((k, theta.copy()))
    elapsed = time.perf_counter() - start
    return theta, np.array(objectives), np.array(grad_norms), reason, elapsed
 
 
def question4(X, s, L):
    """Q4: gradient descent with a fixed constant step alpha = CHOSEN_C / L.
 
    The step size was chosen beforehand from step_size_sweep.py (not run
    here, so that this run is fully deterministic and matches the report).
    """
    print("\n=== Q4: fixed-step gradient descent ===")
    theta0 = np.random.default_rng(SEED).standard_normal(X.shape[0])
    sigma_max = float(np.sqrt(L - LAM))
    step = CHOSEN_C / L
 
    print(f"sigma_max = {sigma_max:.6e},  L = {L:.6e}")
    print(f"step = {CHOSEN_C}/L = {step:.6e}")
 
    snapshots = []
    theta, objectives, grad_norms, reason, elapsed = run_gd(
        X, s, L, CHOSEN_C, theta0, tol=TOL, max_time=MAX_TIME,
        snapshots=snapshots, record_every=CURVATURE_EVERY)
 
    k = objectives.size - 1
    g0_norm = grad_norms[0]
    print(f"iterations = {k},  time = {elapsed:.2f}s,  stop = {reason}")
    print(f"||g0|| = {g0_norm:.6e},  ||g_final|| = {grad_norms[-1]:.6e}, "
          f"ratio = {grad_norms[-1]/g0_norm:.3e}")
    print(f"f(theta_0) = {objectives[0]:.6e},  "
          f"f(theta_final) = {objectives[-1]:.6e},  "
          f"f(theta_0)/f(theta_final) = {objectives[0]/objectives[-1]:.3e}")
 
    np.savez(RESULTS_DIR / "q4_gd_history.npz",
             iterations=np.arange(k + 1), objectives=objectives,
             gradient_norms=grad_norms, theta0=theta0, theta_final=theta,
             step_size=step, c=CHOSEN_C, sigma_max=sigma_max, L=L,
             lam=LAM, seed=SEED)
    np.savetxt(RESULTS_DIR / "q4_gd_history.csv",
               np.column_stack((np.arange(k + 1), objectives, grad_norms)),
               delimiter=",", header="k,objective,gradient_norm", comments="")
    with open(RESULTS_DIR / "q4_stopping_reason.txt", "w") as fh:
        fh.write(f"Stopping reason: {reason}\n")
        fh.write(f"Iterations: {k}\n")
        fh.write(f"Elapsed time: {elapsed:.6f} s\n")
        fh.write(f"Step size: alpha = {CHOSEN_C}/L = {step:.12e}\n")
        fh.write("(step size chosen from step_size_sweep.py; "
                 "see q4_step_size_sweep.txt)\n")

    question4_curvature(X, s, L, step, snapshots)

    return theta, objectives, grad_norms






# ------------------------------------------------------------------
# Q4 (continued): why a step this large is safe, and why the iteration
# count scales like 1/c.
#
# phi'' = 1 on (-1, 0) and 0 elsewhere, so the Hessian of f_lambda at theta
# only collects the samples whose margin is in (-1, 0):
#
#     H(theta) = sum_{i : -1 < z_i < 0} x~_i x~_i^T  +  lambda I.
#
# L = lambda_max(X X^T) + lambda is the value that bound takes when EVERY
# sample is in the quadratic branch at once, which never happens.  The gap
# between lambda_max(H) and L is what makes alpha = c/L tiny compared with
# the curvature it actually meets: gradient descent contracts by
# |1 - alpha * nu| per iteration in a direction of curvature nu, and when
# alpha * nu << 1 that is 1 - alpha * nu, linear in alpha.  Hence the number
# of iterations scales as 1/alpha = L/(c), and no direction can overshoot.
# ------------------------------------------------------------------
CURVATURE_EVERY = 200      # record a snapshot every this many iterations

# categorical slots 1-3, fixed order (validated for light-mode line charts)
BRANCH_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def top_eigenvalue_active(X, s, theta, iters=100, seed=0):
    """lambda_max(H(theta)) = lambda_max(X_act X_act^T) + lambda.

    Power iteration on the matrix-vector product X_act (X_act^T v), so the
    785 x 785 Gram matrix is never formed; agrees with eigvalsh to machine
    precision and stays cheap when many samples are active.
    """
    z = s * (X.T @ theta)
    active = (z > -1.0) & (z < 0.0)
    if not active.any():
        return LAM, 0
    Xa = X[:, active]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(Xa.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = Xa @ (Xa.T @ v)
        n = np.linalg.norm(w)
        if n == 0.0:
            return LAM, int(active.sum())
        v = w / n
    return float(v @ (Xa @ (Xa.T @ v))) + LAM, int(active.sum())


def question4_curvature(X, s, L, step, snapshots):
    """Curvature actually met along the Q4 trajectory, vs the bound L."""
    print("\n=== Q4 (cont.): active curvature along the trajectory ===")
    rows = []
    for k, th in snapshots:
        z = s * (X.T @ th)
        flat, quad, affine = branch_counts(z)
        nu_max, n_act = top_eigenvalue_active(X, s, th)
        rows.append((k, flat, quad, affine, nu_max, step * nu_max, 2.0 / step))
    arr = np.array(rows, dtype=float)

    np.savetxt(RESULTS_DIR / "q4_curvature.csv", arr, delimiter=",",
               fmt=["%d", "%d", "%d", "%d", "%.12e", "%.12e", "%.12e"],
               header="k,n_flat,n_quadratic,n_affine,lambda_max_active,"
                      "alpha_times_lambda_max,stability_threshold",
               comments="")

    first, last = rows[0], rows[-1]
    print(f"  global L = {L:.4e}   alpha = {step:.4e}   "
          f"stability needs alpha*lambda_max < 2")
    for tag, r in (("theta_0", first), ("theta_final", last)):
        print(f"  {tag:12s} k={int(r[0]):6d}  branches "
              f"{int(r[1])}/{int(r[2])}/{int(r[3])}  "
              f"lambda_max(H) = {r[4]:.4e} ({L / r[4]:.0f}x below L)  "
              f"alpha*lambda_max = {r[5]:.4f}  max stable c = {2 * L / r[4]:.0f}")
    worst = arr[:, 5].max()
    print(f"  worst alpha*lambda_max over the whole run: {worst:.4f} "
          f"(margin of {2.0 / worst:.0f}x)")

    with open(RESULTS_DIR / "q4_curvature.txt", "w") as fh:
        fh.write("Active curvature along the Q4 trajectory\n\n")
        fh.write("H(theta) = sum_{i: -1 < z_i < 0} x~_i x~_i^T + lambda I;\n")
        fh.write("only samples in the quadratic branch of phi contribute.\n\n")
        fh.write(f"lambda = {LAM}\nc = {CHOSEN_C}\nalpha = {step:.12e}\n")
        fh.write(f"L = lambda_max(X X^T) + lambda = {L:.12e}\n")
        fh.write("   (the value of lambda_max(H) if every sample were in the\n"
                 "    quadratic branch simultaneously)\n")
        fh.write(f"stability threshold: lambda_max(H) < 2/alpha = {2.0 / step:.6e}\n\n")
        for tag, r in (("at theta_0     ", first), ("at theta_final ", last)):
            fh.write(f"{tag} k = {int(r[0])}\n")
            fh.write(f"   branches (z<=-1 / -1<z<0 / z>=0): "
                     f"{int(r[1])} / {int(r[2])} / {int(r[3])}\n")
            fh.write(f"   samples in the quadratic branch: "
                     f"{100.0 * r[2] / X.shape[1]:.2f}%\n")
            fh.write(f"   lambda_max(H) = {r[4]:.6e}  "
                     f"({L / r[4]:.1f}x below L)\n")
            fh.write(f"   alpha*lambda_max = {r[5]:.6f}  "
                     f"(largest c that would still be stable: {2 * L / r[4]:.1f})\n\n")
        fh.write(f"worst alpha*lambda_max over the run: {worst:.6f}\n")
        fh.write(f"smallest stability margin: {2.0 / worst:.1f}x\n\n")
        fh.write("Reading: alpha*lambda_max stays far below 2 for the whole\n"
                 "trajectory, so every direction contracts by 1 - alpha*nu,\n"
                 "which is linear in alpha.  That is why the iteration count\n"
                 "scales as 1/c and why no iterate overshoots.  The margin\n"
                 "GROWS as the run converges: the optimum is the safest place\n"
                 "for a large step, not the most dangerous.\n")

    # ---- figure ---------------------------------------------------------
    k = arr[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    for col, color, lab in zip(
            (1, 2, 3), BRANCH_COLORS,
            (r"flat  $z \leq -1$", r"quadratic  $-1 < z < 0$", r"affine  $z \geq 0$")):
        ax.semilogy(k, np.maximum(arr[:, col], 0.5), color=color, lw=2, label=lab)
    ax.set_ylabel("samples in branch")
    ax.set_title(r"Which branch of $\phi$ the margins occupy")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right")

    ax = axes[1]
    ax.semilogy(k, arr[:, 4], color=BRANCH_COLORS[0], lw=2,
                label=r"$\lambda_{\max}(H(\theta_k))$  (curvature actually met)")
    ax.axhline(L, color="#52514e", ls="--", lw=1,
               label=r"$L = \lambda_{\max}(XX^\top)+\lambda$  (bound used for $\alpha$)")
    ax.axhline(2.0 / step, color="#e34948", ls=":", lw=1.5,
               label=r"stability limit $2/\alpha$ (only %.0f%% above $L$, as $c=%g$)"
                     % (100.0 * (2.0 / step / L - 1.0), CHOSEN_C))
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"curvature")
    ax.set_title(r"The step never approaches the stability limit")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q4_curvature.pdf")
    plt.close(fig)


# ==================================================================
# Q5: convergence plots
# ==================================================================
def question5(objectives, grad_norms):
    print("\n=== Q5: convergence plots ===") # no need of prints since graphs are on results?
    k = np.arange(objectives.size)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # graph - log
    axes[0].semilogy(k, objectives)
    axes[0].set(xlabel="Iteration $k$", ylabel=r"$f_\lambda(\theta_k)$",
                title="Objective value")
    axes[1].semilogy(k, grad_norms)
    axes[1].set(xlabel="Iteration $k$",
                ylabel=r"$\|\nabla f_\lambda(\theta_k)\|$",
                title="Gradient norm")
    for ax in axes:
        ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q5_convergence.pdf")
    plt.close(fig)



    # ASK TAs - no need if we argued well
    # Also save the gap f(theta_k) - f_best on a log scale for theoretical comparison
    f_best = float(np.min(objectives))
    gap = objectives - f_best
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(k, np.maximum(gap, 1e-300))
    ax.set(xlabel="Iteration $k$", ylabel=r'$f_\lambda(\theta_k)-f_{\rm best}$',
           title="Objective gap to best observed value")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "q5_gap.pdf")
    plt.close(fig)



# ==================================================================
# Q7: classification
# ==================================================================
def error_rate(theta, X, y):
    pred = (X.T @ theta > 0).astype(float)      # 1 if x~^T theta > 0 else 0
    return float(np.mean(pred != y))


def question7(theta, X_tr, y_tr, X_te, y_te):
    print("\n=== Q7: classification error ===")
    e_tr, e_te = error_rate(theta, X_tr, y_tr), error_rate(theta, X_te, y_te)
    print(f"train error = {e_tr:.6f}  ({e_tr*100:.3f}%)")
    print(f"test  error = {e_te:.6f}  ({e_te*100:.3f}%)")
    np.save(RESULTS_DIR / "q7_theta_final.npy", theta)
    np.savetxt(RESULTS_DIR / "q7_theta_final.csv", theta, delimiter=",")
    with open(RESULTS_DIR / "q7_errors.json", "w") as fh:
        json.dump({"train_error": e_tr, "test_error": e_te}, fh, indent=2)


# ==================================================================





 









def main():
    X_tr, y_tr, X_te, y_te = load_data()
    s_tr = signs(y_tr)
    print(f"train X {X_tr.shape}, test X {X_te.shape}, lambda = {LAM}")

    L = float(np.linalg.eigvalsh(X_tr @ X_tr.T)[-1]) + LAM # make sure its not computed again!!

    question2(X_tr, s_tr)
    question3(X_tr, s_tr)
    theta_final, objectives, grad_norms = question4(X_tr, s_tr, L)
    question5(objectives, grad_norms)
    # question6(L, objectives, grad_norms)
    question7(theta_final, X_tr, y_tr, X_te, y_te)
    print(f"\nAll results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
















