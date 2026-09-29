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
matplotlib.use("Agg")            
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# figures for Q5 (imported after the Agg backend is selected above)
from plot_utils import save_q4_results, plot_convergence, plot_theory_horizon

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAM = 0.005            # regularization parameter 
SEED = 42              # seed for theta_0 in Q4
TOL = 1e-3             # relative gradient tolerance in Q4
MAX_TIME = 3 * 60      # seconds for Q4
THETA_SCALE = 0.1          # scales for theta in Q3
Q2_SCALES = [1e-2, 1e-1, 1.0]   # 
FLOOR_FACTOR = 100.0       # 
T_MAX_FIT = 1e-1           # 


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

    # Intercept is already in the data: the last row should be the ones.
    # Reported, not asserted, so an unexpected file does not stop the run.
    # allclose rather than ==, since a float that should be 1 need not be exactly 1.
    if not (np.allclose(X_train[-1, :], 1.0) and np.allclose(X_test[-1, :], 1.0)):
        print("WARNING: the last row of X is not all ones -- check the bias row")
    if not (X_train.shape[1] == y_train.size and X_test.shape[1] == y_test.size):
        print("WARNING: columns of X do not match the number of labels")
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
    z = s * (X.T @ theta)
    f = np.sum(_phi(z)) + 0.5 * lam * (theta @ theta)
    g = X @ (s * _dphi(z)) + lam * theta
    return f, g


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

        fl, fv = f_loop(theta, X, s, LAM), f_vec(theta, X, s, LAM)
        gl, gv = grad_loop(theta, X, s, LAM), grad_vec(theta, X, s, LAM)
        rel_f = abs(fl - fv) / max(1.0, abs(fl))
        rel_g = np.linalg.norm(gl - gv) / max(1.0, np.linalg.norm(gl))

        per_scale.append(dict(
            scale=float(scale),
            rel_objective_error=float(rel_f),
            rel_gradient_error=float(rel_g),
        ))
        print(f"  scale = {scale:>5g}: "
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
        fh.write(f"f_lambda(theta): {f0:.12e}\n")
        fh.write(f"round-off floor (={FLOOR_FACTOR}*eps*f0): {floor:.6e}\n")
        fh.write(f"fit window: t in [{t[mask][0]:.6e}, {t[mask][-1]:.6e}]\n")
        fh.write(f"log-log slope: {slope:.6f}\n")


# ==================================================================
# Q4: fixed-step gradient descent
# ==================================================================
# Q4: gradient descent with a fixed constant step size.
# run_gd() is the shared GD loop, used both here and by step_size_experiment.py.
# The step size (CHOSEN_C / L) was selected beforehand in step_size_experiment.py

CHOSEN_C = 1.9           # fastest within alpha < 2/L --> see q4_step_size_experiment.txt


def run_gd(X, s, L, c, theta0, tol=None, max_iter=None, max_time=None):
    """ 
    Gradient descent run with the constant step size alpha = c / L.
 
    Stops on whichever applies: relative gradient tolerance, iteration cap,
    time cap, or a non-finite value (divergence).
    Returns: (theta, objectives, grad_norms, reason, elapsed).
    """
    # parameters
    step = c / L
    theta = theta0.copy()

    # initialisation
    f, g = f_and_grad(theta, X, s, LAM)
    g0_norm = float(np.linalg.norm(g))
    objectives = [f]
    grad_norms = [g0_norm]

    start = time.perf_counter()
    reason = None
    k = 0
    # iterations
    while True:
        # check stopping reason
        if tol is not None and grad_norms[-1] <= tol * g0_norm:
            reason = "gradient_tolerance"
            break
        if max_iter is not None and k >= max_iter:
            reason = "iteration_cap"
            break
        if max_time is not None and time.perf_counter() - start >= max_time:
            reason = "time_limit"
            break

        # compute results
        theta = theta - step * g
        k += 1
        f, g = f_and_grad(theta, X, s, LAM)
        gn = float(np.linalg.norm(g))

        # safeguard for inf and NaN
        if not (np.isfinite(f) and np.isfinite(gn)):
            objectives.append(f)
            grad_norms.append(gn)
            reason = "diverged"
            break
        # save computations
        objectives.append(f)
        grad_norms.append(gn)

    elapsed = time.perf_counter() - start                         # total time
    return theta, np.array(objectives), np.array(grad_norms), reason, elapsed


def question4(X, s, L):
    """
    Q4: gradient descent with a fixed constant step alpha = CHOSEN_C / L.
 
    Stops on whichever applies: relative gradient tolerance, iteration cap,
    time cap, or a non-finite value (divergence).
    Returns: (theta, objectives, grad_norms, reason, elapsed).
    """
    print("\n=== Q4: fixed-step gradient descent ===")
    # parameters
    theta0 = np.random.default_rng(SEED).standard_normal(X.shape[0])
    sigma_max = float(np.sqrt(L - LAM))
    step = CHOSEN_C / L

    # run gradient descent
    theta, objectives, grad_norms, reason, elapsed = run_gd(
        X, s, L, CHOSEN_C, theta0, tol=TOL, max_time=MAX_TIME)

    k = objectives.size - 1   # vectorisation at 0


    # saves in results
    save_q4_results(RESULTS_DIR, k, objectives, grad_norms, theta0,
                    reason, elapsed, step=step, c=CHOSEN_C,
                    sigma_max=sigma_max, L=L, lam=LAM, seed=SEED,
                    tol=TOL, max_time=MAX_TIME)

    return theta, objectives, grad_norms


# ==================================================================
# Q5: convergence plots
# ==================================================================
# We plot the objective function and its gradient against iterations for both
# our experimental and the theoretical results generalised from the notes -
# Corollary 4.32 (see question 6)
# The method is directly run on question 4 and we use results from theta_final,
# objectives, grad_norms


def question5(objectives, grad_norms, L):
    """
    Q5: plot the run from Question 4 with the theoretical guarantee for the gradient


    Nothing is re-run here: `objectives` and `grad_norms` are the histories
    recorded during the Q4 descent.  The figures themselves live in
    plot_utils.py.
    """
    print("\n=== Q5: convergence plots ===")
    # set parameters
    k = np.arange(objectives.size)
    c = CHOSEN_C
    rho = 1.0 - c * (2.0 - c) / (L / LAM)   # mu = lambda, from Q1
    prefactor = np.sqrt(2.0 * L * objectives[0])
    bound = prefactor * rho ** (k / 2.0)

    # plot results - functions in utils
    plot_convergence(objectives, grad_norms, bound, c, RESULTS_DIR)
    plot_theory_horizon(grad_norms, prefactor, rho, c, TOL, RESULTS_DIR)


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
    question5(objectives, grad_norms, L)
    question7(theta_final, X_tr, y_tr, X_te, y_te)
    print(f"\nAll results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
