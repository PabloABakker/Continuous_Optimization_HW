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


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
RESULTS_DIR = CODE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAM = 0.005           
SEED = 42              # seed for theta_0 in Q4
TOL = 1e-3             # relative gradient tolerance in Q4
MAX_TIME = 3 * 60      # seconds, Q4


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
    assert np.all(X_train[-1, :] == 1.0) and np.all(X_test[-1, :] == 1.0)
    assert X_train.shape[1] == y_train.size and X_test.shape[1] == y_test.size
    return X_train, y_train, X_test, y_test


def signs(y):
    """s_i = 1 - 2 y_i in {-1, +1}  (y=0 -> +1, y=1 -> -1)."""
    return 1.0 - 2.0 * y


# ==================================================================
# Q2: objective and gradient
# ==================================================================

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


# ---- timing --------------------------------------------------------
def time_it(func, args, reps, warmup=2):
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def question2(X, s):
    print("\n=== Q2: objective and gradient ===")
    rng = np.random.default_rng(0)
    n_tests = 10
    f_err, g_err = [], []
    for _ in range(n_tests):
        theta = rng.standard_normal(X.shape[0])
        fl, fv = f_loop(theta, X, s, LAM), f_vec(theta, X, s, LAM)
        gl, gv = grad_loop(theta, X, s, LAM), grad_vec(theta, X, s, LAM)
        f_err.append(abs(fl - fv) / max(1.0, abs(fl)))
        g_err.append(np.linalg.norm(gl - gv) / max(1.0, np.linalg.norm(gl)))
    print(f"max relative objective error over {n_tests} random thetas: {max(f_err):.3e}")
    print(f"max relative gradient  error over {n_tests} random thetas: {max(g_err):.3e}")

    theta = np.random.default_rng(1).standard_normal(X.shape[0])
    args = (theta, X, s, LAM)
    t_fl = time_it(f_loop, args, reps=5)
    t_fv = time_it(f_vec, args, reps=100)
    t_gl = time_it(grad_loop, args, reps=5)
    t_gv = time_it(grad_vec, args, reps=100)
    print(f"avg time  f: loop {t_fl:.3e}s | vec {t_fv:.3e}s | speedup {t_fl/t_fv:.1f}x")
    print(f"avg time  g: loop {t_gl:.3e}s | vec {t_gv:.3e}s | speedup {t_gl/t_gv:.1f}x")

    summary = dict(
        max_rel_objective_error=max(f_err), max_rel_gradient_error=max(g_err),
        time_f_loop=t_fl, time_f_vec=t_fv, time_grad_loop=t_gl, time_grad_vec=t_gv,
        speedup_f=t_fl / t_fv, speedup_grad=t_gl / t_gv,
    )
    with open(RESULTS_DIR / "q2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # figures used in the report (optional)
    fig, ax = plt.subplots(figsize=(7, 4))
    trials = np.arange(1, n_tests + 1)
    ax.semilogy(trials, np.maximum(f_err, 1e-18), "o-", label="Objective")
    ax.semilogy(trials, np.maximum(g_err, 1e-18), "s-", label="Gradient")
    ax.set(xlabel="Random test", ylabel="Relative error",
           title="Loop vs. vectorized", xticks=trials)
    ax.grid(True, which="both", ls=":"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "q2_numerical_agreement.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Objective\nloop", "Objective\nvectorized",
              "Gradient\nloop", "Gradient\nvectorized"]
    times = [t_fl, t_fv, t_gl, t_gv]
    bars = ax.bar(labels, times)
    ax.set_yscale("log")
    ax.set(ylabel="Average runtime (s)", title="Runtime comparison")
    ax.grid(axis="y", ls=":", alpha=0.5)
    for b, tv in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, tv, f"{tv:.2e}",
                ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "q2_runtime_comparison.pdf")
    plt.close(fig)


# ==================================================================
# Q3: gradient check
# ==================================================================
def question3(X, s):
    print("\n=== Q3: gradient check ===")
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(X.shape[0])
    v = rng.standard_normal(X.shape[0])
    v /= np.linalg.norm(v)                       # unit direction

    f0, g0 = f_and_grad(theta, X, s, LAM)
    slope_dir = float(v @ g0)

    t = np.logspace(-8.0, 0.0, num=101)
    err = np.array([abs(f_vec(theta + tk * v, X, s, LAM) - f0 - tk * slope_dir)
                    for tk in t])

    np.savetxt(RESULTS_DIR / "q3_gradient_check.csv",
               np.column_stack((t, err)), delimiter=",", header="t,error", comments="")

    # slope of the straight part (above the round-off floor, below saturation)
    mask = (t >= 1e-4) & (t <= 1e-2) & (err > 0)
    slope, intercept = np.polyfit(np.log10(t[mask]), np.log10(err[mask]), 1)
    print(f"Estimated log-log slope on [1e-4, 1e-2]: {slope:.4f}")

    C = np.median(err[mask] / t[mask] ** 2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t, err, label="Taylor remainder")
    ax.loglog(t, C * t ** 2, "--", label=r"$O(t^2)$ reference")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|f_\lambda(\theta+tv)-f_\lambda(\theta)"
                  r"-t\langle v,\nabla f_\lambda(\theta)\rangle|$")
    ax.set_title("Gradient check")
    ax.grid(True, which="both", ls=":"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "q3_gradient_check.pdf")
    plt.close(fig)

    with open(RESULTS_DIR / "q3_slope.txt", "w") as fh:
        fh.write(f"log-log slope on [1e-4, 1e-2]: {slope:.6f}\n")


# ==================================================================
# Q4: fixed-step gradient descent
# ==================================================================
def question4(X, s):
    print("\n=== Q4: fixed-step gradient descent ===")
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(X.shape[0])
    theta0 = theta.copy()

    # sigma_max(X)^2 = largest eigenvalue of X X^T (785 x 785): cheap
    sigma_max = float(np.sqrt(np.linalg.eigvalsh(X @ X.T)[-1]))
    L = sigma_max ** 2 + LAM
    step = 1.0 / L

    f, g = f_and_grad(theta, X, s, LAM)
    g0_norm = float(np.linalg.norm(g))
    objectives = [f]
    grad_norms = [g0_norm]

    start = time.perf_counter()
    reason = None
    k = 0
    while True:
        if grad_norms[-1] <= TOL * g0_norm:
            reason = "gradient_tolerance"
            break
        if time.perf_counter() - start >= MAX_TIME:
            reason = "time_limit"
            break
        theta = theta - step * g
        k += 1
        f, g = f_and_grad(theta, X, s, LAM)      # one pass per iteration
        objectives.append(f)
        grad_norms.append(float(np.linalg.norm(g)))
    elapsed = time.perf_counter() - start

    objectives = np.array(objectives)
    grad_norms = np.array(grad_norms)
    print(f"sigma_max = {sigma_max:.6e},  L = {L:.6e},  step = {step:.6e}")
    print(f"iterations = {k},  time = {elapsed:.2f}s,  stop = {reason}")
    print(f"||g0|| = {g0_norm:.6e},  ||g_final|| = {grad_norms[-1]:.6e}, "
          f"ratio = {grad_norms[-1]/g0_norm:.3e}")
    print(f"f(theta_0) = {objectives[0]:.6e},  f(theta_final) = {objectives[-1]:.6e}")

    np.savez(RESULTS_DIR / "q4_gd_history.npz",
             iterations=np.arange(k + 1), objectives=objectives,
             gradient_norms=grad_norms, theta0=theta0, theta_final=theta,
             step_size=step, sigma_max=sigma_max, L=L, lam=LAM, seed=SEED)
    np.savetxt(RESULTS_DIR / "q4_gd_history.csv",
               np.column_stack((np.arange(k + 1), objectives, grad_norms)),
               delimiter=",", header="k,objective,gradient_norm", comments="")
    with open(RESULTS_DIR / "q4_stopping_reason.txt", "w") as fh:
        fh.write(f"Stopping reason: {reason}\nIterations: {k}\n"
                 f"Elapsed time: {elapsed:.6f} s\n")
    return theta, objectives, grad_norms


# ==================================================================
# Q5: convergence plots
# ==================================================================
def question5(objectives, grad_norms):
    print("\n=== Q5: convergence plots ===")
    k = np.arange(objectives.size)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Log axes: f decreases by orders of magnitude towards a positive limit,
    # and ||grad f|| decays (roughly) geometrically -> straight line on a log axis.
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

    question2(X_tr, s_tr)
    question3(X_tr, s_tr)
    theta_final, objectives, grad_norms = question4(X_tr, s_tr)
    question5(objectives, grad_norms)
    question7(theta_final, X_tr, y_tr, X_te, y_te)
    print(f"\nAll results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
