# MATH-329 Continuous Optimization — Homework 1 (Group S)

Andrés Alarcón Navarro · Pablo Agustin Bakker · Gabriel Alberto Klingler Mora · Timo Lassoeur

## How to run

From the `code/` folder:

```bash
python main_andres.py
```

This reproduces every file in `results/` except the four `q4_step_size_*` ones.
It reads `../data/mnist_train_test.mat` and takes roughly one minute, most of
it the Q4 gradient descent (~11570 iterations, ~21 s).

The step-size grid search is a separate script, run once:

```bash
python step_size_experiment.py          # ~4 minutes
```

It is deliberately **not** called by `main_andres.py`: the chosen step size is
hard-coded there as `CHOSEN_C = 1.9`, so the graded run is deterministic and
does not pay for the search.

## Where each answer is saved

### Question 2 — objective and gradient

| File | Contents |
|---|---|
| `q2_summary.json` | Relative difference between the loop and vectorised implementations of `f_lambda` and its gradient, at three scales of `theta`; branch counts of `phi` at each scale; runtimes and speed-ups. |

### Question 3 — gradient check

| File | Contents |
|---|---|
| `q3_gradient_check.csv` | `t` and the Taylor remainder `abs(f(theta+t*v) - f(theta) - t*<v, grad f(theta)>)`, 101 values of `t` logarithmically spaced in `[1e-8, 1]`. |
| `q3_slope.txt` | The log–log slope, the fitted window, the round-off floor, and the branch counts of `phi` at the `theta` used. |
| `q3_gradient_check.pdf` | The remainder against `t` in log–log coordinates, with an `O(t^2)` reference, the round-off floor and the fitted window. |

### Question 4 — fixed-step gradient descent

**The per-iteration history required by the question is `q4_gd_history.csv`.**

| File | Contents |
|---|---|
| **`q4_gd_history.csv`** | **One row per iteration `k = 0 … 11570`, including the initial point: `k`, the objective `f_lambda(theta_k)`, and the gradient norm `norm(grad f_lambda(theta_k))`.** |
| `q4_gd_history.npz` | The same two sequences, plus `theta0`, `theta_final`, `step_size`, `c`, `L`, `sigma_max`, `lambda` and `seed`. |
| `q4_stopping_reason.txt` | **The stopping reason**, the iteration count, the elapsed time, the step size, and the initial and final values of the objective and the gradient norm. |
| `q4_step_size_experiment.txt` | The step-size search: the ranking of the candidates from three starting points, and the iterations each needs to reach the stopping rule. Written by `step_size_experiment.py`. |
| `q4_step_size_comparison.pdf` | Gradient-norm decay for each candidate step size. Written by `step_size_experiment.py`. |

### Question 5 — convergence plots

| File | Contents |
|---|---|
| `q5_convergence.pdf` | Objective value and gradient norm against the iteration, both on a logarithmic vertical axis, with the Corollary 4.32 bound for `alpha = 1.9/L` on the gradient panel. |
| `q5_theory_horizon.pdf` | The same bound carried out far enough to reach the stopping tolerance, on a logarithmic iteration axis. |

### Question 7 — classification

| File | Contents |
|---|---|
| `q7_errors.json` | Training and test classification error rates for `theta_final`. |
| `q7_theta_final.csv`, `q7_theta_final.npy` | The final iterate `theta_final` (785 entries: 784 pixel weights and the bias). |

Questions 1, 6 and 8 are theoretical and produce no files.

## Code

| File | Role |
|---|---|
| `main_andres.py` | Everything graded: data loading, both implementations of the objective and gradient, and Questions 2, 3, 4, 5 and 7. |
| `plot_utils.py` | The two Question 5 figures. Imports nothing from `main_andres.py`, so either figure can be redrawn from a saved history. |
| `step_size_experiment.py` | The one-off step-size search behind `CHOSEN_C`. |
| `main.py`, `utils.py` | Exploratory scratch work kept for reference. Not part of the graded pipeline and writes nothing to `results/`. |

## Conventions

- `s_i = 1 - 2*y_i`, so `y = 0` gives `s = +1` and `y = 1` gives `s = -1`.
- The last row of `X` is already the row of ones, so no bias row is appended.
- `lambda = 0.005`, and the seed for `theta_0` in Question 4 is `42`.
