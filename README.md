# MATH-329 Continuous Optimization — Homework 1 (Group S)

Andrés Alarcón Navarro · Pablo Agustin Bakker · Gabriel Alberto Klingler Mora · Timo Lassoeur

## Required packages

Numpy, scipy and matplotlib
can
Versions used: Python 3.13.12, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9.
Matplotlib runs on the `Agg` backend, so no display is needed.

## How to run

From the `code/` folder:

```bash
python main.py
```

Reads from ../data/mnist_train_test.mat` and writes in "../results/"

**Execution time: approximately 20 seconds**

### Note : the step-size search is not part of that run

```bash
python step_size_experiment.py          # approximately 4 min 
```

This one-off script produced `CHOSEN_C = 1.9` and the two files already in
`results/q4_stepsize_selection(disregard)/`. It is deliberately **not** called
by `main.py`: the step size is hard-coded there, so the graded run is
deterministic and does not pay the four minutes.


## Saved results 

### Question 2 — objective and gradient

| File | Contents |
|---|---|
| `q2_summary.json` | Relative difference between the loop and vectorised implementations of `f_lambda` and its gradient, at three scales of `theta`; runtimes and speed-ups. |

### Question 3 — gradient check

| File | Contents |
|---|---|
| `q3_gradient_check.csv` | `t` and the Taylor remainder `abs(f(theta+t*v) - f(theta) - t*<v, grad f(theta)>)`, 101 values of `t` logarithmically spaced in `[1e-8, 1]`. |
| `q3_slope.txt` | The log–log slope, the fitted window and the round-off floor. |
| `q3_gradient_check.pdf` | The remainder against `t` in log–log coordinates, with an `O(t^2)` reference, the round-off floor and the fitted window. |

### Question 4 — fixed-step gradient descent

**The per-iteration history required by the question is `q4_gd_history.csv`.**

| File | Contents |
|---|---|
| **`q4_gd_history.csv`** | **One row per iteration `k = 0 … 11570`, including the initial point: `k`, the objective `f_lambda(theta_k)`, and the gradient norm `norm(grad f_lambda(theta_k))`.** |
| `q4_stopping_reason.txt` | **The stopping reason**, the iteration count, the elapsed time, the step size, and the initial and final values of the objective and the gradient norm. |
The two files behind the choice of step size sit in
`results/q4_stepsize_selection(disregard)/`, in a folder of their own because
`python main.py` does **not** regenerate them — they come from the one-off
`step_size_experiment.py`:

| File | Contents |
|---|---|
| `q4_step_size_experiment.txt` | The step-size search: the ranking of the candidates from three starting points, and the iterations each needs to reach the stopping rule. |
| `q4_step_size_comparison.pdf` | Gradient-norm decay for each candidate step size. |

### Question 5 — convergence plots

| File | Contents |
|---|---|
| `q5_convergence.pdf` | Objective value and gradient norm against the iteration, both on a logarithmic vertical axis, with the Corollary 4.32 bound for `alpha = 1.9/L` on the gradient panel. |
| `q5_theory_horizon.pdf` | The same bound carried out far enough to reach the stopping tolerance, on a logarithmic iteration axis. |

### Question 7 — classification

| File | Contents |
|---|---|
| `q7_errors.json` | Training and test classification error rates for `theta_final`. |
| `q7_theta_final.csv` | The final iterate `theta_final` (785 entries: 784 pixel weights and the bias). |



## Code

| File | Role |
|---|---|
| `main.py` | Everything graded: data loading, both implementations of the objective and gradient, and Questions 2, 3, 4, 5 and 7. |
| `plot_utils.py` | Every figure and the Question 4 file output. Imports nothing from `main.py`, so a figure can be redrawn from a saved history. |
| `step_size_experiment.py` | The one-off step-size search behind `CHOSEN_C`. |



