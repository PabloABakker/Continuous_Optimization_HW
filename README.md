# MATH-329 Continuous Optimization — Homework 1 (Group S)

Andrés Alarcón Navarro · Pablo Agustin Bakker · Gabriel Alberto Klingler Mora · Timo Lassoeur

## Required packages

`numpy`, `scipy`, `matplotlib`.

```bash
pip install numpy scipy matplotlib
```

Versions used: Python 3.13.12, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9.
Matplotlib runs on the `Agg` backend, so no display is needed.

## How to run

From the `code/` folder:

```bash
python main.py
```

Reads `../data/mnist_train_test.mat` and writes everything requested in
Questions 2, 3, 4, 5 and 7 to `../results/`. Every run regenerates these
files from scratch; nothing is loaded from a previous run.

**Execution time: approximately 20 seconds** (measured on our machines; the
gradient-tolerance stopping rule in Question 4 makes the run length
essentially independent of hardware, since the iteration count — not a
wall-clock budget — decides when it stops).

### Note: the step-size search is not part of that run

```bash
python step_size_experiment.py          # approximately 4 minutes
```

This one-off script compares candidate step sizes `c/L` from several
starting points and is what produced `CHOSEN_C = 1.9` (hard-coded in
`main.py`) and the two files in
`results/q4_stepsize_selection_disregard/`. It is deliberately **not**
called by `main.py`, so the graded run stays under the 5-minute limit and is
deterministic; how we chose `1.9` is explained in the report (Question 4)
and documented by these two files.

## Saved results

### Question 2 — objective and gradient

| File | Contents |
|---|---|
| `q2_summary.json` | Relative difference between the loop and vectorized implementations of `f_lambda` and its gradient, at three scales of `theta` (chosen so all three branches of `phi` are exercised); median run times over repeated calls and the resulting speed-ups. |

### Question 3 — gradient check

| File | Contents |
|---|---|
| `q3_gradient_check.csv` | `t` and the Taylor remainder `abs(f(theta+t*v) - f(theta) - t*<v, grad f(theta)>)`, for the 101 values of `t` logarithmically spaced in `[1e-8, 1]` (`np.logspace(-8.0, 0.0, num=101)`), as required by the question. |
| `q3_slope.txt` | The theta scale used, `f_lambda(theta)`, the round-off floor, the fitted window, and the log–log slope. |
| `q3_gradient_check.pdf` | The required plot: the remainder against `t` in log–log coordinates, with an `O(t^2)` reference. |

### Question 4 — fixed-step gradient descent

| File | Contents |
|---|---|
| `q4_gd_history.csv` | One row per iteration `k`, including the initial point (`k=0`): the objective `f_lambda(theta_k)` and the gradient norm `norm(grad f_lambda(theta_k))`, as required by the question. |
| `q4_stopping_reason.txt` | The stopping reason, iteration count, elapsed time, step size, and the initial and final objective and gradient norm. |

On our machines this run stopped by `gradient_tolerance` (not the time
limit) after 11,570 iterations, in about 21.7 s, with step size
`alpha = 1.9/L`; see `q4_stopping_reason.txt` for the exact values from the
graded run.

The files behind the choice of step size sit in
`results/q4_stepsize_selection_disregard/`, in a folder of their own because
`python main.py` does **not** regenerate them — they come from the one-off
`step_size_experiment.py` described above:

| File | Contents |
|---|---|
| `q4_step_size_experiment.txt` | The step-size search: the ranking of the candidates from several starting points, and the iterations each needs to reach the stopping rule. |
| `q4_step_size_comparison.pdf` | Gradient-norm decay for each candidate step size. |

### Question 5 — convergence plots

| File | Contents |
|---|---|
| `q5_convergence.pdf` | The required plot: objective value and gradient norm against the iteration (Question 4's run), both on a logarithmic vertical axis, with the Corollary 4.32 bound for `alpha = 1.9/L` on the gradient panel. |
| `q5_theory_horizon.pdf` | The same theoretical bound carried out far enough to reach the stopping tolerance, on a logarithmic iteration axis — supporting material for the Question 6 discussion, not itself required by Question 5. |

### Question 7 — classification

| File | Contents |
|---|---|
| `q7_errors.json` | Training and test classification error rates for `theta_final`, as required by the question. |
| `q7_theta_final.csv` | The final iterate `theta_final` (785 entries: 784 pixel weights and the bias), as required by the question. |

## Code

| File | Role |
|---|---|
| `main.py` | Everything graded: data loading, both implementations of the objective and gradient (Question 2), and Questions 2, 3, 4, 5 and 7. |
| `plot_utils.py` | Every figure and the Question 4 file output. Imports nothing from `main.py`, so a figure can be redrawn from a saved history alone. |
| `step_size_experiment.py` | The one-off step-size search behind `CHOSEN_C` (not run by `main.py`; see above). |

See `DISCLAIMER.md` for tool-use disclosure.
