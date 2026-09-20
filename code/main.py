# Import libraries
from tkinter import _test
import numpy as np
from scipy.io import loadmat
from utils import *

# Set random seed for reproducibility
rng = np.random.default_rng(0) 

# Create Paths
DATA_PATH = '../data/mnist_train_test.mat'
RESULTS_DIR = '../results'

# Load data and inspect it
data = loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
train, test = data['train'], data['test']

# Cast to float
X_train = np.asarray(train.X, dtype=np.float64)   
y_train = np.asarray(train.y, dtype=np.float64)   
X_test = np.asarray(test.X, dtype=np.float64)    
y_test = np.asarray(test.y, dtype=np.float64)   

# Sanity checks
print("train.X:", X_train.shape, " train.y:", y_train.shape)
print("test.X: ", X_test.shape, " test.y: ", y_test.shape)
print("labels in train:", np.unique(y_train), " labels in test:", np.unique(y_test))
print("last row of X is all ones:", np.all(X_train[-1] == 1), np.all(X_test[-1] == 1))
print("pixel range:", X_train[:-1].min(), X_train[:-1].max())

assert X_train.shape[0] == 28 * 28 + 1 and X_test.shape[0] == 28 * 28 + 1
assert X_train.shape[1] == y_train.shape[0] and X_test.shape[1] == y_test.shape[0]

# Signs s_i = 1 - 2 y_i  
s_train = 1.0 - 2.0 * y_train
s_test = 1.0 - 2.0 * y_test

# Gradient descent loop
lam = 0.1
d = X_train.shape[0]

# Numerical agreement, on several thetas so all 3 pieces of phi are hit
for scale in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
    theta = scale * rng.standard_normal(d)
    fl, fv = f_loop(theta, X_train, s_train, lam), f_vec(theta, X_train, s_train, lam)
    gl, gv = grad_loop(theta, X_train, s_train, lam), grad_vec(theta, X_train, s_train, lam)
    rel_f = abs(fl - fv) / max(abs(fl), 1e-300)
    rel_g = np.linalg.norm(gl - gv) / max(np.linalg.norm(gl), 1e-300)
    print(f"scale={scale:6.0e}  rel err f = {rel_f:.2e}   rel err grad = {rel_g:.2e}")

# Sanity check of the gradient formula itself (central finite differences)
theta = 1e-2 * rng.standard_normal(d)
u = rng.standard_normal(d); u /= np.linalg.norm(u)
h = 1e-6
fd = (f_vec(theta + h*u, X_train, s_train, lam) - f_vec(theta - h*u, X_train, s_train, lam)) / (2*h)
print("directional derivative: finite diff =", fd, " grad.u =", grad_vec(theta, X_train, s_train, lam) @ u)

# Timing check of the loop vs vectorized implementations
theta = 1e-2 * rng.standard_normal(d)
for name, fl_, fv_ in [("f", f_loop, f_vec), ("grad", grad_loop, grad_vec)]:
    tl = best_time(fl_, theta, X_train, s_train, lam, repeats=5)
    tv = best_time(fv_, theta, X_train, s_train, lam, repeats=50)
    print(f"{name}: loop {tl[0]*1e3:.2f} ms (min) / {tl[1]*1e3:.2f} ms (median), "
          f"vec {tv[0]*1e3:.3f} ms / {tv[1]*1e3:.3f} ms, speed-up x{tl[1]/tv[1]:.0f}")