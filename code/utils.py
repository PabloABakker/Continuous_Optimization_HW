# Script containing utility functions for the project.
import numpy as np
import time


# Scalar versions used only in the loop implementation
def phi_scalar(z):
    if z <= -1.0:
        return 0.0
    elif z <= 0.0:
        return 0.5 * (z + 1.0) ** 2
    else:
        return 0.5 + z

def dphi_scalar(z):
    if z <= -1.0:
        return 0.0
    elif z <= 0.0:
        return z + 1.0
    else:
        return 1.0

# Loop implementation 
def f_loop(theta, X, s, lam):
    total = 0.0
    for i in range(X.shape[1]):
        z = s[i] * np.dot(theta, X[:, i])
        total += phi_scalar(z)
    return total + 0.5 * lam * np.dot(theta, theta)

def grad_loop(theta, X, s, lam):
    g = np.zeros_like(theta)
    for i in range(X.shape[1]):
        z = s[i] * np.dot(theta, X[:, i])
        g += s[i] * dphi_scalar(z) * X[:, i]
    return g + lam * theta


# Vectorized implementation 
def phi(z):
    t = np.clip(z + 1.0, 0.0, 1.0)
    return 0.5 * t ** 2 + np.maximum(z, 0.0)

def dphi(z):
    return np.clip(z + 1.0, 0.0, 1.0)

def f_vec(theta, X, s, lam):
    z = s * (X.T @ theta)
    return np.sum(phi(z)) + 0.5 * lam * (theta @ theta)

def grad_vec(theta, X, s, lam):
    z = s * (X.T @ theta)
    return X @ (s * dphi(z)) + lam * theta

def best_time(fn, *args, repeats=20):
    fn(*args)                         
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times)