# Simulation check for Iteration 2.
# We build a rating matrix where we KNOW the true answer (rank 3),
# hide 70% of it, and check that ALS behaves the way our math says.
# Run from the repo root:  python src/simulation.py

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# --- 1. Make fake data with a known answer ---
n_users, n_movies, k, lam = 100, 80, 3, 1.0

R_true = rng.normal(size=(n_users, k)) @ rng.normal(size=(k, n_movies))  # exact rank 3
R = R_true + 0.1 * rng.normal(size=(n_users, n_movies))                 # add small noise
mask = rng.random((n_users, n_movies)) < 0.3                            # True = observed (30%)


def objective(U, V):
    # Q1: squared error on observed ratings + regularization
    error = (R - U @ V.T) * mask
    return np.sum(error**2) + lam * (np.sum(U**2) + np.sum(V**2))


def rmse(pred, truth, where):
    return np.sqrt(np.mean((pred - truth)[where] ** 2))


# --- 2. Run ALS ---
U = rng.normal(size=(n_users, k))
V = rng.normal(size=(n_movies, k))

J_values = [objective(U, V)]
smallest_eig = 999       # for Q3
biggest_grad = 0         # for Q2
went_up = 0              # for Q4

for it in range(50):
    J_old = objective(U, V)

    # update every user (Q2 formula)
    for u in range(n_users):
        Vu = V[mask[u]]                  # movies this user rated
        ru = R[u, mask[u]]               # their ratings
        A = Vu.T @ Vu + lam * np.eye(k)
        b = Vu.T @ ru
        U[u] = np.linalg.solve(A, b)

        smallest_eig = min(smallest_eig, np.linalg.eigvalsh(A).min())
        grad = 2 * A @ U[u] - 2 * b
        biggest_grad = max(biggest_grad, np.abs(grad).max())

    J_mid = objective(U, V)

    # update every movie (same formula, roles swapped)
    for i in range(n_movies):
        Ui = U[mask[:, i]]
        ri = R[mask[:, i], i]
        A = Ui.T @ Ui + lam * np.eye(k)
        V[i] = np.linalg.solve(A, Ui.T @ ri)

    J_new = objective(U, V)
    if J_mid > J_old + 1e-9:
        went_up += 1
    if J_new > J_mid + 1e-9:
        went_up += 1
    J_values.append(J_new)

R_als = U @ V.T

# --- 3. Q5: compare with "fill missing with 0, then SVD" ---
R_zero = np.where(mask, R, 0)
W, s, Zt = np.linalg.svd(R_zero)
R_svd = W[:, :k] @ np.diag(s[:k]) @ Zt[:k]

# --- 4. Print results ---
print("observed fraction:", round(mask.mean(), 3), "(", mask.sum(), "entries )")
print("Q2  biggest gradient at a user update:", biggest_grad)
print("Q3  smallest eigenvalue of A:", round(smallest_eig, 3), "(should be >= lambda =", lam, ")")
print("Q4  times J went up:", went_up, "out of 100 half-steps")
print("    J start, after 1 sweep, final:", round(J_values[0], 1), round(J_values[1], 1), round(J_values[-1], 1))
print("true singular values:", np.round(np.linalg.svd(R_true, compute_uv=False)[:4], 2))
print("ALS  singular values:", np.round(np.linalg.svd(R_als, compute_uv=False)[:4], 2))
print("Q5  RMSE on hidden entries, ALS:      ", round(rmse(R_als, R_true, ~mask), 3))
print("    RMSE on hidden entries, zero+SVD: ", round(rmse(R_svd, R_true, ~mask), 3))

# --- 5. Plot J ---
plt.plot(J_values, marker="o", markersize=3)
plt.yscale("log")
plt.xlabel("ALS sweep")
plt.ylabel("objective J (log scale)")
plt.title("J never goes up")
plt.tight_layout()
plt.savefig("figures/sim_objective.png", dpi=200)
print("saved figures/sim_objective.png")
