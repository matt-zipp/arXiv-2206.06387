# Supplemental material to https://arxiv.org/abs/2206.06387.
# Code authors:  Matthias Zipper & Pascal Bassler
"""
Solvers for the gate synthesis problem.

Numerically solves the problems Eq. 9 (linear program, LP) and Eq. 18 (mixed integer program, MIP) of the paper.
The two important methods are `lp_approach(n, m)` and `mip_approach(n, m)`, each to be called with a qubit number `n`
and a symmetric matrix w/o diagonal `m` that represents the target gate unitary (after divison by the coupling matrix).
Note that the corresponding solvers (GLPK for the LP, MOSEK for the MIP) have to be installed.
"""

from typing import Union

import itertools as it

import numpy as np
import cvxpy as cp


# ======================
# === UTILITY CACHES ===
# ======================

tril_indices = {}  # cache for index pairs in the lower triangle of an nxn matrix
outer_prods = {}  # cache for the matrices of the linear equation system (columns are vectorized outer products)


def get_tril_indices(n: int, *, save_in_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Computes the indices of the lower triangle of an nxn matrix, or retrieves them from cache."""
    try:
        return tril_indices[n]
    except KeyError:
        res = np.tril_indices(n, -1)
        if save_in_cache:
            tril_indices[n] = res
        return res


def get_outer_prods(n: int, *, save_in_cache: bool = True) -> np.ndarray:
    """Computes the LP matrix for n qubits, or retrieves it from cache."""
    try:
        return outer_prods[n]
    except KeyError:
        ms = [(1,) + tail for tail in it.product([1, -1], repeat=(n-1))]
        i_lower = get_tril_indices(n, save_in_cache=save_in_cache)
        res = np.column_stack([np.outer(m, m)[i_lower] for m in ms])
        if save_in_cache:
            outer_prods[n] = res
        return res


# =================================
# === ENCODING SEQUENCE SOLVERS ===
# =================================

def lp_approach(n: int, m: np.ndarray, *, save_in_cache: bool = True, threshold: Union[float, None] = 1e-12) \
        -> np.ndarray:
    """
    Compute the encoding times to implement a GZZ gate by solving the LP of Eq. 9 using GLPK.

    ARGUMENTS:
        n: int - Number of qubits.
        m: ndarray - Data specifying the target gate. Either a symmetric matrix with zero diagonal
                                                      or a vectorized lower triangle of the former.
    OPTIONAL KEYWORD-ONLY ARGUMENTS:
        save_in_cache: bool - Whether to save newly computed helper data in module-level cache for later usage.
        threshold: float, default = 1e-12 - Threshold to truncate tiny entries in the result produced by floating-point
                                            arithmetic. `None` means no truncation.
    RETURNS:
        ndarray of 2^(n-1) encoding times (cf. symmetry argument below Eq. 9)
    """
    # check input:
    if m.shape == (n, n):  # input is square matrix...
        y = m[get_tril_indices(n, save_in_cache=save_in_cache)]  # ... and lower triangle is extracted
    elif m.shape == (n*(n-1) // 2,):  # input is already vectorized
        y = m
    else:
        raise ValueError(f"Invalid input shape {m.shape}")
    # build LP as CVXPY model
    x = cp.Variable(1 << (n-1), nonneg=True)  # non-negativity constraint
    objective = cp.Minimize(cp.norm(x, 1))  # L1 objective function
    constraints = [get_outer_prods(n, save_in_cache=save_in_cache) @ x == y]  # linear equation system
    prob = cp.Problem(objective, constraints)
    # solve LP using simplex method
    prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
    # extract solution and potentially suppress negligibly small entries
    solution = x.value
    if threshold is not None:
        solution[solution < threshold] = 0.0
    # return result
    return solution


def mip_approach(n: int, m: np.ndarray, alpha: float, c_l: float, c_u_scaling: float, rel_opt_tol: float,
                 *, save_in_cache: bool = True, threshold: Union[float, None] = 1e-12) -> np.ndarray:
    """
    Compute the encoding times to implement a GZZ gate by solving the MIP of Eq. 18 using MOSEK.

    ARGUMENTS:
        n: int - Number of qubits.
        m: ndarray - Data specifying the target gate. Either a symmetric matrix with zero diagonal
                                                      or a vectorized lower triangle of the former.
        alpha: float - Weighting parameter between total time and sparsity in the objective function (between 0 and 1).
        c_l: float - Lower threshold for non-zero encoding times.
        c_u_scaling: float - Proporionality constant for the adaptive upper bound of the non-zero encoding times.
        rel_opt_tol: float - Relative optimality tolerance employed by the mixed-integer optimizer.
    OPTIONAL KEYWORD-ONLY ARGUMENTS:
        save_in_cache: bool - Whether to save newly computed helper data in module-level cache for later usage.
        threshold: float, default = 1e-12 - Threshold to truncate tiny entries in the result produced by floating-point
                                            arithmetic. `None` means no truncation.
    RETURNS:
        ndarray of 2^(n-1) encoding times (cf. symmetry argument below Eq. 9)
    """
    # check input:
    if m.shape == (n, n):  # input is square matrix...
        y = m[get_tril_indices(n, save_in_cache=save_in_cache)]  # ... and lower triangle is extracted
    elif m.shape == (n*(n-1) // 2,):  # input is already vectorized
        y = m
    else:
        raise ValueError(f"Invalid input shape {m.shape}")
    # build MIP as CVXPY model
    n2 = 1 << (n-1)
    z = cp.Variable(n2, boolean=True)  # integer variables (either 0 or 1)
    t = cp.Variable(n2, nonneg=True)  # non-negative continuous variables
    # simplify extreme cases of the objective function
    if alpha <= 0.:
        objective = cp.Minimize(cp.norm(z, 1))
    elif alpha >= 1.:
        objective = cp.Minimize(cp.norm(t, 1))
    else:
        objective = cp.Minimize(alpha * cp.norm(t, 1) + (1.-alpha) * cp.norm(z, 1))
    c_u = c_u_scaling * np.max(y)  # compute adaptive upper bound
    constraints = [cp.multiply(c_l, z) <= t, t <= cp.multiply(c_u, z)]  # limit constraints
    constraints.append(get_outer_prods(n, save_in_cache=save_in_cache) @ t == y)  # linear equation system
    prob = cp.Problem(objective, constraints)
    # solve MIP using MOSEK
    prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={"MSK_DPAR_MIO_TOL_REL_GAP": rel_opt_tol})
    # extract solution and potentially suppress negligibly small entries
    solution = t.value
    if threshold is not None:
        solution[solution < threshold] = 0.0
    # return result
    return solution


# =================
# === UTILITIES ===
# =================

def create_random_binary_m(n: int) -> np.ndarray:
    tril = np.tril(np.random.randint(0, 2, (n, n), np.uint8), -1)
    return tril + tril.T


# =====================
# === USAGE EXAMPLE ===
# =====================

if __name__ == '__main__':
    for n in [4, 5, 6]:
        print(f"=== EXAMPLE ON {n} QUBITS ===")
        m = create_random_binary_m(n)
        print("Symmetric matrix with zero diagonal:")
        print(m)
        print("GZZ encoding times, found by solving an LP:")
        print(lp_approach(n, m))
