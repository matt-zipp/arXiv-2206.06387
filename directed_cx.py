# Supplemental material to https://arxiv.org/abs/2206.06387.
# Code authors:  Matthias Zipper & Pascal Bassler
"""
Transformation algorithms to reduce the number and complexity of GZZ gates needed to implement directed CX layers.

Essentially an implementation of Algorithms 1 & 2 of the paper.
"""

from typing import Union

import numpy as np

from qiskit import QuantumCircuit


# ===========================
# === CONVERSION ROUTINES ===
# ===========================

def _b_to_t_cz(n: int, b: np.ndarray) -> np.ndarray:
    """Transform the lower triangular matrix B of some GCX(B) to T_CZ as described below Eq. 30."""
    # Check input shape
    assert b.shape == (n, n)
    # Copy relevant lower triangle from input
    t_cz = np.tril(b, -1)
    # Put control on diagonal, if at least one target is present in column
    for j in range(n-1):
        if sum(t_cz[j+1:, j]):
            t_cz[j, j] = 1
    # Set trivial `1` to follow paper convention
    t_cz[-1, -1] = 1
    # Return result
    return t_cz


def _h_cols_to_t_h(n: int, h_cols: list[int]) -> np.ndarray:
    """Transform a list of H positions on qubits 1, ..., n-1 to T_H as described below Eq. 30."""
    # Create empty array
    t_h = np.zeros((n, n), dtype=np.uint8)
    # Populate by going through `n-1` list entries (`im1` meaning `i-1`)
    for im1, j in enumerate(h_cols):
        if j == 0:
            pass  # Cancellation has taken place, remains an all-zero row
        else:
            t_h[im1+1, [0, j]] = 1, 1  # Place H in the first and j-th row
    # Return result
    return t_h


def _fanouts_to_gcz(n: int, fanouts: list[np.ndarray]) -> np.ndarray:
    """Transform a list of fan-out gates (columns of T_CZ) to the A of some GCZ(A)."""
    a = np.zeros((n, n), dtype=np.uint8)
    for fanout in fanouts:
        try:
            ctrl = np.flatnonzero(fanout)[0]
            a[ctrl+1:, ctrl] = fanout[ctrl+1:]
        except IndexError:
            continue  # Ignore empty objects
    return a + a.T


# =======================
# === CORE ALGORITHMS ===
# =======================

def _algorithm_1_core(n: int, t_cz: np.ndarray) -> list[int]:
    """Core functionality of Algorithm 1. Should be called via public `algorithm_1()`."""
    # Assume T_H in default initial layout (as created by `default_h_layout(n)` below)
    # T_H does not have to be stored as a matrix, we can just store the position the H in each row has been moved to
    h_cols = []
    # -------------------------------------------
    # --- Algorithm 1 (Moving Hadamard gates) ---
    # -------------------------------------------
    h_max = 0  # Position of the rightmost H that has already been moved left
    for i in range(1, n):
        try:
            c = np.flatnonzero(t_cz[i, :i])[-1] + 1  # Find position directly after first CZ to the left
        except IndexError:
            # No CZ found
            h_cols.append(0)  # Cancel H in the first layer
            continue
        if c == i:
            # Unable to move left, attempt to move right
            if len(np.flatnonzero(t_cz[i, i:])) == 0:
                # If no CZ is to the right, move $\Gate{H}$ to the last layer, ...
                h_cols.append(n-1)
            else:
                # ...otherwise remain in place.
                h_cols.append(i)
                h_max = i
        else:
            h_max = max(h_max, c)  # Find the more restrictive condition (either CZ on current qubit or H on previous)
            h_cols.append(h_max)  # Move H to the target layer
    # Return raw list
    return h_cols


def _algorithm_2_core(n: int, t_h: np.ndarray, t_cz: np.ndarray) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Core functionality of Algorithm 2. Should be called via public `algorithm_2()`."""
    sequence = []
    current_gcz = []
    for j in range(n-1):
        if sum(t_h[:, j]) > 0:  # There is at least one hadamard gate at column j
            if sum(t_h[:, j+1]) > 0:  # If the ith column is surrounded by hadamard layers.
                if current_gcz:  # If CZ is non empty, then append the next column of t_cz to CZ
                    if sum(t_cz[:, j]) > 1:
                        current_gcz.append(t_cz[:, j])
                    sequence.append(current_gcz)  # Add CZ to the sequence of global CZ gates
                    current_gcz = []
                else:  # CZ is empty
                    twoq_cz = np.copy(t_h[:, j+1])
                    twoq_cz[j] = 1
                    sequence.append([twoq_cz])  # Append a two qubit CZ (first part of the column of t_cz)
                    remainder_cz = (t_cz[:, j] + t_h[:, j+1]) % 2  # The second part of the column of t_cz
                    if sum(remainder_cz) < 2:
                        current_gcz = []
                    else:
                        current_gcz = [remainder_cz]
            else:
                if sum(t_cz[:, j]) > 1:
                    current_gcz.append(t_cz[:, j])
                sequence.append(current_gcz)
                current_gcz = []
        else:
            if sum(t_cz[:, j]) > 1:
                sequence[-1].append(t_cz[:, j])
    # Compress T_H
    t_h = np.delete(t_h, np.argwhere(np.all(t_h[..., :] == 0, axis=0)), axis=1)  # truncate H pattern
    # Return results
    return t_h, sequence


# =================================
# === PUBLIC ALGORITHM ROUTINES ===
# =================================

def algorithm_1(n: int, t_cz: np.ndarray, return_raw_list: bool = False) -> Union[np.ndarray, list[int]]:
    """
    Implementation of Algorithm 1 (Moving Hadamard gates).

    This method does not need T_H as an input, but instead assumes the default layout shown in Eq. 30.

    ARGUMENTS:
        n: int - Number of qubits.
        t_cz: ndarray - Configuration of CZ gates. Determines the mobility of H gates.
    OPTIONAL ARGUMENTS:
        return_raw_list: bool - Default: False. Whether to proxy the internal result of the core function or
                                                convert the result to the format described in the paper.
    """
    # Run core algorithm
    h_cols = _algorithm_1_core(n, t_cz)
    # Handle sophisticated return
    if return_raw_list:
        # Directly return the list constructed by the loop above
        return h_cols
    else:
        # Build T_H as in the paper
        return _h_cols_to_t_h(n, h_cols)


def algorithm_2(n: int, t_h: np.ndarray, t_cz: np.ndarray, return_raw_list: bool = False) \
        -> tuple[np.ndarray, Union[list[np.ndarray], list[list[np.ndarray]]]]:
    """
    Implementation of Algorithm 2 (Moving CZ gates).

    ARGUMENTS:
        n: int - Number of qubits.
        t_h: ndarray - Configuration of H gates. Determines the mobility of CZ gates.
        t_cz: ndarray - Configuration of CZ gates.
    OPTIONAL ARGUMENTS:
        return_raw_list: bool - Default: False. Whether to proxy the internal result of the core function or
                                                convert the result to the format described in the paper.
    """
    # Run core algorithm
    t_h, sequence = _algorithm_2_core(n, t_h, t_cz)
    # Handle sophisticated return
    if return_raw_list:
        return t_h, sequence
    else:
        return t_h, [_fanouts_to_gcz(n, fanouts) for fanouts in sequence]


def directed_cx_to_t_h_and_gcz_seq(n: int, b: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    An end-to-end concatenation of Algorithms 1 & 2.

    Turns the matrix B specifying some directed CX layer GCX(B) into a sequence of GCZ gates with Hadamards inbetween.

    ARGUMENTS:
        n: int - Number of qubits.
        b: ndarray - Input matrix B specifying some directed CX layer GCX(B).
    RETURNS:
        T_H, a matrix holding the positions of Hadamard gates.
        [A, ...], a list with the matrices A specifying layers of GCZ(A) gates.
    """
    t_cz = _b_to_t_cz(n, b)
    return algorithm_2(n, algorithm_1(n, t_cz), t_cz)


# =================
# === UTILITIES ===
# =================

def create_random_directed_cx(n: int):
    # Create and return random binary strictly lower triangular matrix
    return np.tril(np.random.randint(0, 2, (n, n), np.uint8), -1)


def default_h_layout(n: int) -> np.ndarray:
    # Prepare default Hadamard layout (Eq. 31)
    return _h_cols_to_t_h(n, range(1, n))
    # Equivalent to:
    #   t_h = np.eye(n, dtype=np.uint8)
    #   t_h[0, 0] = 0
    #   t_h[1:, 0] = np.ones(n-1, dtype=np.uint8)
    #   return t_h


def directed_cx_to_t_h_and_gcz_seq_sans_algo(n: int, b: np.ndarray):
    # Trivial conversion of GCX(B) into (T_H, [GCZ(A), ...]) for before/after comparison of algorithms
    t_cz = _b_to_t_cz(n, b)
    return default_h_layout(n), [_fanouts_to_gcz(n, [fanout]) for fanout in t_cz.T[:-1]]


def _append_gcz(n: int, qc: QuantumCircuit, a: np.ndarray):
    for i in range(n):
        for j in range(i):
            if a[i, j]:
                qc.cz(i, j)


def to_qiskit(n: int, t_h, sequence):
    assert t_h.shape[1] == len(sequence) + 1
    qc = QuantumCircuit(n)
    for h_lay, gcz in zip(t_h.T, sequence):
        for i in np.flatnonzero(h_lay):
            qc.h(i)
        qc.barrier()
        if isinstance(gcz, list):
            gcz = _fanouts_to_gcz(n, gcz)
        _append_gcz(n, qc, gcz)
        qc.barrier()
    for i in np.flatnonzero(t_h[:, -1]):
        qc.h(i)
    return qc


# =====================
# === USAGE EXAMPLE ===
# =====================

if __name__ == '__main__':
    for n in [4, 5, 6]:
        print(f"=== EXAMPLE ON {n} QUBITS ===")
        b = create_random_directed_cx(n)
        print("Directed CX matrix:")
        print(b)
        print("Initial CZ circuit:")
        print(to_qiskit(n, *directed_cx_to_t_h_and_gcz_seq_sans_algo(n, b)))
        print("CZ circuit after algorithms 1&2:")
        print(to_qiskit(n, *directed_cx_to_t_h_and_gcz_seq(n, b)))
