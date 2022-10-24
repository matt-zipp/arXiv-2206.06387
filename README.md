# Synthesis of and compilation with time-optimal multi-qubit gates

Code by Matthias Zipper & Pascal Bassler related to [arXiv:2206.06387](https://arxiv.org/abs/2206.06387).

This repository contains three modules:
- `gzz_synthesis.py`: Solvers for the GZZ time-optimal gate synthesis problem.
  Contains implementations for both the linear program (LP) and mixed integer program (MIP) approach,
  but depends on external solver libraries (GLPK, MOSEK).
- `couplings.py`: Functions to compute the Ising coupling matrix in a MAGIC-based ion trap QIP.
  Contains implementations of the Coulomb interaction potential and a quadratic external potential,
  as well as a rudimentary framework to extend this to other potentials.
- `directed_cx.py`: Implementations of the Algorithms 1 & 2 of the paper.
  Manipulates Hadamard and CZ gate configurations to heuristically minimize the number and complexity of GCZ gates
  needed to implement a given directed CX layer.

All modules contain explanatory docstrings, comments and usage examples.
