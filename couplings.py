# Supplemental material to https://arxiv.org/abs/2206.06387.
# Code authors:  Matthias Zipper & Pascal Bassler
"""
Module used to compute the coupling matrix of a MAGIC-based ion trap QIP for a given axial trap potential.
"""

from typing import Union

import warnings
import itertools

import numpy as np
from numpy import linalg
from scipy.optimize import root


# =======================================
# === ABSTRACT CLASSES FOR POTENTIALS ===
# =======================================

class GlobalPotential1D:
    """A potential acting on `n` particles that live in one-dimensional space."""

    def __init__(self, n: int, **config):
        """
        n - number of particles

        Subclasses should accept arbitrary keyword arguments, of which they may use some to initialize their objects.
        """
        self._n = n

    @property
    def n(self):
        """Number of particles."""
        return self._n

    def potential(self, z):
        """Value of the potential at positions `z`."""
        raise NotImplementedError()  # abstract method

    def dk_tensor(self, z, k: int):
        """Value of the `k`-th order derivative tensor at positions `z`."""
        # One could give a default implementation in terms of `self.potential()` that computes
        # numerical derivatives, but we usually have better means.
        raise NotImplementedError()  # abstract method

    def gradient(self, z):
        """Convenience function for the first-order derivative."""
        return self.dk_tensor(z, 1)  # default implementation

    def hessian(self, z):
        """Convenience function for the second-order derivative."""
        return self.dk_tensor(z, 2)  # default implementation


class LocalPotential1D:
    """A potential acting on a single particle that lives in one-dimensional space."""

    def __init__(self, **config):
        """
        Subclasses should accept arbitrary keyword arguments, of which they may use some to initialize their objects.
        """
        pass

    def potential(self, z):
        """Value of the potential at position `z`."""
        raise NotImplementedError()  # abstract method

    def derivative(self, z, order: int = 1):
        """The one-dimensional derivative of order `order` at position `z`."""
        if order > 0:
            # Default implementation uses numerical differentiation.
            # Should be overwritten if derivatives are known algebraically.
            from scipy.misc import derivative
            return derivative(self.potential, z, n=order)
        elif order == 0:
            return self.potential(z)
        else:
            raise ValueError(f"`order` must be a non-negative integer, not: {order}")


class GlobalFromLocal(GlobalPotential1D):
    """
    The total potential on `n` particles caused by a local potential
    that acts independently on each individual particle.
    """

    def __init__(self, n: int, local: LocalPotential1D, **config):
        super().__init__(n, **config)
        self._local = local  # local potential

    def potential(self, z):
        """
        Value of the global potential at positions `z`, which is the sum of the
        local potential values at the different particle positions `z[i]`.
        """
        return sum(self._local.potential(z[i]) for i in range(self._n))

    def dk_tensor(self, z, k: int):
        """
        Value of the `k`-th order derivative tensor at positions `z`.

        Entries are only non-zero if all indices are identical, in which case the entry is simply
        the corresponding `k`-th derivative of the local potential.
        """
        return np.fromiter(map(lambda idx: (self._local.derivative(z[idx[0]], k) if (len(set(idx)) == 1) else 0),
                               itertools.product(range(self._n), repeat=k)),
                           np.float_, self._n**k).reshape(k*(self._n,))


# ===========================
# === CONCRETE POTENTIALS ===
# ===========================

class PairwiseCoulomb1D(GlobalPotential1D):
    """
    The mutual Coulomb potential of `n` identical charged particles in one spatial dimension.

    By default, assumes that particles have unit charge (q = 1) and units are such that the constant e^2/(4*pi*eps_0)
    has value 1 (e2_4pieps0 = 1), but alternative values can be passed via keyword to the constructor.
    """

    def __init__(self, n: int, **config):
        super().__init__(n, **config)
        q = config.get('q', 1)
        e2_4pieps0 = config.get('e2_4pieps0', 1)
        self._q2_4pieps0 = q*q * e2_4pieps0  # (q*e)^2/(4*\pi*\eps_0)

    @staticmethod
    def _potential_raw_function(n: int, z):
        """Sum of inverse distances; no prefactors yet."""
        return sum(1 / abs(z[i] - z[j]) for i in range(n) for j in range(i))

    @staticmethod
    def _gradient_raw_function(n: int, z):
        """
        Exact gradient of ._potential_raw_function().

        Returns a function that maps gradient indices to corresponding gradient entries.
        """
        def gradient_entry(i):
            return (sum(1 / (z[i] - z[j]) ** 2 for j in range(i+1, n))
                    - sum(1 / (z[i] - z[j]) ** 2 for j in range(i)))
        return gradient_entry

    @staticmethod
    def _hessian_raw_function(n: int, z):
        """
        Exact Hessian of ._potential_raw_function().

        Returns a function that maps Hessian index pairs to corresponding Hessian entries.
        """
        def hessian_entry(ij):
            i, j = ij
            if i == j:
                return sum(1 / abs(z[i] - z[k]) ** 3 for k in range(n) if k != i)
            else:
                return -1 / abs(z[i] - z[j]) ** 3
        return hessian_entry

    @staticmethod
    def _d3_tensor_raw_function(n: int, z):
        """
        Exact third derivative tensor of ._potential_raw_function().

        Returns a function that maps index triples to corresponding tensor entries.

        This is not required for a standard leading order MAGIC analysis,
        but can be handy in higher-order analyses.
        """
        def d3_two_sites(i, k):
            if i > k:
                return 1 / (z[i] - z[k]) ** 4
            else:
                return -1 / (z[i] - z[k]) ** 4
        def d3_tensor_entry(ijk):
            i, j, k = ijk
            if i == j == k:
                return (sum(1 / (z[i] - z[l]) ** 4 for l in range(i+1, n))
                        - sum(1 / (z[i] - z[l]) ** 4 for l in range(i)))
            elif i == j:
                return d3_two_sites(i, k)
            elif j == k:
                return d3_two_sites(j, i)
            elif k == i:
                return d3_two_sites(k, j)
            else:
                return 0
        return d3_tensor_entry

    def potential(self, z):
        """Pairwise Coulomb potential at particle positions `z`."""
        return self._q2_4pieps0 * self._potential_raw_function(self._n, z)

    def gradient(self, z):
        return self._q2_4pieps0 * np.fromiter(map(self._gradient_raw_function(self._n, z),
                                                  range(self._n)),
                                              np.float_, self._n)

    def hessian(self, z):
        return 2*self._q2_4pieps0 * np.fromiter(map(self._hessian_raw_function(self._n, z),
                                                    itertools.product(range(self._n), repeat=2)),
                                                np.float_, self._n**2).reshape(2*(self._n,))

    def dk_tensor(self, z, k: int):
        if k == 0:
            return self.potential(z)
        elif k == 1:
            return self.gradient(z)
        elif k == 2:
            return self.hessian(z)
        elif k == 3:
            return 6*self._q2_4pieps0 * np.fromiter(map(self._d3_tensor_raw_function(self._n, z),
                                                        itertools.product(range(self._n), repeat=3)),
                                                    np.float_, self._n**3).reshape(3*(self._n,))
        else:
            # Higher derivatives could be computed numerically, but are unlikely to be needed.
            raise NotImplementedError()


class Quadratic1D(LocalPotential1D):
    """
    A simple parabolic potential of the form 1/2*a*z^2.

    By default, assumes that units are such that the constant in the potential has value 1.
    Alternative values can be passed to the constructor via keyword `mw2` (alluding to the harmonic oscillator value).
    """

    def __init__(self, **config):
        super().__init__(**config)
        self._mw2 = config.get('mw2', 1)  # m*\omega^2

    def potential(self, z):
        return (self._mw2 * z**2) / 2

    def derivative(self, z, order: int = 1):
        if order == 0:
            return self.potential(z)
        elif order == 1:
            return self._mw2 * z
        elif order == 2:
            return self._mw2
        elif order < 0:
            raise ValueError("Derivative order has to be non-negative")
        else:
            # all derivatives beyond the second vanish identically
            return 0


# ============================================
# === FUNCTIONS TOWARDS THE MAGIC COUPLING ===
# ============================================

def get_equilibrium_positions(n: Union[int, GlobalPotential1D],
                              external: Union[LocalPotential1D, GlobalPotential1D],
                              init=None, tries: int = 5):
    """
    Compute the equilibrium configuration of a crystal consisting of `n` ions (by numerical optimization).

    A typical use case for this function may look like this:
      z0 = get_equilibrium_positions(8, Quadratic1D())

    n        - The number of ions. The internal potential of the crystal is by default the corresponding
                 `PairwiseCoulomb1D`. Alternatively, some other `GlobalPotential1D` can be given explicitly.
    external - The axial trap potential, given as a `LocalPotential1D`. If the ions do not feel the same external
                 potential, some other `GlobalPotential1D` can be given explicitly.
    init     - Optional. Initialization for the numerical optimizer.
    tries    - Optional, default: 5. Number of attempts to numerically find an optimum.
                 Retries are conducted with randomly varied initial conditions.

    If no optimum is found after `tries` tries, a RuntimeError is raised.
    """
    # Let the first two arguments default to PairwiseCoulomb1D and GlobalFromLocal.
    if isinstance(n, GlobalPotential1D):
        internal = n
        n = internal.n
    else:
        internal = PairwiseCoulomb1D(n)
    if isinstance(external, LocalPotential1D):
        external = GlobalFromLocal(n, external)

    # Add up external and internal potential for the derivatives needed
    def total_gradient(z):
        return external.gradient(z) + internal.gradient(z)

    def total_hessian(z):
        return external.hessian(z) + internal.hessian(z)

    # Compute the default initialization for the optimizer, if needed
    if init is None:
        # This is a heuristic that works well for external quadratic potential and up to ~20 ions.
        outer = 2.16789415 * pow(n, 0.36617105) - 2.16584614  # approximate position of the ion with the greatest z
        init = np.linspace(-outer, outer, n)  # TODO: use non-linear spacing

    for i in range(tries):
        res = root(total_gradient, init, jac=total_hessian)
        if res.success:
            return res.x
        else:
            warnings.warn(f"Failed to find equilibrium positions on try {i+1}: {res.message}")
            init += 0.2 * (np.random.random_sample(n) - 0.5)  # TODO: change to relative variations
            init.sort()  # ensure ions are in order
    raise RuntimeError(f"Giving up to find equilibrium position after {tries} tries.")


def normal_mode_matrix(n: Union[int, GlobalPotential1D],
                       external: Union[LocalPotential1D, GlobalPotential1D],
                       init=None, tries: int = 5):
    """
    Compute the normal mode matrix of phonons in a crystal consisting of `n` ions (by numerical optimization).

    This is simply the Hessian of the total potential evaluated at the equilibrium configuration.
    A typical use case for this function may look like this:
      a0 = normal_mode_matrix(8, Quadratic1D())

    See `get_equilibrium_positions()` for the documentation of parameters.
    """

    if isinstance(n, PairwiseCoulomb1D):
        internal = n
        n = internal.n
    else:
        internal = PairwiseCoulomb1D(n)
    if isinstance(external, LocalPotential1D):
        external = GlobalFromLocal(n, external)

    def total_hessian(zs):
        return internal.hessian(zs) + external.hessian(zs)

    return total_hessian(get_equilibrium_positions(internal, external, init, tries))


def _delete_diag(j: np.ndarray):
    j[np.diag_indices_from(j)] = 0


def coupling_matrix(n: Union[int, GlobalPotential1D],
                    external: Union[LocalPotential1D, GlobalPotential1D],
                    delete_diagonal: bool = True,
                    init=None, tries: int = 5):
    """
    Compute the MAGIC coupling matrix of an `n` ion crystal (by numerical optimization).

    Up to a hardware-specific factor that is not included in this function, this is just the inverse
    of the normal mode matrix.
    A typical use case for this function may look like this:
      j = coupling_matrix(8, Quadratic1D())

    delete_diagonal - Optional, default: True. Whether to set the diagonal of the result to zero, as it is conventional
                                               for Ising couplings.
    See `get_equilibrium_positions()` for the documentation of the other parameters.
    """
    j = linalg.inv(normal_mode_matrix(n, external, init, tries))
    if delete_diagonal:
        _delete_diag(j)
    return j


# =====================
# === USAGE EXAMPLE ===
# =====================

if __name__ == '__main__':
    external = Quadratic1D()
    for n in [4, 5, 6]:
        print(f'=====\nJ({n}) =')
        print(coupling_matrix(n, external))
