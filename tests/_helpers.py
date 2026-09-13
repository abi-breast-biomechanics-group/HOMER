"""Mesh builders and tolerances shared across the suite.

HOMER evaluates in float32, so a "should be exact" comparison is exact only
to ~1e-6 relative.  The two tolerances below are the ones worth naming: use
:data:`EXACT` when a value is reproduced by construction (a round trip, an
affine map, a basis that can represent the target exactly) and :data:`CLOSE`
when it is reached by a fit or an iterative solve.
"""

import numpy as np


EXACT = 1e-5   #float32 round-off on quantities that are exact in exact arithmetic
CLOSE = 1e-3   #reached by least squares or Newton-Raphson, not by construction


def arr(x):
    """Writable float64 numpy view of a jax or numpy array."""
    return np.array(x, dtype=float)


def node_locs(mesh):
    """(n_nodes, 3) array of nodal positions."""
    return np.array([n.loc for n in mesh.nodes], dtype=float)
