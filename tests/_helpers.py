"""Mesh builders and tolerances shared across the suite.

HOMER evaluates in float32, so a "should be exact" comparison is exact only
to ~1e-6 relative.  The two tolerances below are the ones worth naming: use
:data:`EXACT` when a value is reproduced by construction (a round trip, an
affine map, a basis that can represent the target exactly) and :data:`CLOSE`
when it is reached by a fit or an iterative solve.
"""

import numpy as np

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis

EXACT = 1e-5   #float32 round-off on quantities that are exact in exact arithmetic
CLOSE = 1e-3   #reached by least squares or Newton-Raphson, not by construction


def arr(x):
    """Writable float64 numpy view of a jax or numpy array."""
    return np.array(x, dtype=float)


def bulged_patch(jax_compile=False):
    """The 9-node biquadratic patch that most of the old scripts fitted to.

    Three corners sit on the x = 0 plane and the element centre is pulled out
    to (0.5, 0.5, 0.5), so the patch is curved -- a plane cannot represent it
    and a point projected onto it has a residual worth checking.
    """
    locs = [
        [0, 0, 1], [0, 0, 0.5], [0, 0, 0],
        [0, 0.5, 1], [0.5, 0.5, 0.5], [0, 0.5, 0],
        [0, 1, 1], [0, 1, 0.5], [0, 1, 0],
    ]
    element = MeshElement(node_indexes=list(range(9)), basis_functions=(L2Basis, L2Basis))
    return Mesh(nodes=[MeshNode(loc=l) for l in locs], elements=element,
                jax_compile=jax_compile)


def hermite_cube():
    """Unit cube with non-zero third-direction Hermite derivatives.

    The ``dw`` tangents bow the w-direction edges outwards, so the element is
    genuinely curved: volume, Jacobians and embeddings all differ from the
    trilinear cube built on the same corners.
    """
    corners = [
        ([0, 0, 1], [2, -0.5, 0.5]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 1, 1], [0, 0, 0]),
        ([0, 1, 0], [2, 0.5, -0.5]),
        ([1, 0, 1], [1, -0.5, 0.5]),
        ([1, 0, 0], [1, -0.5, -0.5]),
        ([1, 1, 1], [1, 0.5, 0.5]),
        ([1, 1, 0], [1, 0.5, -0.5]),
    ]
    zero = np.zeros(3)
    nodes = [MeshNode(loc=np.array(loc, dtype=float), du=zero, dv=zero, dw=np.array(dw, dtype=float),
                      dudv=zero, dudw=zero, dvdw=zero, dudvdw=zero)
             for loc, dw in corners]
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(H3Basis, H3Basis, H3Basis))
    return Mesh(nodes=nodes, elements=element)


def unit_hex(basis=None):
    """Axis-aligned unit cube on [0, 1]^3, node order matching :func:`hermite_cube`.

    Built trilinear and rebased, so a *basis* needing nodal derivatives gets
    them from the rebase fit rather than from hand-written zeros.
    """
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0],
            [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(L1Basis, L1Basis, L1Basis))
    mesh = Mesh(nodes=[MeshNode(loc=np.array(l, dtype=float)) for l in locs],
                elements=element)
    return mesh if basis is None else mesh.rebase(tuple(basis))


def node_locs(mesh):
    """(n_nodes, 3) array of nodal positions."""
    return np.array([n.loc for n in mesh.nodes], dtype=float)
