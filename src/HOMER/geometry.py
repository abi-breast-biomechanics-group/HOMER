"""
geometry.py – Convenience factory functions for standard mesh geometries.

Currently provides:

* :func:`cube` – create a unit cube (or scaled/translated cube) mesh.
"""

from typing import Optional
from HOMER.mesher import MeshNode, MeshElement, Mesh
from HOMER.basis_definitions import H3Basis, L1Basis

import numpy as np

def cube(scale: float = 1, centre: Optional[np.ndarray]=None, basis=None) -> Mesh:
    """Create a single-element cube mesh.

    Constructs a mesh with 8 corner nodes and a single trilinear element
    (``L1Basis`` × ``L1Basis`` × ``L1Basis``), then :meth:`~HOMER.mesher.MeshField.rebase`-s
    it to the requested *basis* (defaulting to cubic Hermite in all directions).

    Parameters
    ----------
    scale:
        Side length of the cube.  The default is a unit cube.
    centre:
        Centre of the cube, shape ``(3,)``.  Defaults to the origin.
    basis:
        Sequence of three 1-D basis classes for the resulting mesh.
        Defaults to ``[H3Basis, H3Basis, H3Basis]``.

    Returns
    -------
    Mesh
        A :class:`~HOMER.mesher.Mesh` with the requested basis.
    """
    if centre is None:
        centre = np.zeros(3)
    if basis is None:
        basis = [H3Basis] * 3
    bottom_corner = centre - scale/2
    point0 = MeshNode(loc= bottom_corner + scale *np.array([0,0,0]))
    point1 = MeshNode(loc= bottom_corner + scale *np.array([1,0,0]))
    point2 = MeshNode(loc= bottom_corner + scale *np.array([0,1,0]))
    point3 = MeshNode(loc= bottom_corner + scale *np.array([1,1,0]))
    point4 = MeshNode(loc= bottom_corner + scale *np.array([0,0,1]))
    point5 = MeshNode(loc= bottom_corner + scale *np.array([1,0,1]))
    point6 = MeshNode(loc= bottom_corner + scale *np.array([0,1,1]))
    point7 = MeshNode(loc= bottom_corner + scale *np.array([1,1,1]))
    element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(L1Basis, L1Basis, L1Basis))
    mesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase(basis)
    return mesh
