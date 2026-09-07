"""
geometry.py – Convenience factory functions for standard mesh geometries.

Currently provides:

* :func:`cube` – create a unit cube (or scaled/translated cube) mesh.
"""

from typing import Optional
from HOMER.mesh import MeshNode, MeshElement, Mesh
from HOMER.mesh.reordering import reorder_nodes
from HOMER.basis_definitions import H3Basis, L1Basis

import numpy as np

def cube(scale: float = 1, centre: Optional[np.ndarray]=None, basis=None) -> Mesh:
    """Create a single-element cube mesh.

    Constructs a mesh with 8 corner nodes and a single trilinear element
    (``L1Basis * 3``), then :meth:`~HOMER.mesh.field.MeshField.rebase`-s
    it to the requested *basis* (defaulting to cubic Hermite in all directions).

    Parameters
    ----------
    scale:
        Side length of the cube.  The default is a unit cube.
    centre:
        Centre of the cube, shape ``(3,)``.  Defaults to the origin.
    basis:
        The three 1-D bases for the resulting mesh, e.g. ``H3Basis * 3``.
        Defaults to ``H3Basis * 3``.

    Returns
    -------
    Mesh
        A :class:`~HOMER.mesh.mesh.Mesh` with the requested basis.
    """
    if centre is None:
        centre = np.zeros(3)
    if scale is None:
        scale = 1
    if basis is None:
        basis = H3Basis * 3
    bottom_corner = centre - scale/2
    point0 = MeshNode(loc= bottom_corner + scale *np.array([0,0,0]))
    point1 = MeshNode(loc= bottom_corner + scale *np.array([1,0,0]))
    point2 = MeshNode(loc= bottom_corner + scale *np.array([0,1,0]))
    point3 = MeshNode(loc= bottom_corner + scale *np.array([1,1,0]))
    point4 = MeshNode(loc= bottom_corner + scale *np.array([0,0,1]))
    point5 = MeshNode(loc= bottom_corner + scale *np.array([1,0,1]))
    point6 = MeshNode(loc= bottom_corner + scale *np.array([0,1,1]))
    point7 = MeshNode(loc= bottom_corner + scale *np.array([1,1,1]))
    element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=L1Basis * 3)
    mesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase(basis)
    return mesh

def cubeMNO(res, basis=None, loc=None, scale=None):
    """
    Creates a multi-element unit cube mesh.
    Then re-orders the cube to have sane elements.

    The ordering asked for is the ``'spatial'`` one, lexicographic in
    ``(z, y, x)``.  The refinement's own ``'lattice'`` ordering agrees with it
    for a cube, but sorting the coordinates is what this function has always
    promised, so it is what it still asks for.
    """
    base_cube = cube(basis=basis, centre=loc, scale=scale) 
    base_cube.refine(by_xi_refinement=[np.linspace(0, 1, r+1) for r in res],
                     reorder_nodes=False)
    reorder_nodes(base_cube, 'spatial')
    return base_cube

def basic_surface(corner_locs=None, basis=None):

    if corner_locs is None:
        corner_locs = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1]])

    if basis is None:
        basis = L1Basis * 2

    point0 = MeshNode(loc=corner_locs[0])
    point1 = MeshNode(loc=corner_locs[1])
    point2 = MeshNode(loc=corner_locs[2])
    point3 = MeshNode(loc=corner_locs[3])

    element1 = MeshElement(node_indexes=[0,1,2,3], basis_functions=L1Basis * 2)
    mesh = Mesh(nodes = [point0, point1, point2, point3], elements = element1).rebase(basis)

    return mesh

def basic_surfaceMN(res, basis=None):
    """
    Creates a multi-element surface mesh.
    Then re-orders the cube to have sane elements.

    As with :func:`cubeMNO` the ordering is the ``'spatial'`` one,
    lexicographic in ``(z, y, x)``.
    """
    surf = basic_surface(basis=basis) 
    surf.refine(by_xi_refinement=[np.linspace(0, 1, r+1) for r in res],
                reorder_nodes=False)
    reorder_nodes(surf, 'spatial')
    return surf
