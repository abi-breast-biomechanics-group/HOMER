"""Example meshes for the documentation and the test suite.

These are the meshes HOMER is illustrated with: a curved patch that no plane
can represent, a cube whose edges bow outwards, the axis-aligned unit hex the
other two are compared against, and the library's own name written as a mesh.
They live here, rather than beside either the docs or the tests, so that a page
showing one and a test asserting on it cannot drift apart.
"""

import numpy as np

from HOMER.basis_definitions import H3, L1, L2
from HOMER.mesh import Mesh, MeshElement, MeshNode

__all__ = ["bulged_patch", "hermite_cube", "unit_hex", "wordmark"]


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
    element = MeshElement(node_indexes=list(range(9)), basis_functions=(L2, L2))
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
                          basis_functions=(H3, H3, H3))
    return Mesh(nodes=nodes, elements=element)


def unit_hex(basis=None):
    """Axis-aligned unit cube on [0, 1]^3, node order matching :func:`hermite_cube`.

    Built trilinear and rebased, so a *basis* needing nodal derivatives gets
    them from the rebase fit rather than from hand-written zeros.
    """
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0],
            [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(L1, L1, L1))
    mesh = Mesh(nodes=[MeshNode(loc=np.array(l, dtype=float)) for l in locs],
                elements=element)
    return mesh if basis is None else mesh.rebase(tuple(basis))


#One glyph cell is a quadrilateral in the plane, extruded this far in z.
_DEPTH = 0.3
#Rows are numbered downwards, the way text is, so the layout below reads like
#the word it draws.  _add_cell mirrors them into world space.
_ROWS = 3

#Corners in element slot order: the xi_2 = 0 edge, then the xi_2 = 1 edge.
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
#Two of the five letters need a cell that is not a square.  A repeated corner
#collapses that edge, which is how the middle stroke of M and of E is a wedge.
_PEAK = ((0, 0), (1, 0), (0.5, 1), (0.5, 1))
_POINT = ((0, 0), (0, 1), (1, 0.5), (1, 0.5))
#The R's bowl and leg are sheared cells, and the tangents bow their edges.
_SLANT = ((0, 0), (1, 1), (1, 0), (2, 1))
_SLANT_BACK = ((1, 0), (0, 1), (2, 0), (1, 1))
_BOW = ((0, 2, 0), (0, 2, 0), (0.5, 0, 0), (0, 2, 0))
_STRAIGHT = ((0, 2, 0),) * 4

#(cell shape, column, row, edge tangents).  Columns are spaced to leave a gap
#between letters; every letter is three rows tall.
_GLYPHS = (
    #H
    *[(_SQUARE, x, y, None) for x, y in
      ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2))],
    #O
    *[(_SQUARE, x, y, None) for x, y in
      ((3.5, 0), (4.5, 0), (5.5, 0), (3.5, 1), (5.5, 1), (3.5, 2), (4.5, 2), (5.5, 2))],
    #M
    *[(_SQUARE, x, y, None) for x, y in
      ((7, 0), (8, 0), (9, 0), (7, 1), (9, 1), (7, 2), (9, 2))],
    (_PEAK, 8, 1, None),
    #E
    *[(_SQUARE, x, y, None) for x, y in
      ((10.5, 0), (11.5, 0), (12.5, 0), (10.5, 1), (10.5, 2), (11.5, 2), (12.5, 2))],
    (_POINT, 11.5, 1, None),
    #R
    *[(_SQUARE, 14, y, None) for y in (0, 1, 2)],
    (_SLANT, 15, 0, _BOW),
    (_SLANT_BACK, 15, 1, _STRAIGHT),
    (_SLANT, 15, 2, _BOW),
)


def _add_cell(nodes, elements, quad, origin, tangents):
    """Extrude one planar quad into a tricubic-Hermite element.

    The row mirror that turns a text layout into world space also mirrors the
    element, so the extrusion runs the other way to keep the Jacobian positive.
    """
    zero = np.zeros(3)
    slot = {}
    for corner, tangent in zip(quad, tangents or ((0, 0, 0),) * 4):
        if corner in slot:
            continue
        slot[corner] = len(nodes)
        x = corner[0] + origin[0]
        y = _ROWS - (corner[1] + origin[1])
        #the tangent is mirrored with the geometry it is a tangent to
        dv = np.array([tangent[0], -tangent[1], tangent[2]], dtype=float)
        #the two nodes of a corner are consecutive, so a slot pair is (i, i + 1)
        nodes.extend(
            MeshNode(loc=np.array([x, y, z], dtype=float),
                     du=zero, dv=dv, dw=zero,
                     dudv=zero, dudw=zero, dvdw=zero, dudvdw=zero)
            for z in (0.0, _DEPTH)
        )
    elements.append(MeshElement(
        node_indexes=[slot[corner] + k for corner in quad for k in (0, 1)],
        basis_functions=H3**3))


def wordmark():
    """The word HOMER, as a mesh.

    Thirty-seven tricubic-Hermite elements: mostly extruded squares, two
    collapsed to wedges where a letter needs a diagonal, and three sheared and
    bowed into the R.  Lying in the z = 0 plane, so a camera looking down z --
    :meth:`pyvista.Plotter.view_xy`, with parallel projection so the letters
    are not skewed -- reads it as text before anyone turns it.
    """
    nodes, elements = [], []
    for quad, x, y, tangents in _GLYPHS:
        _add_cell(nodes, elements, quad, (x, y), tangents)
    return Mesh(nodes=nodes, elements=elements)
