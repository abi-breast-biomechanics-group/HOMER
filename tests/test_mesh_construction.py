"""Building meshes: bases, mixed bases, collapsed elements, ids, editing.

Replaces the ``create_mesh_*_test.py`` and ``index_mesh_*`` scripts, which
built a mesh, drew it, and left the verdict to the reader.  What those
scripts were really checking is that the element interpolates its own nodes
and that the mesh has the size and volume it should -- both of which are
statable.
"""

import copy
import itertools
import logging
from dataclasses import replace

import numpy as np
import pytest

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import (B3Basis, BasisGroup, H3Basis, L1Basis, L2Basis,
                                     L3Basis, L4Basis)
from HOMER.mesh import GAUSS, volume_quadrature_order

from _helpers import EXACT, arr, bulged_patch, hermite_cube, node_locs, unit_hex

CORNERS_3D = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
CORNERS_2D = np.array(list(itertools.product([0.0, 1.0], repeat=2)))


def reference_volume(mesh, n=24):
    """Volume by a tensor Gauss-Legendre rule built outside HOMER entirely.

    24 points per direction is exact well past any basis here, so this is the
    answer ``get_volume`` is trying to reach.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), -1).reshape(-1, 3)
    weights = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()

    total = 0.0
    for element in range(len(mesh.elements)):
        jac = arr(mesh.evaluate_jacobians_ele_xi_pair(np.full(len(grid), element), grid))
        total += float((np.linalg.det(jac) * weights).sum())
    return total


def distorted_hex(basis):
    """A hexahedron with every corner jittered, so det(J) is genuinely varying."""
    rng = np.random.default_rng(3)
    locs = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
    locs = locs + rng.uniform(-0.25, 0.25, locs.shape)
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(L1Basis, L1Basis, L1Basis))
    mesh = Mesh(nodes=[MeshNode(loc=l) for l in locs], elements=element)
    return mesh if basis is L1Basis else mesh.rebase([basis] * 3)


############################################### single elements, every basis

@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis],
                         ids=lambda b: b.__name__)
def test_volume_element_interpolates_its_corner_nodes(basis):
    """Whatever the basis, the eight xi-corners must land on the eight nodes."""
    mesh = unit_hex(basis=[basis] * 3)

    corners = arr(mesh.evaluate_embeddings(0, CORNERS_3D))

    #these meshes are built by rebasing, so the corners come out of a float32
    #least-squares solve; the quartic one has 125 nodes and sits right on the
    #round-off floor for a system that size
    atol = 2e-5 if basis is L4Basis else EXACT
    np.testing.assert_allclose(np.sort(corners, axis=0),
                               np.sort(np.array(list(itertools.product([0., 1.], repeat=3))), axis=0),
                               atol=atol)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis],
                         ids=lambda b: b.__name__)
def test_unit_cube_has_unit_volume_in_every_basis(basis):
    assert unit_hex(basis=[basis] * 3).get_volume() == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("basis", [(H3Basis, L1Basis, H3Basis),
                                   (L2Basis, H3Basis, L1Basis),
                                   (L3Basis, L1Basis, L2Basis)],
                         ids=lambda b: "".join(x.__name__[:2] for x in b))
def test_mixed_bases_build_and_keep_the_geometry(basis):
    """Each parametric direction may carry its own basis."""
    mesh = unit_hex(basis=basis)

    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)
    np.testing.assert_allclose(np.sort(arr(mesh.evaluate_embeddings(0, CORNERS_3D)), axis=0),
                               np.sort(CORNERS_3D, axis=0), atol=EXACT)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, H3Basis],
                         ids=lambda b: b.__name__)
def test_surface_element_interpolates_its_corner_nodes(basis):
    from HOMER.geometry import basic_surface

    mesh = basic_surface(basis=[basis] * 2)

    corners = arr(mesh.evaluate_embeddings(0, CORNERS_2D))
    np.testing.assert_allclose(np.sort(corners, axis=0),
                               np.sort(node_locs(basic_surface(basis=[L1Basis] * 2)), axis=0),
                               atol=EXACT)


def test_bulged_patch_is_actually_curved():
    """The recurring fitting target: the element centre is pulled off the plane."""
    patch = bulged_patch()

    np.testing.assert_allclose(arr(patch.evaluate_embeddings(0, np.array([[0.5, 0.5]])))[0],
                               [0.5, 0.5, 0.5], atol=EXACT)
    #every corner is on x = 0, so a plane through them could not reach x = 0.5
    np.testing.assert_allclose(arr(patch.evaluate_embeddings(0, CORNERS_2D))[:, 0], 0.0, atol=EXACT)


############################################### element requirements

def test_hermite_basis_without_nodal_derivatives_is_rejected():
    """The clearest failure mode when hand-building a Hermite mesh."""
    nodes = [MeshNode(loc=np.array(l, dtype=float)) for l in CORNERS_3D]
    element = MeshElement(node_indexes=list(range(8)), basis_functions=(H3Basis,) * 3)

    with pytest.raises(ValueError, match="did not have the required field"):
        Mesh(nodes=nodes, elements=element)


def test_collapsed_element_maps_several_corners_onto_one_node():
    """A degenerate hexahedron: four local slots share a single node.

    ``create_collapsed_mesh_H3H3H3_test.py`` drew this and asked whether it
    looked collapsed.  It is collapsed exactly when those corners evaluate to
    the same point, and when the element still has positive volume.
    """
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0]]
    tangents = [[2, -.5, .5], [0, 0, 0], [0, 0, 0], [2, .5, -.5], [1, -.5, .5], [1, -.5, -.5]]
    zero = np.zeros(3)
    nodes = [MeshNode(loc=np.array(l, dtype=float), du=zero, dv=zero,
                      dw=np.array(t, dtype=float), dudv=zero, dudw=zero, dvdw=zero, dudvdw=zero)
             for l, t in zip(locs, tangents)]
    element = MeshElement(node_indexes=[0, 1, 2, 3, 5, 5, 5, 5], basis_functions=(H3Basis,) * 3)

    mesh = Mesh(nodes=nodes, elements=element)

    corners = arr(mesh.evaluate_embeddings(0, CORNERS_3D))
    collapsed = np.isclose(corners, mesh.nodes[5].loc, atol=EXACT).all(-1)
    assert collapsed.sum() == 4
    assert mesh.get_volume() > 0


############################################### quadrature

@pytest.mark.parametrize("ng", [[2, 2, 2], [3, 3, 3], [4, 4, 4], [6, 4, 3]])
def test_gauss_grid_weights_are_a_partition_of_the_reference_cell(ng):
    grid, weights = unit_hex().gauss_grid(ng)

    assert grid.shape == (int(np.prod(ng)), 3)
    assert arr(weights).sum() == pytest.approx(1.0)
    assert grid.min() > 0 and grid.max() < 1        #open rule: no points on the faces


def test_gauss_quadrature_reproduces_the_volume_of_an_affine_element():
    """Independent of ``get_volume``: sum |det J| w over the Gauss points."""
    mesh = unit_hex()
    grid, weights = mesh.gauss_grid([3, 3, 3])

    jac = arr(mesh.evaluate_jacobians_ele_xi_pair(np.zeros(len(grid), int), grid))
    quadrature = (np.abs(np.linalg.det(jac)) * arr(weights)).sum()

    assert quadrature == pytest.approx(1.0, abs=EXACT)
    assert quadrature == pytest.approx(mesh.get_volume(), abs=EXACT)


def test_get_volume_is_exact_on_a_curved_element():
    """Checked against an independent, far higher-order quadrature.

    ``get_volume`` picks its Gauss order from
    :func:`~HOMER.mesher.volume_quadrature_order` so that det(J) is integrated
    exactly.  The reference below is a 24-point tensor Gauss-Legendre rule
    built straight from numpy -- nothing in HOMER decides it -- so agreement
    is evidence about the rule, not a restatement of it.
    """
    mesh = hermite_cube()

    assert mesh.get_volume() == pytest.approx(reference_volume(mesh), rel=1e-5)
    assert mesh.get_volume() < 1.0                  #the dw tangents pinch the element


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis],
                         ids=lambda b: b.__name__)
def test_get_volume_is_exact_on_a_distorted_hexahedron(basis):
    """The case the old basis-order rule got wrong: a non-affine element.

    At one Gauss point per direction the trilinear version of this mesh came
    out 1% off.
    """
    mesh = distorted_hex(basis)

    assert mesh.get_volume() == pytest.approx(reference_volume(mesh), rel=1e-4)


def test_volume_quadrature_order_follows_the_degree_of_det_J():
    """n = ceil(3p/2), per direction, and mixed bases are handled per direction."""
    assert volume_quadrature_order([L1Basis] * 3) == [2, 2, 2]
    assert volume_quadrature_order([L2Basis] * 3) == [3, 3, 3]
    assert volume_quadrature_order([H3Basis] * 3) == [5, 5, 5]
    assert volume_quadrature_order([L4Basis] * 3) == [6, 6, 6]
    assert volume_quadrature_order([L1Basis, H3Basis, L4Basis]) == [2, 5, 6]


def test_volume_quadrature_order_warns_rather_than_failing_on_an_exotic_basis(caplog):
    """A degree past the tabulated rules is clamped, loudly."""
    degree7 = replace(L4Basis, name='Degree7Basis', order=7)

    with caplog.at_level(logging.WARNING):
        orders = volume_quadrature_order(degree7 * 3)

    assert orders == [max(GAUSS)] * 3
    assert 'under-integrated' in caplog.text


def test_an_untabulated_gauss_rule_is_reported_clearly():
    with pytest.raises(ValueError, match="No 9-point Gauss rule is tabulated"):
        unit_hex().gauss_grid(9)


def test_get_volume_rejects_a_surface_mesh():
    from HOMER.geometry import basic_surface

    with pytest.raises(ValueError, match="only defined on a 3-D mesh"):
        basic_surface(basis=[L1Basis] * 2).get_volume()


############################################### identifiers

def test_elements_may_reference_nodes_by_id():
    """Ids can be any hashable: str, int, or tuple."""
    zero = np.zeros(3)
    ids = ['node_1', 2, '3', (1, 1)]
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0]]
    nodes = [MeshNode(loc=np.array(l, dtype=float), du=zero, dv=zero, dudv=zero, id=i)
             for l, i in zip(locs, ids)]
    element = MeshElement(node_ids=ids, basis_functions=(H3Basis, H3Basis), id='test_elem')

    mesh = Mesh(nodes=nodes, elements=element)

    assert mesh.node_id_to_ind == {'node_1': 0, 2: 1, '3': 2, (1, 1): 3}
    assert mesh.element_id_to_ind == {'test_elem': 0}
    np.testing.assert_allclose(mesh.get_node('node_1').loc, [0, 0, 1], atol=EXACT)
    assert mesh.get_element(['test_elem'])[0] is mesh.elements[0]


def test_associated_node_index_locates_named_fields_in_the_param_vector():
    """Answers "which entries of the parameter array are node 3's du?"."""
    mesh = unit_hex(basis=[H3Basis] * 3)
    per_node = len(mesh.true_param_array) // len(mesh.nodes)

    loc_inds = arr(mesh.associated_node_index(['loc'])).reshape(len(mesh.nodes), 3)
    du_inds = arr(mesh.associated_node_index(['du'])).reshape(len(mesh.nodes), 3)

    np.testing.assert_array_equal(loc_inds[0], [0, 1, 2])
    np.testing.assert_array_equal(du_inds[0], [3, 4, 5])
    np.testing.assert_array_equal(loc_inds[:, 0], np.arange(len(mesh.nodes)) * per_node)


def test_associated_node_index_leaves_the_mesh_unchanged():
    """It walks the parameter vector by overwriting it, so it must put it back."""
    mesh = unit_hex(basis=[H3Basis] * 3)
    before = arr(mesh.true_param_array)

    mesh.associated_node_index(['loc'], nodes_to_gather=[0, 2])

    np.testing.assert_allclose(arr(mesh.true_param_array), before, atol=EXACT)


############################################### editing

def test_transform_moves_every_node_and_preserves_volume():
    mesh = unit_hex()
    tform = np.eye(4)
    tform[:3, :3] = np.diag([1.0, 1.0, 1.0])
    tform[:3, 3] = [1.0, 2.0, 3.0]
    before = node_locs(mesh)

    mesh.transform(tform)

    np.testing.assert_allclose(node_locs(mesh), before + [1, 2, 3], atol=EXACT)
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)


def test_adding_two_meshes_concatenates_them():
    from HOMER.geometry import cube

    left = cube(basis=[L1Basis] * 3)
    right = cube(basis=[L1Basis] * 3, centre=np.array([1.0, 0.0, 0.0]))

    joined = left + right

    assert len(joined.elements) == 2
    assert len(joined.nodes) == len(left.nodes) + len(right.nodes)
    assert joined.get_volume() == pytest.approx(2.0, abs=EXACT)


def test_drop_elements_removes_the_element_and_its_orphaned_nodes():
    from HOMER.geometry import cubeMNO

    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    before_volume = mesh.get_volume()

    mesh.drop_elements([0])

    assert len(mesh.elements) == 7
    assert len(mesh.nodes) == 26                    #the far corner of the dropped element
    assert mesh.get_volume() == pytest.approx(before_volume * 7 / 8, rel=1e-4)


def test_update_from_params_round_trips():
    mesh = unit_hex()
    original = arr(mesh.optimisable_param_array)

    mesh.update_from_params(original * 2.0)
    assert mesh.get_volume() == pytest.approx(8.0, abs=EXACT)

    mesh.update_from_params(original)
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)


def test_fixing_a_parameter_shrinks_the_optimisable_vector():
    mesh = unit_hex()
    full = len(mesh.optimisable_param_array)

    mesh.nodes[0].fix_parameter('loc', inds=[2])
    mesh.generate_mesh()
    assert len(mesh.optimisable_param_array) == full - 1

    mesh.unfix_mesh()
    assert len(mesh.optimisable_param_array) == full
    assert mesh.optimisable_param_bool.all()


def test_get_element_params_gathers_the_element_dofs():
    mesh = unit_hex(basis=[L2Basis] * 3)

    params = arr(mesh.get_element_params(0))

    assert params.size == len(mesh.elements[0].nodes) * mesh.fdim


def test_an_element_accepts_every_spelling_of_its_basis_group():
    """A group, a list, a tuple and a bare basis all mean the same element."""
    spellings = [L1Basis * 3,
                 [L1Basis, L1Basis, L1Basis],
                 (L1Basis, L1Basis, L1Basis)]
    for basis in spellings:
        element = MeshElement(node_indexes=list(range(8)), basis_functions=basis)
        assert element.basis_functions == L1Basis * 3
        assert element.ndim == 3

    line = MeshElement(node_indexes=[0, 1], basis_functions=L1Basis)
    assert line.basis_functions == BasisGroup([L1Basis]) and line.ndim == 1


def test_an_element_rejects_a_dimensionality_it_cannot_build():
    """Four directions used to fail deep inside the basis-product indices."""
    with pytest.raises(ValueError, match="1, 2 or 3 bases"):
        MeshElement(node_indexes=list(range(16)), basis_functions=L1Basis * 4)
