"""Element adjacency: crossing an element boundary and staying in the same place.

``test_mesh_topology.py`` asserted one hand-worked case.  The property behind
it is stronger and easy to state: for a mesh whose elements are affine, the
point reached by extrapolating past xi = 1 in one element is the same point
the neighbour reaches at the mapped coordinate.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import B3, H3, L1, L2
from HOMER.geometry import basic_surface, basic_surfaceMN, cube, cubeMNO
from HOMER.topomap_operations import refine_connectivity

from _helpers import EXACT, arr


@pytest.fixture(scope="module")
def block():
    mesh = cube(scale=1, centre=np.zeros(3), basis=[L1] * 3)
    mesh.refine(2)
    return mesh


def test_the_hand_worked_case(block):
    """Overshooting xi_3 by 0.1 lands 0.1 into the element above."""
    element, xi, valid = block.topomap(0, [0.5, 0.5, 1.1])

    assert valid
    assert int(element) == 4
    np.testing.assert_allclose(np.asarray(xi), [0.5, 0.5, 0.1], atol=1e-5)


def test_a_coordinate_inside_the_element_is_returned_unchanged(block):
    element, xi, valid = block.topomap(0, [0.5, 0.5, 0.5])

    assert valid and int(element) == 0
    np.testing.assert_allclose(np.asarray(xi), [0.5, 0.5, 0.5], atol=1e-6)


def test_stepping_off_the_mesh_is_reported_invalid(block):
    _, _, valid = block.topomap(0, [0.5, 0.5, -2.0])

    assert not valid


@pytest.mark.parametrize("mesh_factory", [
    lambda: cubeMNO([3, 3, 3], basis=[L1] * 3),
    lambda: basic_surfaceMN([3, 3], basis=[L1] * 2),
], ids=["volume", "surface"])
def test_topomap_preserves_the_physical_point(mesh_factory):
    """The invariant that makes topomap usable for marching across a mesh."""
    mesh = mesh_factory()
    rng = np.random.default_rng(0)
    checked = 0

    for _ in range(40):
        xi = rng.random(mesh.ndim)
        xi[rng.integers(0, mesh.ndim)] = 1.0 + rng.random() * 0.3
        source = int(rng.integers(0, len(mesh.elements)))

        element, mapped, valid = mesh.topomap(source, xi)
        if not valid:
            continue
        checked += 1

        #the elements are affine, so extrapolating past xi = 1 is exact
        np.testing.assert_allclose(arr(mesh.evaluate_embeddings(int(element),
                                                                np.asarray(mapped)[None])),
                                   arr(mesh.evaluate_embeddings(source, xi[None])),
                                   atol=1e-4)

    assert checked > 10


############################################### faces and boundaries

def test_get_faces_lists_every_exposed_element_face(block):
    faces = block.get_faces()

    assert len(faces) == 24                 #2x2x2 elements: 4 exposed faces per cube side
    assert all(len(f) == 3 for f in faces)  #(element, direction, side)
    assert {f[1] for f in faces} == {0, 1, 2}
    assert {f[2] for f in faces} == {0, 1}


def test_a_surface_mesh_has_no_exposed_volume_faces():
    assert basic_surface(basis=[L1] * 2).get_faces() == []


def test_get_xi_surface_nodes_selects_one_face_of_the_mesh(block):
    elements, nodes = block.get_xi_surface_nodes(2, 1)

    assert len(elements) == 4                #the four elements touching that face
    locs = np.array([block.nodes[n].loc for n in nodes], dtype=float)
    np.testing.assert_allclose(locs[:, 2], 0.5, atol=EXACT)


def test_hermite_surface_nodes_leave_the_far_node_behind():
    """A Hermite node carries derivative weights as well as a value weight, and
    all of them vanish at the far end of the direction - so the node at xi = 0
    is no more a part of the xi = 1 face than it is for a Lagrange basis."""
    mesh = cube(scale=1, centre=np.zeros(3), basis=[H3] * 3)
    mesh.refine(2)

    elements, nodes = mesh.get_xi_surface_nodes(2, 1)

    assert len(elements) == 4
    locs = np.array([mesh.nodes[n].loc for n in nodes], dtype=float)
    np.testing.assert_allclose(locs[:, 2], 0.5, atol=EXACT)


def test_control_net_surface_spans_every_layer_with_support():
    """A B-spline face is shaped by three layers of control points.

    ``B3(0) = [1/6, 4/6, 1/6, 0]``, so three of an element's four control
    layers have support on the xi = 0 face.  Picking the nodes out by their
    position in the element node list alone would return only the nearest
    layer, which does not determine the surface.
    """
    mesh = cube(scale=1, centre=np.zeros(3), basis=[L1] * 3)
    mesh.refine(2)
    mesh = mesh.rebase([B3] * 3)

    elements, nodes = mesh.get_xi_surface_nodes(2, 0)

    assert len(elements) == 4
    assert len(nodes) == 3 * 25            #three of the five 5x5 control layers


@pytest.mark.parametrize('basis', [[H3] * 3, [B3] * 3], ids=['H3', 'B3'])
def test_surface_nodes_are_the_nodes_that_move_the_surface(basis):
    """The contract, stated as geometry rather than as topology.

    Shifting a node the call returns has to be able to move the face; shifting
    any other node must leave it exactly where it was.
    """
    mesh = cube(scale=1, centre=np.zeros(3), basis=[L1] * 3)
    mesh.refine(2)
    mesh = mesh.rebase(basis)

    elements, nodes = mesh.get_xi_surface_nodes(2, 0)
    grid = np.linspace(0, 1, 4)
    xi = np.insert(np.stack(np.meshgrid(grid, grid, indexing='ij'), -1).reshape(-1, 2), 2, 0, axis=1)
    face = lambda: arr(mesh.evaluate_embeddings(elements, xi))

    before = face()
    moves_the_face = set()
    for n in range(len(mesh.nodes)):
        mesh.nodes[n].loc = mesh.nodes[n].loc + np.array([0., 0., 1.])
        mesh.generate_mesh()
        if np.abs(face() - before).max() > 0:
            moves_the_face.add(n)
        mesh.nodes[n].loc = mesh.nodes[n].loc - np.array([0., 0., 1.])
        mesh.generate_mesh()

    assert set(int(n) for n in nodes) == moves_the_face


############################################### connectivity refinement

def test_refine_connectivity_produces_one_entry_per_child_element(block):
    lookup, parents = refine_connectivity(block._topo_lookup, [2, 2, 2])

    assert lookup.shape == (len(block.elements) * 8, 3, 2)
    assert parents.shape == (len(block.elements) * 8, 3)
    assert parents.min() == 0 and parents.max() == 1


def test_refined_connectivity_is_symmetric(block):
    """If A says B is its neighbour on a face, B must say the same in reverse."""
    lookup, _ = refine_connectivity(block._topo_lookup, [2, 2, 2])

    for element in range(lookup.shape[0]):
        for axis in range(3):
            for side in range(2):
                neighbour = int(lookup[element, axis, side])
                if neighbour < 0:
                    continue
                assert int(lookup[neighbour, axis, 1 - side]) == element
