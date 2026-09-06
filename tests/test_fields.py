"""Secondary fields carried on a mesh.

``create_mesh_field.py`` fitted a vector field and a scalar field to samples
taken off a sphere, drew arrows, and left it there.  What it was really
checking is that the fitted field reproduces the data it was fitted to, and
that the field stays registered to the geometry through refinement and
serialisation.
"""

import math

import numpy as np
import pytest

from HOMER import Mesh, MeshElement, MeshField, MeshNode
from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis, L3Basis
from HOMER.geometry import cube

from _helpers import CLOSE, EXACT, arr


def fibonacci_sphere(n, radius, centre):
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n)
    z = radius * (1.0 - 2.0 * i / (n - 1))
    r_xy = radius * np.sqrt(np.clip(1.0 - (z / radius) ** 2, 0, None))
    return np.vstack((r_xy * np.cos(golden_angle * i),
                      r_xy * np.sin(golden_angle * i), z)).T + centre


@pytest.fixture(scope="module")
def sampled_shells():
    """Points on three nested spheres inside the unit cube, with a radial field."""
    centre = np.array([0.5, 0.5, 0.5])
    points = np.concatenate([fibonacci_sphere(300, r, centre) for r in (0.49, 0.39, 0.29)])
    directions = points - centre
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return points, directions, points[:, -1].copy()


@pytest.fixture(scope="module")
def field_mesh(sampled_shells):
    points, directions, heights = sampled_shells
    mesh = cube(basis=[L1Basis] * 3, centre=np.array([0.5, 0.5, 0.5]))
    mesh.refine(2)
    mesh.new_field('vec_dir', field_dimension=3, field_locs=points,
                   field_values=directions, new_basis=[H3Basis] * 3)
    mesh.new_field('vec_mag', field_dimension=1, field_locs=points,
                   field_values=heights, new_basis=[L3Basis] * 3)
    return mesh


############################################### construction and access

def test_new_field_registers_under_its_name(field_mesh):
    assert set(field_mesh.fields) == {'vec_dir', 'vec_mag'}
    assert field_mesh['vec_dir'].fdim == 3
    assert field_mesh['vec_mag'].fdim == 1
    assert isinstance(field_mesh['vec_dir'], MeshField)


def test_a_field_shares_the_element_layout_of_its_mesh(field_mesh):
    """A field is sampled at the geometry's (element, xi), so the topology
    has to line up."""
    assert len(field_mesh['vec_dir'].elements) == len(field_mesh.elements)
    assert field_mesh['vec_dir'].ndim == field_mesh.ndim


def test_fitted_field_reproduces_the_data_it_was_fitted_to(field_mesh, sampled_shells):
    points, directions, heights = sampled_shells

    ele, xi = field_mesh.embed_points(points, iterations=20)
    vectors = arr(field_mesh['vec_dir'].evaluate_embeddings_ele_xi_pair(np.asarray(ele), np.asarray(xi)))
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    magnitudes = arr(field_mesh['vec_mag'].evaluate_embeddings_ele_xi_pair(
        np.asarray(ele), np.asarray(xi))).ravel()

    assert np.abs(np.sum(vectors * directions, axis=-1)).mean() > 0.999
    assert np.abs(magnitudes - heights).max() < 1e-2


def test_a_field_can_be_given_its_parameters_directly():
    mesh = cube(basis=[L1Basis] * 3)
    values = np.arange(8, dtype=float)

    mesh.new_field('index', field_dimension=1, field_params=values)

    corners = arr(mesh['index'].evaluate_embeddings(0, np.array(list(np.ndindex(2, 2, 2)),
                                                               dtype=float)))
    np.testing.assert_allclose(np.sort(corners.ravel()), np.sort(values), atol=EXACT)


def test_a_field_can_be_assigned_directly():
    mesh = cube(basis=[L1Basis] * 3)
    doubled = MeshField(nodes=[MeshNode(loc=node.loc * 2.0) for node in mesh.nodes],
                        elements=MeshElement(node_indexes=mesh.elements[0].nodes,
                                             basis_functions=mesh.elements[0].basis_functions))

    mesh['double'] = doubled

    np.testing.assert_allclose(
        arr(mesh['double'].evaluate_embeddings(0, np.array([[0.5, 0.5, 0.5]]))),
        arr(mesh.evaluate_embeddings(0, np.array([[0.5, 0.5, 0.5]]))) * 2.0, atol=EXACT)


############################################### staying registered

def test_refining_the_mesh_refines_its_fields():
    """The field must follow the geometry, or the two stop agreeing on what
    (element, xi) means -- and the value read at a physical point changes."""
    rng = np.random.default_rng(1)
    mesh = cube(basis=[L1Basis] * 3, centre=np.array([0.5, 0.5, 0.5]))
    mesh.new_field('linear', field_dimension=1,
                   field_params=np.array([n.loc[0] * 2 - n.loc[1] for n in mesh.nodes]))
    probes = rng.random((50, 3)) * 0.8 + 0.1

    def field_at(m, points):
        ele, xi = m.embed_points(points, iterations=20)
        return arr(m['linear'].evaluate_embeddings_ele_xi_pair(np.asarray(ele),
                                                               np.asarray(xi))).ravel()

    before = field_at(mesh, probes)
    mesh.refine(2)

    assert len(mesh['linear'].elements) == len(mesh.elements)
    assert len(mesh['linear'].nodes) == len(mesh.nodes)
    np.testing.assert_allclose(field_at(mesh, probes), before, atol=CLOSE)


def test_rebasing_the_mesh_keeps_its_fields():
    mesh = cube(basis=[L1Basis] * 3)
    mesh.new_field('index', field_dimension=1, field_params=np.arange(8, dtype=float))

    out = mesh.rebase([L2Basis] * 3)

    assert 'index' in out.fields


def test_a_field_value_is_recovered_at_the_point_it_was_placed():
    """The end-to-end contract: embed a point, read the field, get the value
    that was fitted there."""
    mesh = cube(basis=[L1Basis] * 3, centre=np.array([0.5, 0.5, 0.5]))
    mesh.refine(2)
    rng = np.random.default_rng(0)
    locs = rng.random((400, 3)) * 0.8 + 0.1
    values = locs[:, 0] * 2.0 - locs[:, 1]        #linear, so L1 can hold it exactly

    mesh.new_field('linear', field_dimension=1, field_locs=locs,
                   field_values=values, new_basis=[L1Basis] * 3)

    ele, xi = mesh.embed_points(locs, iterations=20)
    got = arr(mesh['linear'].evaluate_embeddings_ele_xi_pair(np.asarray(ele), np.asarray(xi))).ravel()

    np.testing.assert_allclose(got, values, atol=CLOSE)


def test_asking_for_a_field_that_is_not_there_raises():
    mesh = cube(basis=[L1Basis] * 3)

    with pytest.raises(KeyError):
        mesh['no_such_field']
