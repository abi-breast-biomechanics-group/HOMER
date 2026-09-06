"""Evaluating a mesh: embeddings, derivatives, normals, quadrature grids.

Three spellings of every evaluator coexist -- the "wide" form (every element
at every xi), the ``_ele_xi_pair`` form (one xi per element) and the
``_in_every_element`` form -- plus a chunking/rematerialisation layer in
:mod:`HOMER.mesh_decorators` that none of the old scripts touched.  They must
all agree, and the derivative evaluators must agree with autodiff.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis, L3Basis
from HOMER.geometry import basic_surface, cube

from _helpers import EXACT, arr, hermite_cube, unit_hex


@pytest.fixture(scope="module")
def refined_cube():
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(2)
    return mesh


############################################### the three evaluator spellings

def test_wide_evaluation_is_the_cross_product_of_elements_and_xis(refined_cube):
    xi = refined_cube.xi_grid(3)
    eles = np.array([0, 1, 2])

    wide = arr(refined_cube.evaluate_embeddings(eles, xi))
    pair = arr(refined_cube.evaluate_embeddings_ele_xi_pair(
        np.repeat(eles, len(xi)), np.tile(xi, (len(eles), 1))))

    assert wide.shape == (len(eles) * len(xi), 3)
    np.testing.assert_array_equal(wide, pair)


def test_in_every_element_covers_all_elements(refined_cube):
    xi = refined_cube.xi_grid(3)

    everywhere = arr(refined_cube.evaluate_embeddings_in_every_element(xi))
    wide = arr(refined_cube.evaluate_embeddings(np.arange(len(refined_cube.elements)), xi))

    assert everywhere.shape == (len(refined_cube.elements) * len(xi), 3)
    np.testing.assert_array_equal(everywhere, wide)


@pytest.mark.parametrize("chunk_size", [1, 4, 10_000])
def test_chunking_does_not_change_the_answer(refined_cube, chunk_size):
    """The chunked scan in ``mesh_decorators`` exists only to bound memory."""
    xi = refined_cube.xi_grid(4)

    reference = arr(refined_cube.evaluate_embeddings_in_every_element(xi))
    chunked = arr(refined_cube.evaluate_embeddings_in_every_element(xi, chunk_size=chunk_size))

    #a chunk boundary reassociates the float32 sums, so this is not bit-identical
    np.testing.assert_allclose(chunked, reference, atol=EXACT)


def test_rematerialisation_does_not_change_the_answer(refined_cube):
    xi = refined_cube.xi_grid(4)

    reference = arr(refined_cube.evaluate_embeddings_in_every_element(xi))
    remat = arr(refined_cube.evaluate_embeddings_in_every_element(xi, remat=True))

    np.testing.assert_array_equal(remat, reference)


def test_evaluate_element_embeddings_looks_the_element_up_by_id():
    """The only evaluator that takes a user id rather than a list index."""
    from HOMER import Mesh, MeshElement, MeshNode

    locs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    nodes = [MeshNode(loc=np.array(l, dtype=float)) for l in locs]
    element = MeshElement(node_indexes=[0, 1, 2, 3],
                          basis_functions=(L1Basis, L1Basis), id='patch')
    mesh = Mesh(nodes=nodes, elements=element)
    xi = mesh.xi_grid(3)

    by_id = arr(mesh.evaluate_element_embeddings('patch', xi))
    by_index = arr(mesh.evaluate_embeddings(np.array([0]), xi))

    np.testing.assert_allclose(by_id.reshape(-1, 3), by_index, atol=EXACT)
    with pytest.raises(KeyError):
        mesh.evaluate_element_embeddings('no_such_element', xi)


def test_fit_params_override_the_stored_parameters():
    mesh = unit_hex()
    doubled = arr(mesh.optimisable_param_array) * 2.0
    xi = np.array([[0.5, 0.5, 0.5]])

    with_override = arr(mesh.evaluate_embeddings_ele_xi_pair(np.array([0]), xi, fit_params=doubled))

    #the override must not be written back
    np.testing.assert_allclose(arr(mesh.evaluate_embeddings_ele_xi_pair(np.array([0]), xi)),
                               with_override / 2.0, atol=EXACT)


############################################### derivatives

def test_jacobians_match_autodiff_of_the_embedding():
    mesh = hermite_cube()

    def embed(xi):
        return mesh.evaluate_embeddings_ele_xi_pair(jnp.array([0]), xi[None])

    for point in ([0.3, 0.4, 0.6], [0.0, 1.0, 0.5], [0.9, 0.1, 0.1]):
        autodiff = np.asarray(jax.jacfwd(embed)(jnp.array(point)))
        got = arr(mesh.evaluate_jacobians_ele_xi_pair(np.array([0]), np.array([point])))
        np.testing.assert_allclose(got, autodiff, atol=EXACT)


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_first_deriv_embeddings_are_the_matching_jacobian_column(direction):
    mesh = hermite_cube()
    xi = np.array([[0.3, 0.4, 0.6]])

    deriv = arr(mesh.evaluate_deriv_embeddings_ele_xi_pair(np.array([0]), xi, direction))
    jac = arr(mesh.evaluate_jacobians_ele_xi_pair(np.array([0]), xi))

    np.testing.assert_allclose(deriv.ravel(), jac[:, direction.index(1)], atol=EXACT)


def test_numeric_jacobian_agrees_with_the_analytic_one_on_affine_elements():
    """``eval_numeric_jac`` seeds the robust embedding solve; on an affine
    element the finite difference is exact, so the two must coincide."""
    for mesh in (unit_hex(), basic_surface(basis=[L1Basis] * 2)):
        xi = mesh.xi_grid(3)
        analytic = arr(mesh.evaluate_jacobians(0, xi))
        numeric = arr(mesh.eval_numeric_jac(0, xi))

        assert numeric.shape == analytic.shape
        np.testing.assert_allclose(numeric, analytic, atol=EXACT)


def test_numeric_jacobian_is_a_reasonable_estimate_on_a_curved_element():
    mesh = hermite_cube()
    xi = mesh.xi_grid(3)

    analytic = arr(mesh.evaluate_jacobians(0, xi))
    numeric = arr(mesh.eval_numeric_jac(0, xi, step=1e-3))

    assert np.abs(numeric - analytic).max() < 5e-2


############################################### normals

def test_surface_normals_are_orthogonal_to_both_tangents():
    mesh = basic_surface(basis=[L2Basis] * 2)
    mesh.refine(2)
    xi = mesh.xi_grid(4)
    eles = np.zeros(len(xi), dtype=int)

    normals = arr(mesh.evaluate_normals_ele_xi_pair(eles, xi))
    jac = arr(mesh.evaluate_jacobians_ele_xi_pair(eles, xi))

    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.einsum('ni,nij->nj', normals, jac), 0.0, atol=1e-5)


def test_flat_patch_has_a_constant_normal():
    mesh = basic_surface(basis=[L1Basis] * 2)
    xi = mesh.xi_grid(4)

    normals = arr(mesh.evaluate_normals_ele_xi_pair(np.zeros(len(xi), int), xi))

    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    #the patch lies on x = 0, so every normal is +/- the x axis
    np.testing.assert_allclose(np.abs(normals), np.tile([1.0, 0.0, 0.0], (len(normals), 1)), atol=1e-5)


def test_normals_are_undefined_on_a_volume_mesh():
    mesh = unit_hex()

    with pytest.raises(ValueError, match="Normals aren't defined"):
        mesh.evaluate_normals(np.array([0]), mesh.xi_grid(2))


############################################### sobolev terms

def test_sobolev_has_one_block_per_derivative_combination():
    mesh = unit_hex(basis=[L3Basis] * 3)

    blocked = arr(mesh.evaluate_sobolev(flatten=False))
    flat = arr(mesh.evaluate_sobolev())

    #L3 tabulates 2 derivative functions per direction, minus the all-zero combination
    assert blocked.shape[0] == 2 ** 3 - 1
    assert flat.size == blocked.size


def test_sobolev_is_translation_invariant():
    """It is built from derivatives, so moving the mesh must not change it."""
    mesh = unit_hex(basis=[L2Basis] * 3)
    before = arr(mesh.evaluate_sobolev())

    tform = np.eye(4)
    tform[:3, 3] = [5.0, -2.0, 7.0]
    mesh.transform(tform)

    np.testing.assert_allclose(arr(mesh.evaluate_sobolev()), before, atol=EXACT)


def test_sobolev_weights_scale_each_block():
    mesh = unit_hex(basis=[L2Basis] * 3)
    blocks = arr(mesh.evaluate_sobolev(flatten=False))

    zeroed = arr(mesh.evaluate_sobolev(weights=np.zeros(len(blocks)), flatten=False))

    np.testing.assert_allclose(zeroed, 0.0, atol=EXACT)


def test_sobolev_rejects_a_mismatched_weight_vector():
    mesh = unit_hex(basis=[L2Basis] * 3)

    with pytest.raises(ValueError, match="did not match the number of sobolev terms"):
        mesh.evaluate_sobolev(weights=np.ones(2))


############################################### xi grids

@pytest.mark.parametrize("res", [2, 5])
def test_xi_grid_spans_the_unit_cell(res):
    grid = unit_hex().xi_grid(res)

    assert grid.shape == (res ** 3, 3)
    np.testing.assert_allclose(grid.min(0), 0.0, atol=1e-9)
    np.testing.assert_allclose(grid.max(0), 1.0, atol=1e-9)


def test_xi_grid_can_exclude_the_element_boundary():
    """Used when sampling adjacent elements without double-counting the seam."""
    grid = unit_hex().xi_grid(4, boundary_points=False)

    assert grid.min() > 0 and grid.max() < 1


def test_xi_grid_dim_override_returns_a_2d_grid():
    grid = unit_hex().xi_grid(4, dim=2)

    assert grid.shape == (16, 2)


def test_xi_grid_surface_covers_the_six_faces():
    grid = unit_hex().xi_grid(5, surface=True)

    assert np.asarray(grid).size == 3 * 2 * 25 * 3


############################################### drawable geometry

def test_get_surface_returns_points_inside_the_mesh_bounds(refined_cube):
    surface = arr(refined_cube.get_surface(res=6))

    np.testing.assert_allclose(surface.min(0), -0.5, atol=1e-4)
    np.testing.assert_allclose(surface.max(0), 0.5, atol=1e-4)


def test_get_surface_just_faces_only_visits_boundary_faces(refined_cube):
    faces = refined_cube.get_faces()
    surface = arr(refined_cube.get_surface(just_faces=True, res=5))

    assert len(faces) == 6 * 4                       #2x2x2 elements, 4 exposed faces per side
    assert len(surface) == len(faces) * 25
    #every returned point is on the boundary of the cube
    assert np.isclose(np.abs(surface), 0.5, atol=1e-4).any(axis=-1).all()


def test_get_triangle_surface_indexes_real_points(refined_cube):
    points, faces = refined_cube.get_triangle_surface(res=5)

    points, faces = arr(points), np.asarray(faces)
    assert faces.shape[1] == 3
    assert faces.max() < len(points)


def test_get_lines_returns_a_polydata_spanning_the_mesh(refined_cube):
    lines = refined_cube.get_lines(res=8)

    assert lines.n_lines > 0
    np.testing.assert_allclose(np.asarray(lines.bounds).reshape(3, 2),
                               [[-0.5, 0.5]] * 3, atol=1e-4)


def test_get_hex_surface_returns_points_and_connectivity(refined_cube):
    points, connectivity = refined_cube.get_hex_surface([0])

    assert np.asarray(points).shape[1] == 3
    assert np.asarray(connectivity).max() < len(points)


def test_eval_surface_samples_only_the_boundary():
    mesh = unit_hex()

    surface = arr(mesh.eval_surface(res=5))

    #every sampled point sits on a face of the unit cube
    on_a_face = np.isclose(surface[..., None], [0.0, 1.0], atol=1e-5).any(-1)
    assert on_a_face.any(axis=-1).all()
