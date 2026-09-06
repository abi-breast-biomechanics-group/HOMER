"""Projecting points onto a mesh.

``embed_mesh_test.py`` printed "mean normal similarity: ... This should
essentially be 1" and drew the residual vectors.  Both of those are checkable:
a converged projection has its residual along the surface normal, and points
taken off the mesh must come back to where they started.

Masked and multi-state embedding is covered separately in
``test_masked_multistate_embed.py``.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis
from HOMER.geometry import basic_surface, cube

from _helpers import EXACT, arr, bulged_patch, hermite_cube


def residual_norm(residual):
    return np.linalg.norm(arr(residual), axis=-1)


@pytest.fixture(scope="module")
def patch():
    mesh = bulged_patch()
    mesh.refine(2)
    return mesh


@pytest.fixture(scope="module")
def sampled(patch):
    """Points taken from known (element, xi) pairs on the patch."""
    rng = np.random.default_rng(0)
    xi = rng.random((300, 2))
    ele = rng.integers(0, len(patch.elements), 300)
    return ele, xi, arr(patch.evaluate_embeddings_ele_xi_pair(ele, xi))


############################################### the round trip

def test_points_on_the_mesh_embed_with_no_residual(patch, sampled):
    _, _, points = sampled

    _, residual = patch.embed_points(points, return_residual=True, iterations=20)

    assert residual_norm(residual).max() < EXACT


def test_embedding_recovers_the_point_it_started_from(patch, sampled):
    _, _, points = sampled

    (ele, xi), _ = patch.embed_points(points, return_residual=True, iterations=20)

    back = arr(patch.evaluate_embeddings_ele_xi_pair(np.asarray(ele), np.asarray(xi)))
    np.testing.assert_allclose(back, points, atol=EXACT)


def test_distinct_points_get_distinct_parametric_coordinates(patch, sampled):
    """Points collapsing onto a shared (elem, xi) is the classic symptom of a
    mis-set element boundary; ``embed_mesh_test.py`` hunted for it by eye."""
    _, _, points = sampled

    (ele, xi), _ = patch.embed_points(points, return_residual=True, iterations=20)

    key = np.round(np.column_stack((np.asarray(ele), np.asarray(xi))), 4)
    assert len(np.unique(key, axis=0)) == len(points)


############################################### off-surface points

def test_residual_of_an_off_surface_point_is_along_the_normal():
    """The convergence criterion the old script printed and eyeballed.

    A converged projection is the closest point, so the residual is parallel
    to the surface normal.  Newton-Raphson gets the bulk of the points there
    immediately and the hardest few only asymptotically, so the mean is the
    tight claim and the worst case is bounded and shown to improve with more
    iterations.
    """
    rng = np.random.default_rng(1)
    mesh = bulged_patch()
    query = rng.random((400, 3))
    query[:, 0] = 0.6                       #a slab in front of the patch

    def alignment(iterations):
        (ele, xi), residual = mesh.embed_points(query, return_residual=True,
                                                iterations=iterations)
        normals = arr(mesh.evaluate_normals_ele_xi_pair(np.asarray(ele), np.asarray(xi)))
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        direction = arr(residual) / residual_norm(residual)[:, None]
        return np.abs(np.sum(normals * direction, axis=-1))

    quick, converged = alignment(20), alignment(100)

    assert quick.mean() > 1 - 1e-4
    assert quick.min() > 0.99
    assert converged.min() > quick.min()
    assert converged.mean() > 1 - 1e-5


def test_projection_is_the_closest_point_on_a_flat_patch():
    """On a plane the answer is known in closed form."""
    rng = np.random.default_rng(2)
    mesh = basic_surface(basis=[L1Basis] * 2)
    query = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(0.1, 0.9, 200),
                             rng.uniform(0.1, 0.9, 200)])

    (ele, xi), residual = mesh.embed_points(query, return_residual=True, iterations=20)

    projected = arr(mesh.evaluate_embeddings_ele_xi_pair(np.asarray(ele), np.asarray(xi)))
    np.testing.assert_allclose(projected[:, 0], 0.0, atol=EXACT)          #the patch plane
    np.testing.assert_allclose(projected[:, 1:], query[:, 1:], atol=EXACT)
    np.testing.assert_allclose(residual_norm(residual), np.abs(query[:, 0]), atol=EXACT)


############################################### volume meshes

def test_interior_points_of_a_volume_mesh_have_no_residual():
    rng = np.random.default_rng(3)
    mesh = cube(basis=[L1Basis] * 3)
    inside = rng.random((200, 3)) - 0.5

    (ele, xi), residual = mesh.embed_points(inside, return_residual=True, iterations=20)

    assert residual_norm(residual).max() < EXACT
    assert np.asarray(xi).min() >= -EXACT and np.asarray(xi).max() <= 1 + EXACT


def test_surface_embed_pins_the_result_to_an_element_face():
    rng = np.random.default_rng(4)
    mesh = cube(basis=[L1Basis] * 3)
    outside = rng.random((200, 3)) * 3 - 1.5

    (_, xi), _ = mesh.embed_points(outside, return_residual=True,
                                   surface_embed=True, iterations=20)

    xi = np.asarray(xi)
    on_a_face = (np.isclose(xi, 0.0, atol=1e-5) | np.isclose(xi, 1.0, atol=1e-5)).any(axis=-1)
    assert on_a_face.all()


def test_robust_init_seeds_a_near_degenerate_mesh():
    """``hermite_cube`` has a pinched face; the plain seed can strand a point
    in the wrong element, which is what ``robust_init_est`` is for."""
    rng = np.random.default_rng(5)
    mesh = hermite_cube()
    mesh.refine(2)
    query = rng.random((300, 3)) * 1.5 - 0.25

    _, residual = mesh.embed_points(query, return_residual=True, iterations=15,
                                    robust_init_est=True)
    _, plain = mesh.embed_points(query, return_residual=True, iterations=15)

    #the robust seed may not beat the plain one everywhere, but it must not be worse overall
    assert residual_norm(residual).mean() <= residual_norm(plain).mean() * 1.05


############################################### solver controls

def test_a_warm_start_reproduces_the_cold_answer_in_fewer_iterations(patch, sampled):
    _, _, points = sampled

    (ele, xi), cold = patch.embed_points(points, return_residual=True, iterations=20)
    warm_elexi, warm = patch.embed_points(points, init_elexi=(ele, xi),
                                          return_residual=True, iterations=2)

    np.testing.assert_allclose(np.asarray(warm_elexi[1]), np.asarray(xi), atol=EXACT)
    assert residual_norm(warm).max() < EXACT


def test_more_iterations_do_not_make_the_fit_worse():
    rng = np.random.default_rng(6)
    mesh = bulged_patch()
    query = rng.random((200, 3))

    coarse = mesh.embed_points(query, return_residual=True, iterations=2)[1]
    fine = mesh.embed_points(query, return_residual=True, iterations=25)[1]

    assert residual_norm(fine).mean() <= residual_norm(coarse).mean() + 1e-6


@pytest.mark.parametrize("grid_res", [3, 10, 20])
def test_the_coarse_seed_resolution_does_not_change_the_converged_answer(grid_res):
    rng = np.random.default_rng(7)
    mesh = bulged_patch()
    query = rng.random((150, 3))

    _, residual = mesh.embed_points(query, return_residual=True,
                                    grid_res=grid_res, iterations=25)

    reference = mesh.embed_points(query, return_residual=True, grid_res=10, iterations=25)[1]
    #a coarser or finer seed can land in a different basin near the patch edges,
    #so this pins the scale of the disagreement rather than demanding equality
    np.testing.assert_allclose(residual_norm(residual), residual_norm(reference), atol=2e-3)


def test_fit_params_override_moves_the_target_surface():
    mesh = bulged_patch()
    shifted = arr(mesh.optimisable_param_array).reshape(-1, 3)
    shifted[:, 0] += 1.0
    query = np.array([[1.0, 0.5, 0.5]])

    _, residual = mesh.embed_points(query, fit_params=shifted.ravel(),
                                    return_residual=True, iterations=20)

    #the patch centre moved to x = 1.5, and the query is on the shifted surface elsewhere
    assert residual_norm(residual).max() < 0.5
    unshifted = mesh.embed_points(query, return_residual=True, iterations=20)[1]
    assert residual_norm(unshifted).max() > residual_norm(residual).max()


def test_dim_mask_removes_a_dimension_from_the_objective():
    """With x ignored, the projection is decided by y and z alone."""
    rng = np.random.default_rng(8)
    mesh = bulged_patch()
    query = np.column_stack([rng.uniform(-9, 9, 50),        #absurd x
                             rng.uniform(0.1, 0.9, 50),     #sensible y and z
                             rng.uniform(0.1, 0.9, 50)])

    (ele, xi), _ = mesh.embed_points(query, return_residual=True, iterations=25,
                                     dim_mask=np.array([False, True, True]))

    projected = arr(mesh.evaluate_embeddings_ele_xi_pair(np.asarray(ele), np.asarray(xi))).reshape(-1, 3)
    np.testing.assert_allclose(projected[:, 1:], query[:, 1:], atol=1e-3)


############################################### reporting

def test_verbose_reports_the_error_without_opening_a_window(patch, sampled, capsys, plotter):
    _, _, points = sampled

    patch.embed_points(points, verbose=1, iterations=10)
    assert "final mean error" in capsys.readouterr().out

    #verbose=3 draws the residual vectors; given a scene it must not call show()
    patch.embed_points(points[:50], verbose=3, iterations=10, scene=plotter)
    assert len(plotter.actors) > 0
