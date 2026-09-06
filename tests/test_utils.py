"""Numerical helpers in :mod:`HOMER.utils`.

These are the pieces the mesh code leans on without ever naming them in a
traceback: nearest-neighbour seeds for the embedding solve, the element
adjacency lookup, the hexahedron volume rule.  A silent regression in any of
them shows up much later as a mesh that "looks wrong".
"""

import jax.numpy as jnp
import numpy as np
import pytest

from HOMER.utils import (all_pairings, aknn_closest_indices,
                         approx_closest_indices_Morton_nd, bcoo_repeat_scalar,
                         block_diagonal_jacobian, build_full_lookup, h_tform,
                         hex_surface_to_spherical, jax_aknn,
                         make_tiling, masked_closest_indices, rodrigues_exp,
                         skew_symmetric, spheres_to_polydata,
                         spherical_to_hex_surface, vol_hexahedron, vol_tet)


def exact_closest(a, b, weights=None):
    """Brute-force argmin, the definition the approximations are judged against."""
    diff = (b[:, None] - a[None]) ** 2
    if weights is not None:
        diff = diff * weights
    return np.argmin(diff.sum(-1), axis=1)


############################################### rotations

def test_rodrigues_exp_returns_a_rotation():
    for w in ([0.3, -0.2, 0.5], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]):
        R = np.asarray(rodrigues_exp(jnp.array(w)), dtype=float)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_rodrigues_exp_matches_the_axis_angle_definition():
    """A rotation of pi/2 about z sends x to y."""
    R = np.asarray(rodrigues_exp(jnp.array([0.0, 0.0, np.pi / 2])), dtype=float)
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-6)


def test_rodrigues_exp_of_zero_is_the_identity():
    np.testing.assert_allclose(np.asarray(rodrigues_exp(jnp.zeros(3))), np.eye(3), atol=1e-6)


def test_skew_symmetric_reproduces_the_cross_product():
    w = np.array([0.3, -0.2, 0.5])
    v = np.array([1.0, 2.0, -3.0])
    K = np.asarray(skew_symmetric(jnp.array(w)), dtype=float)

    np.testing.assert_allclose(K, -K.T, atol=1e-7)
    np.testing.assert_allclose(K @ v, np.cross(w, v), atol=1e-6)


############################################### nearest neighbours

@pytest.mark.parametrize("fdim", [2, 3, 6])
def test_aknn_matches_brute_force(fdim):
    rng = np.random.default_rng(0)
    a, b = rng.random((200, fdim)), rng.random((17, fdim))

    got = np.asarray(aknn_closest_indices(jnp.array(a), jnp.array(b)))

    np.testing.assert_array_equal(got, exact_closest(a, b))


def test_masked_closest_indices_ignores_the_masked_components():
    """The masked search must not let a masked component break the tie.

    The two candidates differ only in the masked dimension, and the second is
    the one the *unmasked* search would reject.
    """
    rng = np.random.default_rng(1)
    a, b = rng.random((150, 3)), rng.random((11, 3))
    mask = jnp.array([True, True, False])

    got = np.asarray(masked_closest_indices(jnp.array(a), jnp.array(b), mask))

    np.testing.assert_array_equal(got, exact_closest(a, b, np.array([1.0, 1.0, 0.0])))


def test_masked_closest_indices_with_a_full_mask_is_the_plain_search():
    rng = np.random.default_rng(2)
    a, b = rng.random((80, 3)), rng.random((9, 3))

    masked = np.asarray(masked_closest_indices(jnp.array(a), jnp.array(b),
                                               jnp.ones(3, dtype=bool)))

    np.testing.assert_array_equal(masked, exact_closest(a, b))


def test_morton_search_is_close_but_is_only_an_approximation():
    """The Z-curve seed is allowed to miss; it must not miss by much.

    This is what makes it usable as a coarse seed and unusable as an answer,
    and it is the reason the masked path exists at all.
    """
    rng = np.random.default_rng(3)
    a, b = rng.random((500, 3)), rng.random((60, 3))

    got = np.asarray(approx_closest_indices_Morton_nd(jnp.array(a), jnp.array(b)))
    best = exact_closest(a, b)

    approx_d = np.linalg.norm(b - a[got], axis=-1)
    best_d = np.linalg.norm(b - a[best], axis=-1)
    assert np.mean(got == best) > 0.5
    assert np.max(approx_d - best_d) < 0.2


def test_jax_aknn_returns_sorted_distances_and_matching_indices():
    rng = np.random.default_rng(4)
    a, b = rng.random((60, 3)), rng.random((40, 3))

    dists, inds = jax_aknn(jnp.array(a), jnp.array(b), 3)

    dists, inds = np.asarray(dists), np.asarray(inds)
    assert dists.shape == inds.shape == (60, 3)
    np.testing.assert_array_equal(np.sort(dists, axis=-1), dists)
    np.testing.assert_allclose(np.linalg.norm(a[:, None] - b[inds], axis=-1),
                               dists, atol=1e-5)


############################################### volumes and transforms

def test_vol_tet_matches_the_determinant_formula():
    p = np.eye(3)
    assert vol_tet(np.zeros(3), p[0], p[1], p[2]) == pytest.approx(1 / 6)


def test_vol_hexahedron_of_the_unit_cube_is_one():
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
    assert vol_hexahedron(corners) == pytest.approx(1.0)


def test_vol_hexahedron_scales_cubically():
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
    assert vol_hexahedron(corners * 3.0) == pytest.approx(27.0)


def test_h_tform_applies_a_homogeneous_transform():
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    tform = np.eye(4)
    tform[:3, :3] = np.asarray(rodrigues_exp(jnp.array([0.0, 0.0, np.pi / 2])), dtype=float)
    tform[:3, 3] = [10.0, 0.0, 0.0]

    got = np.asarray(h_tform(points, tform), dtype=float).reshape(-1, 3)

    expected = points @ tform[:3, :3].T + tform[:3, 3]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_h_tform_identity_is_a_no_op():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    np.testing.assert_allclose(np.asarray(h_tform(points, np.eye(4))).reshape(-1, 3),
                               points, atol=1e-6)


############################################### indexing and sparsity

def test_all_pairings_is_a_product_with_the_first_list_varying_fastest():
    assert all_pairings([0, 1], [2, 3]) == [(0, 2), (1, 2), (0, 3), (1, 3)]


def test_block_diagonal_jacobian_has_one_dense_block_per_output_group():
    mat = block_diagonal_jacobian(2, 3, 2).toarray()

    assert mat.shape == (4, 6)
    assert mat.sum() == 2 * 3 * 2
    np.testing.assert_array_equal(mat[:2, 3:], 0)
    np.testing.assert_array_equal(mat[2:, :3], 0)


def test_bcoo_repeat_scalar_repeats_along_the_requested_axis():
    from jax.experimental import sparse

    dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    repeated = bcoo_repeat_scalar(sparse.BCOO.fromdense(dense), 3, axis=1).todense()

    np.testing.assert_allclose(np.asarray(repeated), np.repeat(np.asarray(dense), 3, axis=1))


def test_build_full_lookup_marks_unreachable_neighbours():
    """A single element has no neighbours, so only the centre entry is itself."""
    lookup = -np.ones((1, 3, 2), dtype=int)

    full = np.asarray(build_full_lookup(jnp.array(lookup))).reshape(1, 3, 3, 3)

    assert full[0, 1, 1, 1] == 0                    #stay put
    assert (full[0] == -1).sum() == 26              #every step off the element leaves the mesh


############################################### surface parameterisations

def test_hex_surface_spherical_round_trip():
    """theta lives in [0, pi] and phi in [0, 2pi); the round trip is the identity there."""
    angles = np.array([[0.3, 0.4], [1.2, 5.5], [2.0, 0.1], [np.pi / 2, np.pi]])

    back = np.asarray(hex_surface_to_spherical(spherical_to_hex_surface(angles)), dtype=float)

    np.testing.assert_allclose(back, angles, atol=1e-4)


def test_spherical_to_hex_surface_lands_on_the_cube_face():
    """Every mapped point must have at least one coordinate pinned to 0 or 1."""
    rng = np.random.default_rng(6)
    angles = np.stack([rng.uniform(0.05, np.pi - 0.05, 200),
                       rng.uniform(0, 2 * np.pi, 200)], axis=-1)

    xi = np.asarray(spherical_to_hex_surface(angles), dtype=float)

    assert xi.min() >= -1e-6 and xi.max() <= 1 + 1e-6
    on_a_face = np.isclose(xi, 0, atol=1e-5) | np.isclose(xi, 1, atol=1e-5)
    assert np.all(on_a_face.any(axis=-1))


def test_make_tiling_connectivity_indexes_real_points():
    points, lines = make_tiling(2, 2)

    lines = np.asarray(lines).reshape(-1, 3)
    assert np.asarray(points).shape[1] == 2
    assert np.all(lines[:, 0] == 2)                 #vtk line cells: 2 points each
    assert lines[:, 1:].max() < len(points)


def test_spheres_to_polydata_merges_instances_and_offsets_connectivity():
    """M copies of one shared triangulation become a single mesh."""
    verts = np.random.default_rng(5).random((4, 6, 3))     #4 instances, 6 vertices each
    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])             #vtk flat triangles

    poly = spheres_to_polydata(verts, faces)

    assert poly.n_points == 4 * 6
    assert poly.n_cells == 4 * 2
    np.testing.assert_allclose(poly.points, verts.reshape(-1, 3), atol=1e-6)
