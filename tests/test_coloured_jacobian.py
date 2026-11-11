"""The coloured sparse Jacobian, against the dense one it stands in for.

:func:`~HOMER.jacobian_evaluator.make_jac_for_mesh_func` reconstructs a
Jacobian from one jvp per mesh colour rather than one per parameter.  That is
only worth anything if the result is the Jacobian, so every test here compares
against ``jax.jacfwd`` on the same residual.

It keeps the columns it decoded between calls, and trusts them only while one
jvp along a random direction agrees, which makes three things worth pinning:
that the kept pattern still gives the right matrix once the parameters and the
per-call data have moved, that an entry masked out when it was decoded comes
back when it is live again, and that a colouring that cannot be right is
refused rather than returned.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

from HOMER.basis_definitions import L1
from HOMER.geometry import cube
from HOMER.jacobian_evaluator import make_jac_for_mesh_func
from jax.flatten_util import ravel_pytree

from _helpers import EXACT


@pytest.fixture(scope="module")
def problem():
    """A mesh, a residual taking per-call data, and two sets of that data."""
    mesh = cube(basis=[L1] * 3)
    mesh.refine([2, 2, 2])
    mesh.generate_mesh()

    eles, grid = np.arange(len(mesh.elements)), mesh.xi_grid(3)
    n_pts = eles.shape[0] * grid.shape[0]

    def residual(params, targets, weights):
        surface = mesh.evaluate_embeddings(eles, grid, fit_params=params).reshape(-1, 3)
        return ((surface - targets) * weights[:, None]).ravel()

    rng = np.random.default_rng(0)
    data = [{"targets": jnp.asarray(rng.normal(size=(n_pts, 3))),
             "weights": jnp.asarray(rng.random(n_pts))} for _ in range(2)]
    unit = {"targets": data[0]["targets"], "weights": jnp.ones(n_pts)}

    params = jnp.asarray(mesh.optimisable_param_array)
    moved = params + 0.01 * jnp.asarray(rng.normal(size=params.shape))
    return mesh, residual, params, moved, unit, data


def dense_jac(residual, params, **kwargs):
    return np.asarray(jax.jacfwd(lambda p: residual(p, **kwargs))(params))


@pytest.mark.parametrize("fields_separable", [None, False, True])
def test_the_coloured_jacobian_matches_the_dense_one(problem, fields_separable):
    mesh, residual, params, _, unit, _ = problem
    jac = make_jac_for_mesh_func(mesh, residual, fields_separable)[1]

    assert np.allclose(jac(params, **unit).toarray(),
                       dense_jac(residual, params, **unit), atol=EXACT)


def test_the_kept_pattern_holds_as_params_and_data_move(problem):
    mesh, residual, params, moved, unit, data = problem
    jac = make_jac_for_mesh_func(mesh, residual)[1]

    for kwargs in [unit] + data:
        for point in (params, moved):
            assert np.allclose(jac(point, **kwargs).toarray(),
                               dense_jac(residual, point, **kwargs), atol=EXACT)


def test_it_returns_something_scipy_can_use(problem):
    mesh, residual, params, _, unit, _ = problem
    out = make_jac_for_mesh_func(mesh, residual)[1](params, **unit)

    assert isinstance(out, scipy.sparse.csr_array)
    assert out.shape == (residual(params, **unit).shape[0], params.shape[0])
    assert out.nnz > 0


def test_an_entry_masked_when_decoded_comes_back_once_it_is_live(problem):
    """Data that zeroes entries on the first call cannot cost them later."""
    mesh, residual, params, _, unit, _ = problem
    masked = {"targets": unit["targets"],
              "weights": jnp.zeros_like(unit["weights"]).at[:5].set(1.0)}
    jac = make_jac_for_mesh_func(mesh, residual)[1]
    jac(params, **masked)

    assert np.allclose(jac(params, **unit).toarray(),
                       dense_jac(residual, params, **unit), atol=EXACT)


def test_a_separable_colouring_is_not_trusted_for_a_coupled_residual():
    """Asked for, it is refused; left to itself, the maker finds the coupled one.

    The exact embedding derivative slides each point as the mesh moves, so
    every residual component depends on every component of its element.
    """
    from HOMER.examples import bulged_patch
    mesh = bulged_patch()
    mesh.refine(2)
    points = jnp.asarray(np.random.default_rng(1).random((40, 3)))

    def residual(params):
        return mesh.embed_points(points, fit_params=params, return_residual=True)[1].ravel()

    params = jnp.asarray(mesh.optimisable_param_array)
    with pytest.raises(ValueError, match="colouring"):
        make_jac_for_mesh_func(mesh, residual, fields_seperable=True)[1](params)
    np.testing.assert_allclose(make_jac_for_mesh_func(mesh, residual)[1](params).toarray(),
                               np.asarray(jax.jacfwd(residual)(params)), atol=1e-4)


############################################### parameter trees

def tree_reference(residual, params):
    """``jax.jacfwd`` over the ravelled tree, the column order the maker promises."""
    flat, unravel = ravel_pytree(params)
    return np.asarray(jax.jacfwd(lambda x: residual(unravel(x)))(flat))


@pytest.fixture(scope="module")
def grid_problem():
    from HOMER.basis_definitions import H3
    mesh = cube(basis=[H3] * 3)
    mesh.refine(2)
    rng = np.random.default_rng(3)
    eles = rng.integers(len(mesh.elements), size=200)
    xis = jnp.asarray(rng.random((200, 3)))
    return mesh, eles, xis, rng


def test_a_shape_model_gets_a_block_for_the_mesh_and_one_for_its_modes(grid_problem):
    """p = mean + offset + modes @ weights: the offset coloured, the weights dense."""
    mesh, eles, xis, rng = grid_problem
    mean = jnp.asarray(mesh.optimisable_param_array)
    modes = jnp.asarray(rng.normal(scale=0.05, size=(mean.size, 4)))

    def residual(params):
        offset, weights = params
        return mesh.evaluate_embeddings_ele_xi_pair(
            eles, xis, fit_params=mean + offset + modes @ weights).ravel()

    params = (jnp.zeros_like(mean), jnp.asarray(rng.normal(size=4)))
    jac = make_jac_for_mesh_func((mesh, None), residual, init_estimate=params)[1](ravel_pytree(params)[0])

    assert jac.shape == (residual(params).size, mean.size + 4)
    np.testing.assert_allclose(jac.toarray(), tree_reference(residual, params), atol=EXACT)


def test_two_meshes_and_the_term_coupling_them(grid_problem):
    mesh, eles, xis, _ = grid_problem
    other = cube(basis=[L1] * 3)
    other.refine(3)
    other_eles = np.random.default_rng(4).integers(len(other.elements), size=len(eles))

    def residual(params):
        a = mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params['a'])
        b = other.evaluate_embeddings_ele_xi_pair(other_eles, xis, fit_params=params['b'])
        return jnp.concatenate([a.ravel(), b.ravel(), (a - b).ravel()])

    params = {'a': jnp.asarray(mesh.optimisable_param_array),
              'b': jnp.asarray(other.optimisable_param_array)}
    jac = make_jac_for_mesh_func({'a': mesh, 'b': other}, residual,
                                 init_estimate=params)[1](ravel_pytree(params)[0])

    np.testing.assert_allclose(jac.toarray(), tree_reference(residual, params), atol=EXACT)


def test_a_pose_beside_the_mesh_parameters(grid_problem):
    """A rigid translation applied after evaluation: a dense leaf of three."""
    mesh, eles, xis, _ = grid_problem

    def residual(params):
        return (mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params['shape'])
                + params['shift']).ravel()

    params = {'shape': jnp.asarray(mesh.optimisable_param_array), 'shift': jnp.ones(3)}
    jac = make_jac_for_mesh_func({'shape': mesh, 'shift': None}, residual,
                                 init_estimate=params)[1](ravel_pytree(params)[0])

    np.testing.assert_allclose(jac.toarray(), tree_reference(residual, params), atol=EXACT)


def test_each_mesh_settles_its_own_colouring():
    """One mesh held to points it embeds (coupled), the other evaluated (separable)."""
    from HOMER.examples import bulged_patch
    patch = bulged_patch()
    patch.refine(2)
    block = cube(basis=[L1] * 3)
    block.refine(2)
    rng = np.random.default_rng(5)
    points = jnp.asarray(rng.random((30, 3)))
    eles, xis = rng.integers(len(block.elements), size=30), jnp.asarray(rng.random((30, 3)))

    def residual(params):
        embedded = patch.embed_points(points, fit_params=params[0], return_residual=True)[1]
        held = block.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params[1])
        return jnp.concatenate([embedded.ravel(), held.ravel()])

    params = (jnp.asarray(patch.optimisable_param_array), jnp.asarray(block.optimisable_param_array))
    jac = make_jac_for_mesh_func((patch, block), residual,
                                 init_estimate=params)[1](ravel_pytree(params)[0])

    np.testing.assert_allclose(jac.toarray(), tree_reference(residual, params), atol=1e-4)


def test_a_cohort_stacked_side_by_side_shares_one_colouring(grid_problem):
    """Subjects never meet in a residual, so each can reuse the mesh's colours."""
    mesh, eles, xis, rng = grid_problem
    colours = mesh.get_colouring_dict(fields_seperable=True)
    colour_of = np.array([colours[k] for k in range(len(colours))])
    other_eles = rng.integers(len(mesh.elements), size=len(eles))

    def residual(cohort):
        return jnp.concatenate([
            mesh.evaluate_embeddings_ele_xi_pair(e, xis, fit_params=p).ravel()
            for e, p in zip((eles, other_eles), cohort)])

    params = jnp.stack([jnp.asarray(mesh.optimisable_param_array)] * 2)
    jac = make_jac_for_mesh_func(np.tile(colour_of, 2), residual,
                                 init_estimate=params)[1](params.ravel())

    np.testing.assert_allclose(jac.toarray(), tree_reference(residual, params), atol=EXACT)


def test_a_colouring_too_coarse_for_the_function_raises(grid_problem):
    mesh, eles, xis, _ = grid_problem
    params = jnp.asarray(mesh.optimisable_param_array)

    def residual(p):
        return mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=p).ravel()

    with pytest.raises(ValueError, match="no colouring"):
        make_jac_for_mesh_func(np.zeros(params.size, dtype=int), residual)[1](params)


def test_a_batch_is_block_diagonal_with_each_item_on_its_own_data(grid_problem):
    """Three subjects on one template, each with its own points, a pose beside each shape."""
    mesh, _, xis, rng = grid_problem
    eles = jnp.asarray(rng.integers(len(mesh.elements), size=(3, len(xis))))

    def residual(params, eles):
        return (mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params['shape'])
                + params['shift']).ravel()

    shape = jnp.asarray(mesh.optimisable_param_array)
    params = {'shape': jnp.stack([shape, shape * 1.1, shape * 0.9]), 'shift': jnp.ones((3, 3))}
    jac = make_jac_for_mesh_func({'shape': mesh, 'shift': None}, residual, batched=True,
                                 init_estimate=params)[1]
    out = jac(ravel_pytree(params)[0], eles=eles)

    cohort = lambda p: jax.vmap(residual)(p, eles).ravel()
    np.testing.assert_allclose(out.toarray(), tree_reference(cohort, params), atol=EXACT)


def test_entries_crossing_the_threshold_are_kept_at_their_columns(grid_problem):
    """Squared parameters: an entry is zero wherever its parameter is, and comes back when it is not."""
    mesh, eles, xis, rng = grid_problem

    def residual(p):
        return mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=p * p).ravel()

    base = np.asarray(mesh.optimisable_param_array)
    jac = make_jac_for_mesh_func(mesh, residual)[1]
    for zeroed in (0.5, 0.3, 0.0, 0.3):
        params = jnp.asarray(np.where(rng.random(base.size) < zeroed, 0.0, base + 0.1))
        np.testing.assert_allclose(jac(params).toarray(), np.asarray(jax.jacfwd(residual)(params)),
                                   atol=EXACT)


############################################### blocks of rows

def mixed_residual(mesh, eles, xis):
    """A coupled block -- each row sums the components -- beside a separable one."""
    def residual(params):
        points = mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params)
        return {'coupled': points.sum(-1), 'separable': points.ravel()}
    return residual


def test_each_block_of_rows_is_differentiated_on_its_own(grid_problem):
    mesh, eles, xis, _ = grid_problem
    residual = mixed_residual(mesh, eles, xis)
    params = jnp.asarray(mesh.optimisable_param_array)
    flat = lambda p: ravel_pytree(residual(p))[0]

    jac = make_jac_for_mesh_func(mesh, residual)[1](params)

    np.testing.assert_allclose(jac.toarray(), np.asarray(jax.jacfwd(flat)(params)), atol=EXACT)


def test_each_block_picks_its_own_colouring_from_the_candidates(grid_problem):
    mesh, eles, xis, _ = grid_problem
    residual = mixed_residual(mesh, eles, xis)
    params = jnp.asarray(mesh.optimisable_param_array)
    as_array = lambda colouring: np.array([colouring[k] for k in range(len(colouring))])
    separable = as_array(mesh.get_colouring_dict(fields_seperable=True))
    coupled = as_array(mesh.get_colouring_dict(fields_seperable=False))
    flat = lambda p: ravel_pytree(residual(p))[0]

    with pytest.raises(ValueError, match=r"\['coupled'\]"):
        make_jac_for_mesh_func(separable, residual)[1](params)
    jac = make_jac_for_mesh_func(np.stack([separable, coupled]), residual)[1](params)

    np.testing.assert_allclose(jac.toarray(), np.asarray(jax.jacfwd(flat)(params)), atol=EXACT)


def test_a_batch_of_blocks_follows_the_batched_output(grid_problem):
    mesh, _, xis, rng = grid_problem
    eles = jnp.asarray(rng.integers(len(mesh.elements), size=(2, len(xis))))

    def residual(params, eles):
        return mixed_residual(mesh, eles, xis)(params)

    shape = jnp.asarray(mesh.optimisable_param_array)
    params = jnp.stack([shape, shape * 1.1])
    flat = lambda p: ravel_pytree(jax.vmap(residual)(p.reshape(params.shape), eles))[0]

    jac = make_jac_for_mesh_func(mesh, residual, batched=True,
                                 init_estimate=params)[1](params.ravel(), eles=eles)

    np.testing.assert_allclose(jac.toarray(), np.asarray(jax.jacfwd(flat)(params.ravel())), atol=EXACT)


@pytest.mark.parametrize("batched", [False, True])
def test_the_residual_comes_back_in_the_jacobians_row_order(grid_problem, batched):
    """The Jacobian of the returned residual is the Jacobian returned beside it."""
    mesh, _, xis, rng = grid_problem
    eles = jnp.asarray(rng.integers(len(mesh.elements), size=(2, len(xis))))
    shape = jnp.asarray(mesh.optimisable_param_array)

    def terms(params, eles):
        return mixed_residual(mesh, eles, xis)(params)

    params, data = (jnp.stack([shape, shape * 1.1]), eles) if batched else (shape, eles[0])
    residual, jacobian = make_jac_for_mesh_func(mesh, terms, batched=batched, init_estimate=params)

    x = params.ravel()
    np.testing.assert_allclose(jacobian(x, eles=data).toarray(),
                               np.asarray(jax.jacfwd(lambda x: residual(x, eles=data))(x)), atol=EXACT)


def test_a_tree_or_a_batch_needs_its_structure():
    mesh = cube(basis=[L1] * 3)
    with pytest.raises(ValueError, match="init_estimate"):
        make_jac_for_mesh_func((mesh, None), lambda p: p[1])
    with pytest.raises(ValueError, match="init_estimate"):
        make_jac_for_mesh_func(mesh, lambda p: p, batched=True)
