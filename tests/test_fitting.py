"""Fitting a mesh to data: the linear solve, the nonlinear solve, and the
sparse Jacobian machinery underneath both.

``linear_fit_mesh_test.py``, ``optimise_mesh_test.py`` and
``point_to_plane_fit_test.py`` all ended by drawing the fitted mesh over the
target cloud.  A fit either reaches the target or it does not, and when the
target is representable in the fitting basis the answer is exact -- so that
is what is asserted here.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from HOMER.basis_definitions import B3Basis, H3Basis, L1Basis, L2Basis, L3Basis
from HOMER.fitting import point_cloud_fit
from HOMER.geometry import basic_surface, cube
from HOMER.jacobian_evaluator import estimate_sparsity, jacobian
from HOMER.mesh import column_equilibrated_lstsq

from _helpers import CLOSE, EXACT, arr, bulged_patch, unit_hex


def dense(jac_result):
    return jac_result.toarray() if hasattr(jac_result, 'toarray') else np.asarray(jac_result)


############################################### the weight matrix

def test_weight_matrix_reproduces_the_embedding():
    """``get_xi_weight_mat`` is the linear map the whole linear fit rests on."""
    mesh = bulged_patch()
    grid = mesh.xi_grid(5)
    eles = np.zeros(len(grid), dtype=int)

    weights = arr(mesh.get_xi_weight_mat(eles, grid))
    params = arr(mesh.true_param_array).reshape(-1, 3)

    np.testing.assert_allclose(weights @ params,
                               arr(mesh.evaluate_embeddings_ele_xi_pair(eles, grid)),
                               atol=EXACT)


def test_weight_matrix_rows_sum_to_one_for_a_lagrange_basis():
    """A partition of unity in 1-D stays one under the tensor product."""
    mesh = bulged_patch()
    grid = mesh.xi_grid(5)

    weights = arr(mesh.get_xi_weight_mat(np.zeros(len(grid), int), grid))

    np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-5)


############################################### the preconditioned solve

def badly_scaled_system(rng, dead_column=False):
    """Full rank, but with column norms spread over seven orders of magnitude."""
    A = np.asarray(rng.random((40, 8)))
    A[:, 5] *= 1e-4
    A[:, 2] *= 1e3
    if dead_column:
        A[:, 3] = 0.0
    return A, np.asarray(rng.random((40, 3)))


def test_equilibration_does_not_change_the_answer():
    """Column scaling cannot move the minimiser of a full-rank system.

    It is unique and invariant under column scaling, so this is a statement
    about exact arithmetic -- checked in float64, where there is enough
    precision for the two paths to agree.  Everything the equilibration buys
    is in the float32 path.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    try:
        A, b = badly_scaled_system(np.random.default_rng(0))
        A64, b64 = jnp.asarray(A, jnp.float64), jnp.asarray(b, jnp.float64)

        plain = np.asarray(jnp.linalg.lstsq(A64, b64)[0])
        equilibrated = np.asarray(column_equilibrated_lstsq(A64, b64)[0])

        assert np.linalg.cond(A) > 1e6
        np.testing.assert_allclose(equilibrated, plain, rtol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("basis,improvement", [(L2Basis, 1), (H3Basis, 10), (B3Basis, 10)],
                         ids=lambda x: getattr(x, '__name__', x))
def test_equilibration_recovers_precision_on_a_real_weight_matrix(basis, improvement):
    """The matrices HOMER actually builds are why this exists.

    A refined Hermite mesh reaches ``cond(W) = 1.1e4`` and a B-spline control
    net 4.1e4 -- the derivative and control-point weights are an order of
    magnitude smaller than the value weights, cubed over three directions.
    Recovering known parameters through such a matrix in float32 is where the
    scaling pays; on a well-conditioned Lagrange matrix it is a wash.
    """
    rng = np.random.default_rng(0)
    mesh = unit_hex(basis=[basis] * 3)
    mesh.refine(2)
    grid = mesh.xi_grid(basis.order + 2)
    eles = np.repeat(np.arange(len(mesh.elements)), len(grid))
    weights = np.asarray(mesh.get_xi_weight_mat(eles, np.tile(grid, (len(mesh.elements), 1))))

    exact = rng.random((weights.shape[1], 3))
    targets = weights @ exact                            #a consistent system
    A, b = jnp.asarray(weights, jnp.float32), jnp.asarray(targets, jnp.float32)

    plain = np.abs(np.asarray(jnp.linalg.lstsq(A, b)[0]) - exact).max()
    equilibrated = np.abs(np.asarray(column_equilibrated_lstsq(A, b)[0]) - exact).max()

    assert equilibrated <= plain / improvement


def test_equilibration_stays_jit_able_and_differentiable():
    """``linear_fit`` is used inside traced code, so the solve must survive it."""
    import jax

    rng = np.random.default_rng(2)
    for dead in (False, True):
        A, b = badly_scaled_system(rng, dead_column=dead)
        A, b = jnp.asarray(A), jnp.asarray(b)

        jitted = jax.jit(lambda M, t: column_equilibrated_lstsq(M, t)[0])
        np.testing.assert_allclose(np.asarray(jitted(A, b)),
                                   np.asarray(column_equilibrated_lstsq(A, b)[0]), atol=1e-4)

        for argnum in (0, 1):
            grad = jax.grad(lambda *a: jnp.sum(column_equilibrated_lstsq(*a)[0] ** 2),
                            argnums=argnum)(A, b)
            assert np.all(np.isfinite(np.asarray(grad))), (dead, argnum)


def test_a_dead_column_is_not_amplified():
    """A parameter nothing depends on must come back at zero, not at 1/tiny.

    Scaling by a column norm of ~0 is the obvious trap here: the column stays
    zero going in, but its round-off comes back multiplied by the reciprocal.
    """
    A, b = badly_scaled_system(np.random.default_rng(3), dead_column=True)

    params = np.asarray(column_equilibrated_lstsq(jnp.asarray(A), jnp.asarray(b))[0])

    assert np.abs(params[3]).max() < 1e-5
    assert np.all(np.isfinite(params))


def test_equilibration_accepts_a_scalar_target():
    """Scalar fields give a 1-D target, so the rescale must not assume 2-D."""
    A, b = badly_scaled_system(np.random.default_rng(4))

    params = np.asarray(column_equilibrated_lstsq(jnp.asarray(A), jnp.asarray(b[:, 0]))[0])

    assert params.shape == (8,)


############################################### the linear fit

def test_linear_fit_is_exact_when_the_target_is_representable():
    """L1 geometry sampled onto an L3 mesh: the cubic space contains it."""
    target = unit_hex(basis=[L1Basis] * 3)
    fitted = unit_hex(basis=[L3Basis] * 3)
    grid = fitted.xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)
    targets = arr(target.evaluate_embeddings_ele_xi_pair(eles, grid))

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    np.testing.assert_allclose(arr(fitted.evaluate_embeddings_ele_xi_pair(eles, grid)),
                               targets, atol=CLOSE)


def test_linear_fit_across_several_elements():
    """The version ``linear_fit_mesh_test.py`` drew: four elements, one solve.

    Both meshes are refined from the same single-quad topology, so element k
    of one covers the same parametric patch as element k of the other and the
    two can be sampled on a shared (element, xi) list.
    """
    target = bulged_patch()
    target.refine(2)
    fitted = basic_surface(basis=[L3Basis] * 2)
    fitted.refine(2)

    res = 8
    eles = np.repeat(np.arange(len(fitted.elements)), res ** 2)
    grid = np.tile(fitted.xi_grid(res), (len(fitted.elements), 1))
    targets = arr(target.evaluate_embeddings_ele_xi_pair(eles, grid))

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    got = arr(fitted.evaluate_embeddings_ele_xi_pair(eles, grid))
    np.testing.assert_allclose(got, targets, atol=CLOSE)


def test_linear_fit_return_params_does_not_touch_the_mesh():
    mesh = unit_hex(basis=[L2Basis] * 3)
    grid = mesh.xi_grid(5)
    eles = np.zeros(len(grid), dtype=int)
    before = arr(mesh.true_param_array)

    params = mesh.linear_fit(np.zeros((len(grid), 3)),
                            weight_mat=mesh.get_xi_weight_mat(eles, grid),
                            return_params=True)

    assert params is not None
    np.testing.assert_allclose(arr(mesh.true_param_array), before, atol=EXACT)


def test_linear_fit_does_not_respect_fixed_parameters():
    """Documented limitation, pinned so it cannot change silently.

    ``linear_fit`` solves the unconstrained normal equations; use
    :func:`point_cloud_fit` when constraints matter.
    """
    target = bulged_patch()
    fitted = basic_surface(basis=[L2Basis] * 2)
    fitted.nodes[0].fix_parameter('loc')
    fitted.generate_mesh()
    pinned = np.array(fitted.nodes[0].loc, dtype=float)
    grid = fitted.xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)

    fitted.linear_fit(arr(target.evaluate_embeddings_ele_xi_pair(eles, grid)),
                      weight_mat=fitted.get_xi_weight_mat(eles, grid))

    assert np.abs(np.array(fitted.nodes[0].loc, dtype=float) - pinned).max() > 0.1


def test_linear_fit_rejects_an_underdetermined_system():
    mesh = unit_hex(basis=[L3Basis] * 3)
    grid = mesh.xi_grid(2)                       #8 samples, 64 unknowns
    eles = np.zeros(len(grid), dtype=int)

    with pytest.raises(AssertionError, match="undertederimined"):
        mesh.linear_fit(arr(mesh.evaluate_embeddings_ele_xi_pair(eles, grid)),
                        weight_mat=mesh.get_xi_weight_mat(eles, grid))


def test_linear_fit_ignores_rows_marked_empty():
    """Rows equal to ``target_empty`` drop out of the solve."""
    target = unit_hex(basis=[L1Basis] * 3)
    grid = unit_hex(basis=[L2Basis] * 3).xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)
    targets = arr(target.evaluate_embeddings_ele_xi_pair(eles, grid))

    fitted = unit_hex(basis=[L2Basis] * 3)
    weights = arr(fitted.get_xi_weight_mat(eles, grid))
    fitted.linear_fit(targets, weight_mat=weights)
    reference = arr(fitted.true_param_array)

    spoiled = np.concatenate([targets, np.full((20, 3), -1.0)])
    spoiled_weights = np.concatenate([weights, np.zeros((20, weights.shape[1]))])
    with_junk = unit_hex(basis=[L2Basis] * 3)
    with_junk.linear_fit(spoiled, weight_mat=spoiled_weights)

    np.testing.assert_allclose(arr(with_junk.true_param_array), reference, atol=CLOSE)


############################################### the nonlinear fit

@pytest.fixture(scope="module")
def curved_target():
    mesh = bulged_patch()
    grid = mesh.xi_grid(20)
    return arr(mesh.evaluate_embeddings_ele_xi_pair(np.zeros(len(grid), int), grid))


def test_point_cloud_fit_moves_the_mesh_onto_the_cloud(curved_target):
    """``optimise_mesh_test.py`` drew the before and after; this measures them."""
    mesh = basic_surface(basis=[H3Basis] * 2)
    fit_fn, jac_fn = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=0.0)
    start = arr(mesh.optimisable_param_array)

    before = cKDTree(curved_target).query(arr(mesh.get_surface(res=20)))[0].max()
    result = least_squares(fit_fn, start, jac=jac_fn, verbose=0, max_nfev=60)
    mesh.update_from_params(result.x)
    after = cKDTree(curved_target).query(arr(mesh.get_surface(res=20)))[0].max()

    assert after < before / 4
    assert after < 0.1


def test_point_cloud_fit_holds_fixed_nodes(curved_target):
    """The constrained pathway: a pinned corner must not move."""
    mesh = basic_surface(basis=[H3Basis] * 2)
    mesh.nodes[1].fix_parameter('loc')
    mesh.nodes[2].fix_parameter('loc')
    mesh.generate_mesh()
    pinned = np.array([np.array(mesh.nodes[i].loc, dtype=float) for i in (1, 2)])

    fit_fn, jac_fn = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=0.0)
    result = least_squares(fit_fn, arr(mesh.optimisable_param_array), jac=jac_fn,
                           verbose=0, max_nfev=40)
    mesh.update_from_params(result.x)

    after = np.array([np.array(mesh.nodes[i].loc, dtype=float) for i in (1, 2)])
    np.testing.assert_allclose(after, pinned, atol=EXACT)


def test_the_sobolev_term_actually_reaches_the_optimiser(curved_target):
    """The regularisation must be evaluated at the *trial* parameters.

    Evaluated at the mesh's stored parameters instead, the block is constant,
    its Jacobian rows are identically zero, and ``sob_weight`` silently does
    nothing.
    """
    mesh = basic_surface(basis=[H3Basis] * 2)
    n_sobolev = arr(mesh.evaluate_sobolev()).size
    start = arr(mesh.optimisable_param_array)

    fit_fn, jac_fn = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=0.01)

    block = dense(jac_fn(start))[-n_sobolev:]
    assert np.abs(block).sum() > 0
    #and the residual block responds to the parameters it is handed
    assert not np.allclose(np.asarray(fit_fn(start))[-n_sobolev:],
                           np.asarray(fit_fn(start * 1.5))[-n_sobolev:])


def test_sobolev_weight_changes_the_cost(curved_target):
    mesh = basic_surface(basis=[H3Basis] * 2)
    start = arr(mesh.optimisable_param_array)

    light = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=0.01)[0]
    heavy = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=1.0)[0]

    assert np.sum(np.asarray(heavy(start)) ** 2) > np.sum(np.asarray(light(start)) ** 2)


############################################### sparse jacobians

def simple_cost(params):
    """Each output touches exactly two inputs, so the sparsity is known."""
    import jax.numpy as jnp
    return jnp.stack([params[0] ** 2 + params[1],
                      params[1] * params[2],
                      params[2] + params[3] ** 3])


def test_sparse_and_dense_jacobians_agree():
    import jax

    start = np.array([1.0, 2.0, 3.0, 4.0])

    _, sparse_jac = jacobian(simple_cost, init_estimate=start, sparse=True)
    _, dense_jac = jacobian(simple_cost, init_estimate=start, sparse=False)
    reference = np.asarray(jax.jacfwd(simple_cost)(start))

    np.testing.assert_allclose(dense(sparse_jac(start)), reference, atol=1e-5)
    np.testing.assert_allclose(dense(dense_jac(start)), reference, atol=1e-5)


def test_estimate_sparsity_finds_the_true_pattern():
    start = np.array([1.0, 2.0, 3.0, 4.0])

    pattern = np.asarray(estimate_sparsity(simple_cost, start).todense()) != 0

    expected = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    np.testing.assert_array_equal(pattern, expected)


def test_a_supplied_sparsity_is_used_as_given():
    import jax
    from jax.experimental import sparse as jsparse

    start = np.array([1.0, 2.0, 3.0, 4.0])
    pattern = estimate_sparsity(simple_cost, start)

    _, jac_fn = jacobian(simple_cost, init_estimate=start, sparsity=pattern)

    np.testing.assert_allclose(dense(jac_fn(start)),
                               np.asarray(jax.jacfwd(simple_cost)(start)), atol=1e-5)


def test_jacobian_rejects_a_callable_sparsity():
    """The dynamic-sparsity pathway is not implemented; it must say so."""
    with pytest.raises(ValueError, match="Non-static sparsities"):
        jacobian(simple_cost, init_estimate=np.ones(4), sparsity=lambda p: None)


def test_jacobian_needs_something_to_work_from():
    with pytest.raises(ValueError, match="initial estimate"):
        jacobian(simple_cost)


def test_jacobian_can_be_used_as_a_decorator_factory():
    import jax

    start = np.array([1.0, 2.0, 3.0, 4.0])
    make = jacobian(init_estimate=start)

    fn, jac_fn = make(simple_cost)

    np.testing.assert_allclose(np.asarray(fn(start)), np.asarray(simple_cost(start)), atol=1e-6)
    np.testing.assert_allclose(dense(jac_fn(start)),
                               np.asarray(jax.jacfwd(simple_cost)(start)), atol=1e-5)


def test_mesh_residual_jacobian_is_block_sparse():
    """The structure ``point_to_plane_fit_test.py`` hand-built: each residual
    depends only on the parameters of the element it landed in."""
    rng = np.random.default_rng(0)
    mesh = bulged_patch()
    mesh.refine(2)
    points = rng.random((60, 3))

    def residual(params):
        return mesh.embed_points(points, fit_params=params, return_residual=True)[1].flatten()

    _, jac_fn = jacobian(residual, init_estimate=arr(mesh.optimisable_param_array))
    matrix = dense(jac_fn(arr(mesh.optimisable_param_array)))

    assert matrix.shape == (points.size, len(mesh.optimisable_param_array))
    density = np.count_nonzero(matrix) / matrix.size
    assert density < 0.5
