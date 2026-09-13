"""Fitting a mesh to data: the linear solve, the nonlinear solve, and the
sparse Jacobian machinery underneath both.

``linear_fit_mesh_test.py``, ``optimise_mesh_test.py`` and
``point_to_plane_fit_test.py`` all ended by drawing the fitted mesh over the
target cloud.  A fit either reaches the target or it does not, and when the
target is representable in the fitting basis the answer is exact -- so that
is what is asserted here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from HOMER.basis_definitions import B3, H3, L1, L2, L3
from HOMER.fitting import point_cloud_fit
from HOMER.geometry import basic_surface, cube
from HOMER.jacobian_evaluator import (estimate_column_norms, estimate_sparsity,
                                      jacobian, make_jac_for_mesh_func,
                                      make_static_jac_for_mesh_func,
                                      matrix_free_jacobian)
from HOMER.mesh import column_equilibrated_lstsq, sparse_equilibrated_lstsq

from _helpers import CLOSE, EXACT, arr
from HOMER.examples import bulged_patch, unit_hex


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


def test_the_weight_blocks_scatter_to_the_weight_matrix():
    """``get_xi_weight_blocks`` is ``get_xi_weight_mat`` before the scatter."""
    mesh = bulged_patch()
    grid = mesh.xi_grid(5)
    eles = np.zeros(len(grid), dtype=int)

    weights, columns = mesh.get_xi_weight_blocks(eles, grid)
    scattered = np.zeros_like(arr(mesh.get_xi_weight_mat(eles, grid)))
    np.add.at(scattered, (np.repeat(np.arange(len(grid)), columns.shape[1]),
                          np.asarray(columns).ravel()), arr(weights).ravel())

    np.testing.assert_allclose(scattered, arr(mesh.get_xi_weight_mat(eles, grid)), atol=EXACT)


@pytest.mark.parametrize('basis', [[L1] * 3, [H3] * 3, [B3] * 3],
                         ids=['L1', 'H3', 'B3'])
def test_the_sparse_solve_fits_at_least_as_well_as_the_dense_one(basis):
    """The system a mesh builds is block-sparse, and solving it sparse is not a
    compromise: it minimises the same residual, in float64 rather than float32,
    so it lands no further from the targets than the dense solve does.

    Stated as a comparison of residuals rather than of parameters, because for
    a rank-deficient system the two tie-break differently while fitting the
    same geometry -- which is the distinction ``column_equilibrated_lstsq``'s
    own docstring draws.
    """
    mesh = cube(basis=[L1] * 3)
    mesh.refine(2)
    mesh = mesh.rebase(basis)

    grid = mesh.xi_grid(4)
    eles = np.repeat(np.arange(len(mesh.elements)), len(grid))
    xis = np.tile(grid, (len(mesh.elements), 1))
    targets = arr(mesh.evaluate_embeddings_ele_xi_pair(eles, xis))

    W = arr(mesh.get_xi_weight_mat(eles, xis))
    weights, columns = mesh.get_xi_weight_blocks(eles, xis)
    n_cols = mesh.true_param_array.shape[0] // mesh.fdim

    p_dense = arr(column_equilibrated_lstsq(W, targets)[0])
    p_sparse = arr(sparse_equilibrated_lstsq(weights, columns, n_cols, targets))

    assert p_sparse.shape == p_dense.shape
    residual = lambda p: np.linalg.norm(W @ p - targets)
    assert residual(p_sparse) <= residual(p_dense) + EXACT


def test_the_sparse_solve_differentiates_like_the_dense_one():
    """A refinement can sit inside a loss function.

    Only the square solve leaves JAX, inside ``jax.lax.custom_linear_solve``,
    so JAX derives the JVP and the transpose from its own rule rather than
    from anything written here.  What that has to buy is agreement with the
    dense path, which JAX differentiates natively -- for the targets, and for
    the weights, which enter through the closure rather than the residual.
    """
    rng = np.random.default_rng(1)
    n_rows, K, n_cols = 300, 6, 30
    columns = jnp.asarray(rng.integers(0, n_cols, (n_rows, K)))
    weights = jnp.asarray(rng.random((n_rows, K)) + 0.5, dtype=jnp.float32)
    targets = jnp.asarray(rng.random((n_rows, 3)), dtype=jnp.float32)

    def densify(w):
        rows = jnp.repeat(jnp.arange(n_rows), K)
        return jnp.zeros((n_rows, n_cols)).at[rows, columns.ravel()].add(w.ravel())

    sparse = lambda w, b: (sparse_equilibrated_lstsq(w, columns, n_cols, b) ** 2).sum()
    dense = lambda w, b: (column_equilibrated_lstsq(densify(w), b)[0] ** 2).sum()

    np.testing.assert_allclose(sparse(weights, targets), dense(weights, targets), atol=EXACT)
    for argnums in (0, 1):
        np.testing.assert_allclose(arr(jax.grad(sparse, argnums=argnums)(weights, targets)),
                                   arr(jax.grad(dense, argnums=argnums)(weights, targets)),
                                   atol=EXACT)


def test_the_sparse_solve_agrees_forwards_backwards_and_under_jit():
    """Both modes come from one operator, so they must not disagree.

    ``jax.lax.custom_linear_solve`` derives the JVP and the transpose from the
    augmented system itself rather than from a rule written here, so the check
    is the defining identity between them: the pairing ``<w, J v>`` taken
    forwards has to equal ``<J.T w, v>`` taken backwards.
    """
    rng = np.random.default_rng(2)
    n_rows, K, n_cols = 200, 4, 20
    columns = jnp.asarray(rng.integers(0, n_cols, (n_rows, K)))
    weights = jnp.asarray(rng.random((n_rows, K)) + 0.5, dtype=jnp.float32)
    targets = jnp.asarray(rng.random((n_rows, 2)), dtype=jnp.float32)

    solve = lambda b: sparse_equilibrated_lstsq(weights, columns, n_cols, b)

    np.testing.assert_allclose(arr(jax.jit(solve)(targets)), arr(solve(targets)), atol=EXACT)
    np.testing.assert_allclose(arr(jax.jit(jax.grad(lambda b: solve(b).sum()))(targets)),
                               arr(jax.grad(lambda b: solve(b).sum())(targets)), atol=EXACT)

    tangent = jnp.asarray(rng.standard_normal((n_rows, 2)), dtype=jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((n_cols, 2)), dtype=jnp.float32)
    forwards = (cotangent * jax.jvp(solve, (targets,), (tangent,))[1]).sum()
    backwards = (jax.vjp(solve, targets)[1](cotangent)[0] * tangent).sum()
    np.testing.assert_allclose(arr(forwards), arr(backwards), rtol=1e-5)


def test_a_one_dimensional_target_comes_back_one_dimensional():
    """A secondary field fits a scalar per point, and refines through here too."""
    rng = np.random.default_rng(3)
    n_rows, K, n_cols = 120, 4, 15
    columns = jnp.asarray(rng.integers(0, n_cols, (n_rows, K)))
    weights = jnp.asarray(rng.random((n_rows, K)) + 0.5, dtype=jnp.float32)
    targets = jnp.asarray(rng.random(n_rows), dtype=jnp.float32)

    flat = sparse_equilibrated_lstsq(weights, columns, n_cols, targets)
    column = sparse_equilibrated_lstsq(weights, columns, n_cols, targets[:, None])

    assert flat.shape == (n_cols,)
    np.testing.assert_allclose(arr(flat), arr(column)[:, 0], atol=EXACT)


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


@pytest.mark.parametrize("basis,improvement", [(L2, 1), (H3, 10), (B3, 10)],
                         ids=lambda x: getattr(x, '__name__', x))
def test_equilibration_recovers_precision_on_a_real_weight_matrix(basis, improvement):
    """The matrices HOMER actually builds are why this exists.

    A refined Hermite mesh reaches ``cond(W) = 1.1e4`` and a B-spline control
    net 4.1e4 -- the derivative and control-point weights are an order of
    magnitude smaller than the value weights, cubed over three directions.
    Recovering known parameters through such a matrix in float32 is where the
    scaling pays; on a well-conditioned Lagrange matrix it is a wash -- so the
    Lagrange case only asks that equilibration stay in the same order of
    magnitude.  Equilibration cannot make a well-conditioned system worse in
    any way that matters, but which of the two lands ahead is decided by
    float32 summation order, and that moves whenever the mesh renumbers its
    nodes or the BLAS underneath changes.
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

    slack = 10.0 if improvement == 1 else 1.0
    assert equilibrated <= plain / improvement * slack


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
    target = unit_hex(basis=[L1] * 3)
    fitted = unit_hex(basis=[L3] * 3)
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
    fitted = basic_surface(basis=[L3] * 2)
    fitted.refine(2)

    res = 8
    eles = np.repeat(np.arange(len(fitted.elements)), res ** 2)
    grid = np.tile(fitted.xi_grid(res), (len(fitted.elements), 1))
    targets = arr(target.evaluate_embeddings_ele_xi_pair(eles, grid))

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    got = arr(fitted.evaluate_embeddings_ele_xi_pair(eles, grid))
    np.testing.assert_allclose(got, targets, atol=CLOSE)


def test_linear_fit_return_params_does_not_touch_the_mesh():
    mesh = unit_hex(basis=[L2] * 3)
    mesh.nodes[0].fix_parameter('loc')
    mesh.generate_mesh()
    grid = mesh.xi_grid(5)
    eles = np.zeros(len(grid), dtype=int)
    before = arr(mesh.true_param_array)

    params = mesh.linear_fit(np.zeros((len(grid), 3)),
                            weight_mat=mesh.get_xi_weight_mat(eles, grid),
                            return_params=True)

    assert params is not None
    np.testing.assert_allclose(arr(mesh.true_param_array), before, atol=EXACT)
    #the full vector, not the optimisable subset, with the held node carried
    #through rather than left as a gap
    assert params.shape == before.shape
    np.testing.assert_allclose(arr(params)[:3], before[:3], atol=EXACT)


def constrained_minimum(mesh, weights, targets):
    """The constrained least-squares answer, written out in plain numpy.

    Holding a parameter is the same problem with fewer columns and a corrected
    right-hand side, one system per component because the free set is per
    component.  Short enough to say twice, and saying it twice is the point --
    the reference must not share the solve under test.
    """
    free = mesh.optimisable_param_bool.reshape(-1, mesh.fdim)
    held = np.where(free, 0.0, arr(mesh.true_param_array).reshape(-1, mesh.fdim))
    expected = held.copy()
    for d in range(mesh.fdim):
        cols = free[:, d]
        expected[cols, d] = np.linalg.lstsq(weights[:, cols],
                                            targets[:, d] - weights @ held[:, d],
                                            rcond=None)[0]
    return expected


def constrained_patch(*fixings, basis=L2):
    """A patch to fit, a target to fit it to, and the system between them."""
    fitted = basic_surface(basis=[basis] * 2)
    for node, kwargs in fixings:
        fitted.nodes[node].fix_parameter('loc', **kwargs)
    fitted.generate_mesh()

    grid = fitted.xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)
    targets = arr(bulged_patch().evaluate_embeddings_ele_xi_pair(eles, grid))
    return fitted, eles, grid, targets


def test_linear_fit_holds_fixed_parameters():
    """A fixed parameter keeps its value, and the free ones fit around it.

    The reverse of what this file used to assert: the solve fitted every
    column and the constraint was overwritten by the answer it came back with.
    """
    fitted, eles, grid, targets = constrained_patch((0, {}))
    pinned = arr(fitted.nodes[0].loc)

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    np.testing.assert_array_equal(arr(fitted.nodes[0].loc), pinned)


def test_linear_fit_finds_the_constrained_minimum():
    """Not merely that the held value survived -- the free parameters have to
    be the best they can be given it, which is a different fit from the
    unconstrained one truncated."""
    fitted, eles, grid, targets = constrained_patch((0, {}), (4, {}))
    weights = arr(fitted.get_xi_weight_mat(eles, grid))

    fitted.linear_fit(targets, weight_mat=weights)

    np.testing.assert_allclose(arr(fitted.true_param_array).reshape(-1, 3),
                               constrained_minimum(fitted, weights, targets), atol=CLOSE)


def test_linear_fit_holds_single_components():
    """``fix_parameter`` takes component indices, so the free set can differ
    between the components of the field.  The weight matrix is shared across
    them, so this is the case that forces the solve to group the components
    instead of running one system -- three groups here.
    """
    #node 0 is the corner the bulge lifts in z, node 4 the centre it pushes
    #out in x; both hold the one component the target would have moved
    fitted, eles, grid, targets = constrained_patch((0, {'inds': [2]}), (4, {'inds': [0]}))
    before = arr(fitted.true_param_array).reshape(-1, 3)
    weights = arr(fitted.get_xi_weight_mat(eles, grid))

    fitted.linear_fit(targets, weight_mat=weights)

    after = arr(fitted.true_param_array).reshape(-1, 3)
    assert after[0, 2] == before[0, 2]
    assert after[4, 0] == before[4, 0]
    np.testing.assert_allclose(after, constrained_minimum(fitted, weights, targets), atol=CLOSE)
    #and the components left free still move -- the fit is constrained, not frozen
    assert np.abs(after[6, 2] - before[6, 2]) > 0.5


def test_a_held_parameter_does_not_drift():
    """Held, not re-fitted.  The solve around it runs in float32, so a value
    that is carried through it comes back changed; this one must not."""
    pinned = np.array([0.3, 0.1234567890123, 0.7])   #none of it exact in float32
    fitted, eles, grid, targets = constrained_patch((0, {'values': pinned}))

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    np.testing.assert_array_equal(arr(fitted.nodes[0].loc), pinned)


def test_linear_fit_holds_fixed_parameters_through_the_sparse_path():
    """The block form cannot drop a column out of a rectangular block, so the
    held entries are zeroed and folded into the right-hand side instead.  That
    has to land where the dense solve lands."""
    fixings = ((0, {'inds': [2]}), (4, {}))
    dense, eles, grid, targets = constrained_patch(*fixings)
    dense.linear_fit(targets, weight_mat=dense.get_xi_weight_mat(eles, grid))

    sparse, eles, grid, targets = constrained_patch(*fixings)
    weights, columns = sparse.get_xi_weight_blocks(eles, grid)
    sparse.linear_fit(targets, weight_mat=weights, sparse_columns=columns)

    np.testing.assert_allclose(arr(sparse.true_param_array), arr(dense.true_param_array),
                               atol=CLOSE)


def test_linear_fit_with_every_parameter_fixed_leaves_the_mesh_alone():
    """No free columns anywhere: there is nothing to solve, and the fit is a
    no-op rather than a degenerate system."""
    fitted, eles, grid, targets = constrained_patch(
        *[(n, {}) for n in range(9)])
    before = arr(fitted.true_param_array)

    fitted.linear_fit(targets, weight_mat=fitted.get_xi_weight_mat(eles, grid))

    np.testing.assert_array_equal(arr(fitted.true_param_array), before)


def test_linear_fit_rejects_an_underdetermined_system():
    mesh = unit_hex(basis=[L3] * 3)
    grid = mesh.xi_grid(2)                       #8 samples, 64 unknowns
    eles = np.zeros(len(grid), dtype=int)

    with pytest.raises(AssertionError, match="undertederimined"):
        mesh.linear_fit(arr(mesh.evaluate_embeddings_ele_xi_pair(eles, grid)),
                        weight_mat=mesh.get_xi_weight_mat(eles, grid))


def test_linear_fit_ignores_rows_marked_empty():
    """Rows equal to ``target_empty`` drop out of the solve."""
    target = unit_hex(basis=[L1] * 3)
    grid = unit_hex(basis=[L2] * 3).xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)
    targets = arr(target.evaluate_embeddings_ele_xi_pair(eles, grid))

    fitted = unit_hex(basis=[L2] * 3)
    weights = arr(fitted.get_xi_weight_mat(eles, grid))
    fitted.linear_fit(targets, weight_mat=weights)
    reference = arr(fitted.true_param_array)

    spoiled = np.concatenate([targets, np.full((20, 3), -1.0)])
    spoiled_weights = np.concatenate([weights, np.zeros((20, weights.shape[1]))])
    with_junk = unit_hex(basis=[L2] * 3)
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
    mesh = basic_surface(basis=[H3] * 2)
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
    mesh = basic_surface(basis=[H3] * 2)
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
    mesh = basic_surface(basis=[H3] * 2)
    n_sobolev = arr(mesh.evaluate_sobolev()).size
    start = arr(mesh.optimisable_param_array)

    fit_fn, jac_fn = point_cloud_fit(mesh, curved_target, compile=True, sob_weight=0.01)

    block = dense(jac_fn(start))[-n_sobolev:]
    assert np.abs(block).sum() > 0
    #and the residual block responds to the parameters it is handed
    assert not np.allclose(np.asarray(fit_fn(start))[-n_sobolev:],
                           np.asarray(fit_fn(start * 1.5))[-n_sobolev:])


def test_sobolev_weight_changes_the_cost(curved_target):
    mesh = basic_surface(basis=[H3] * 2)
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


############################################### the coloured jacobians


def grid_residual(mesh, res=3):
    """``evaluate_embeddings`` on an xi grid: one element per residual entry."""
    eles = np.arange(len(mesh.elements))
    xis = mesh.xi_grid(res)

    def residual(params):
        return jnp.ravel(mesh.evaluate_embeddings(eles, xis, fit_params=params))

    return residual


def reembedding_residual(mesh, points):
    """A data-to-model term.  ``approx_jac`` holds each point at the
    ``(element, xi)`` it embedded to, which is what keeps the residual
    separable in fields; which element that is still moves with the mesh, so
    the sparsity pattern moves with it."""
    def residual(params):
        return mesh.embed_points(points, fit_params=params, return_residual=True,
                                 approx_jac=True)[1].flatten()

    return residual


@pytest.mark.parametrize("basis", [L1, H3], ids=lambda b: b.__name__)
def test_the_coloured_jacobians_match_the_dense_one(basis):
    """Both read the pattern off the topology instead of probing for it, so
    both have to land exactly where ``jax.jacfwd`` does."""
    mesh = cube(basis=[basis] * 3)
    mesh.refine(2)
    residual = grid_residual(mesh)
    start = jnp.asarray(mesh.optimisable_param_array)
    reference = np.asarray(jax.jacfwd(residual)(start))

    dynamic = make_jac_for_mesh_func(mesh, residual, fields_seperable=True)
    static = make_static_jac_for_mesh_func(mesh, residual, start, fields_seperable=True)

    np.testing.assert_allclose(np.asarray(dynamic(start).todense()), reference, atol=CLOSE)
    np.testing.assert_allclose(np.asarray(static(start).todense()), reference, atol=CLOSE)


def test_a_static_pattern_holds_away_from_where_it_was_read():
    """Fixed ``(element, xi)`` sampling: the pattern is the same everywhere, so
    freezing it costs nothing."""
    mesh = cube(basis=[H3] * 3)
    mesh.refine(2)
    residual = grid_residual(mesh)
    start = jnp.asarray(mesh.optimisable_param_array)
    moved = start + jnp.asarray(
        np.random.default_rng(0).normal(scale=0.05, size=start.shape))

    static = make_static_jac_for_mesh_func(mesh, residual, start, fields_seperable=True)

    np.testing.assert_allclose(np.asarray(static(moved).todense()),
                               np.asarray(jax.jacfwd(residual)(moved)), atol=CLOSE)


def test_only_the_dynamic_jacobian_follows_a_pattern_that_moves():
    """The reason the second pass is worth paying for: once the data re-embeds
    into different elements, the frozen indices are the wrong indices."""
    rng = np.random.default_rng(0)
    mesh = bulged_patch()
    mesh.refine(2)
    residual = reembedding_residual(mesh, jnp.asarray(rng.random((60, 3))))

    start = jnp.asarray(mesh.optimisable_param_array)
    moved = start + jnp.asarray(rng.normal(scale=0.25, size=start.shape))
    reference = np.asarray(jax.jacfwd(residual)(moved))

    dynamic = make_jac_for_mesh_func(mesh, residual, fields_seperable=True)
    static = make_static_jac_for_mesh_func(mesh, residual, start, fields_seperable=True)

    np.testing.assert_allclose(np.asarray(dynamic(moved).todense()), reference, atol=CLOSE)
    assert np.abs(np.asarray(static(moved).todense()) - reference).max() > 0.1


def sparsejac_colour_count(residual, start):
    """How wide sparsejac compresses the same Jacobian, by its own colouring of
    the probed pattern -- the work ``jacobian(sparse=True)`` does internally."""
    from sparsejac.sparsejac import _greedy_color, _input_connectivity_from_sparsity
    import scipy.sparse

    pattern = estimate_sparsity(jax.jit(residual), start)
    as_scipy = scipy.sparse.coo_matrix(
        (np.asarray(pattern.data), np.asarray(pattern.indices).T), shape=pattern.shape)
    return _greedy_color(_input_connectivity_from_sparsity(as_scipy), "largest_first")[1]


@pytest.mark.parametrize("basis", [L1, L2, H3], ids=lambda b: b.__name__)
def test_the_mesh_colouring_is_no_wider_than_the_probed_one(basis):
    """A colouring is only worth having if it compresses as hard as the one
    sparsejac derives from the probed pattern, and on a residual that samples
    every element the two come out equal.  Asserted as ``<=`` so that the guard
    is on HOMER's colouring getting worse, not on sparsejac's heuristic staying
    put.  What the mesh buys is the probe itself: sparsejac pays one residual
    evaluation per parameter to find the pattern, the element map is already
    there.
    """
    mesh = cube(basis=[basis] * 3)
    mesh.refine(2)
    #4 samples per direction: an L2 grid of 3 lands on the nodes, where every
    #basis but one is zero, and the pattern degenerates to one entry per row
    residual = grid_residual(mesh, res=4)
    start = jnp.asarray(mesh.optimisable_param_array)

    separable = max(mesh.get_colouring_dict(fields_seperable=True).values()) + 1

    assert separable <= sparsejac_colour_count(residual, start)


def test_the_colouring_jacobian_covers_only_optimisable_parameters():
    mesh = cube(basis=[L1] * 3)
    mesh.refine(2)
    mesh.nodes[0].fix_parameter('loc')
    mesh.generate_mesh()
    residual = grid_residual(mesh)
    start = jnp.asarray(mesh.optimisable_param_array)

    jac = make_jac_for_mesh_func(mesh, residual, fields_seperable=True)(start)

    assert jac.shape == (len(residual(start)), len(mesh.optimisable_param_array))
    np.testing.assert_allclose(np.asarray(jac.todense()),
                               np.asarray(jax.jacfwd(residual)(start)), atol=CLOSE)

############################################### the matrix-free jacobian

def hermite_weights(refine=2, basis=H3):
    """A weight matrix whose column norms span orders of magnitude.

    The derivative dofs of a Hermite basis carry weights an order of magnitude
    below the value dofs', cubed over three directions -- which is the whole
    reason the matrix-free operator has to be preconditioned.
    """
    mesh = unit_hex(basis=[basis] * 3)
    mesh.refine(refine)
    grid = mesh.xi_grid(basis.order + 2)
    eles = np.repeat(np.arange(len(mesh.elements)), len(grid))
    return np.asarray(mesh.get_xi_weight_mat(eles, np.tile(grid, (len(mesh.elements), 1))))


def representable_system(refine=2):
    """``cost(p) = W p - W p_true``: a target the basis reaches exactly.

    The reachable *residual* is zero by construction, which is what the fit is
    scored on here.  The parameters are not: ``W`` is one column short of full
    rank, so a null-space direction is free and ``p_true`` itself is only one
    of the answers.
    """
    weights = hermite_weights(refine)
    weights_j = jnp.asarray(weights)
    p_true = jnp.asarray(np.random.default_rng(0).random(weights.shape[1]), jnp.float32)
    target = weights_j @ p_true

    def cost(params):
        return weights_j @ params - target

    return cost, weights, np.asarray(p_true)


def test_the_operator_applies_the_dense_jacobian():
    """matvec and rmatvec against ``jax.jacfwd``: the independent code path."""
    import jax

    rng = np.random.default_rng(1)
    start = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    reference = np.asarray(jax.jacfwd(simple_cost)(start))

    _, jac_operator, scale = matrix_free_jacobian(simple_cost, start, precondition=False)
    operator = jac_operator(start)

    np.testing.assert_allclose(scale, np.ones(4), atol=EXACT)
    v, w = rng.random(4).astype(np.float32), rng.random(3).astype(np.float32)
    np.testing.assert_allclose(operator.matvec(v), reference @ v, atol=1e-5)
    np.testing.assert_allclose(operator.rmatvec(w), reference.T @ w, atol=1e-5)


def test_the_scaled_operator_stays_its_own_adjoint():
    """``<Jv, w> == <v, J^T w>`` -- a scaling applied to one side alone is
    otherwise silent, and lsmr would simply converge to the wrong step."""
    rng = np.random.default_rng(2)
    cost, weights, p_true = representable_system(refine=1)
    n_res, n_par = weights.shape

    _, jac_operator, scale = matrix_free_jacobian(cost, np.zeros(n_par, np.float32))
    operator = jac_operator(p_true / scale)

    v = rng.standard_normal(n_par).astype(np.float32)
    w = rng.standard_normal(n_res).astype(np.float32)

    np.testing.assert_allclose(operator.matvec(v) @ w, v @ operator.rmatvec(w),
                               rtol=1e-4)


def test_probes_price_the_columns_they_cannot_form():
    """The estimator against the norms of the assembled matrix it replaces."""
    cost, weights, _ = representable_system(refine=1)
    n_res, n_par = weights.shape
    vjp = lambda w: jnp.asarray(weights.T) @ w

    estimated = estimate_column_norms(vjp, n_res, probes=256, dtype=np.float32)

    true_norms = np.linalg.norm(weights, axis=0)
    np.testing.assert_allclose(estimated / true_norms, 1.0, rtol=0.25)


def test_the_scaling_flattens_the_column_norms():
    """What the preconditioner is for: unit columns out of a Hermite basis."""
    cost, weights, _ = representable_system(refine=1)
    true_norms = np.linalg.norm(weights, axis=0)

    _, _, scale = matrix_free_jacobian(cost, np.zeros(weights.shape[1], np.float32),
                                       probes=256)

    assert true_norms.max() / true_norms.min() > 100
    np.testing.assert_allclose(true_norms * scale, 1.0, rtol=0.25)


def test_a_dead_column_keeps_unit_scale():
    """A parameter no residual touches must not be scaled by 1/0."""
    def cost(params):
        return jnp.stack([params[0], params[2] * 2.0])

    _, jac_operator, scale = matrix_free_jacobian(cost, np.zeros(3, np.float32))

    assert np.isfinite(scale).all()
    assert scale[1] == 1.0


def test_lsmr_cannot_solve_the_subproblem_unpreconditioned():
    """The end-to-end claim, on a target the basis reaches exactly.

    Both solves get the same budget -- the same residual evaluations, the same
    lsmr iterations per trust-region subproblem.  Preconditioned, that budget
    drives the residual to the float32 floor; on the raw operator the
    subproblem is never solved well enough to take a Newton-sized step, and the
    fit is still three orders of magnitude short when the budget runs out.
    """
    cost, weights, _ = representable_system()
    start = np.zeros(weights.shape[1], np.float32)
    budget = dict(max_nfev=20, tr_solver='lsmr', tr_options=dict(maxiter=10))

    def remaining_cost(precondition):
        fwd, jac, scale = matrix_free_jacobian(cost, start, precondition=precondition)
        return least_squares(fwd, start / scale, jac=jac, **budget).cost

    preconditioned = remaining_cost(precondition=True)

    assert preconditioned < 1e-4
    assert remaining_cost(precondition=False) > 100 * preconditioned


def test_the_operator_takes_what_scipy_hands_it():
    """``least_squares`` probes the operator with an int8 vector, and lsmr
    passes ``(n, 1)`` columns.  JAX rejects both, so the operator normalises."""
    start = np.array([1.0, 2.0, 3.0, 4.0], np.float32)

    _, jac_operator, _ = matrix_free_jacobian(simple_cost, start)
    operator = jac_operator(start)

    assert operator.dtype == np.float32
    assert operator.matvec(np.ones(4, np.int8)).shape == (3,)
    assert operator.matvec(np.ones((4, 1))).shape == (3, 1)     #scipy re-columns it
    assert operator.rmatvec(np.ones((3, 1))).shape == (4, 1)
