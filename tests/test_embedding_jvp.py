"""Differentiating through ``embed_points``.

The embedding is an iterative solve, so JAX cannot differentiate it naively;
:mod:`HOMER.embedding` supplies a custom JVP.  ``test_custom_embedding_jvp.py``
printed a derivative and left the reader to decide whether it looked right,
and ``test_iterative_point_to_model.py`` demonstrated a per-element shortcut
for the same Jacobian without ever comparing it against the honest one.
Both comparisons are made here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

from HOMER import cube
from HOMER.basis_definitions import L2, L3
from HOMER.utils import rodrigues_exp

from _helpers import arr

TRUTH = np.array([0.15, -0.10, 0.20, 0.05, -0.03, 0.04])   #rotation vector, translation


def central_difference(fn, x, step=1e-3):
    x = np.asarray(x, dtype=float)
    columns = []
    for i in range(x.size):
        delta = np.zeros_like(x)
        delta[i] = step
        columns.append((np.asarray(fn(x + delta)) - np.asarray(fn(x - delta))) / (2 * step))
    return np.stack(columns, axis=-1)


############################################### derivative w.r.t. mesh parameters

def test_xi_moves_with_the_mesh_it_is_embedded_in():
    """Sliding every node along z must slide the recovered xi the other way."""
    mesh = cube(basis=[L3] * 3)
    base = jnp.array(np.asarray(mesh.true_param_array).reshape(-1, 3))

    def embedded_xi(shift):
        moved = base.at[:, -1].add(shift)
        _, xi = mesh.embed_points(jnp.array([0.3, 0.0, 0.0]), fit_params=moved.ravel())
        return xi.ravel()

    gradient = np.asarray(jax.jacfwd(embedded_xi)(0.0))

    np.testing.assert_allclose(gradient, central_difference(embedded_xi, np.array([0.0]))[:, 0],
                               atol=1e-3)
    #the point stays put while the mesh moves up, so its z-parameter falls
    assert gradient[2] < -0.5


def test_the_jvp_is_finite_and_not_silently_zero():
    """A missing custom rule shows up as a zero (or NaN) derivative."""
    mesh = cube(basis=[L2] * 3)
    base = jnp.array(np.asarray(mesh.true_param_array).reshape(-1, 3))

    def residual(shift):
        moved = base.at[:, 0].add(shift)
        _, res = mesh.embed_points(jnp.array([[2.0, 0.0, 0.0]]), fit_params=moved.ravel(),
                                   return_residual=True)
        return res.ravel()

    gradient = np.asarray(jax.jacfwd(residual)(0.0))

    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient).max() > 0.1


############################################### end-to-end rigid registration

@pytest.fixture(scope="module")
def registration_problem():
    mesh = cube(basis=[L2] * 3)
    surface = np.asarray(mesh.eval_surface(res=8))
    rotation = np.asarray(rodrigues_exp(jnp.array(TRUTH[:3])), dtype=float)
    target = jnp.asarray(surface @ rotation.T + TRUTH[3:])
    return mesh, target


def residual_fn(mesh, target):
    structure = jnp.asarray(mesh.optimisable_param_array)

    def residual(params):
        moved = structure.reshape(-1, 3) @ rodrigues_exp(params[:3]).T + params[3:]
        _, res = mesh.embed_points(target, fit_params=moved.ravel(), return_residual=True,
                                   surface_embed=True, grid_res=20, iterations=20)
        return res.ravel()

    return residual


def test_a_rigid_transform_is_recovered_by_differentiating_the_embedding(registration_problem):
    """The whole point of the custom JVP: gradient-based registration works."""
    mesh, target = registration_problem
    residual = residual_fn(mesh, target)

    result = least_squares(jax.jit(residual), np.zeros(6), verbose=0, max_nfev=60)

    np.testing.assert_allclose(result.x, TRUTH, atol=1e-4)
    assert result.cost < 1e-9


def test_the_analytic_jacobian_matches_finite_differences(registration_problem):
    mesh, target = registration_problem
    residual = residual_fn(mesh, target)
    probe = np.array([0.05, -0.02, 0.03, 0.01, 0.0, -0.01])

    analytic = np.asarray(jax.jacfwd(residual)(jnp.array(probe)))
    numeric = central_difference(residual, probe, step=1e-4)

    #a handful of samples sit on an element edge, where the surface projection
    #switches face and the derivative genuinely jumps -- a central difference
    #straddles that jump, so the comparison is on the bulk, not the worst case
    error = np.abs(analytic - numeric)
    assert np.percentile(error, 99) < 1e-3
    assert np.mean(error > 1e-2) < 0.005


############################################### the per-element shortcut

def test_a_per_element_jvp_reproduces_the_full_jacobian(registration_problem):
    """The trick from ``test_iterative_point_to_model.py``, now checked.

    Only the parameters of the element a point landed in can affect that
    point's residual, so the Jacobian can be assembled element-wise from a
    single-element "phantom" mesh instead of differentiating the whole solve.
    The phantom must be embedded exactly the way the real mesh was -- the
    original script used a plain volume embed against a ``surface_embed``
    solve, and the two answers have nothing to do with each other.
    """
    mesh, target = registration_problem
    structure = jnp.asarray(mesh.optimisable_param_array)
    probe = jnp.array([0.05, -0.02, 0.03, 0.01, 0.0, -0.01])
    embed_kwargs = dict(return_residual=True, surface_embed=True, grid_res=20, iterations=20)

    def moved_params(params):
        return structure.reshape(-1, 3) @ rodrigues_exp(params[:3]).T + params[3:]

    def residual(params):
        _, res = mesh.embed_points(target, fit_params=moved_params(params).ravel(), **embed_kwargs)
        return res.ravel()

    phantom = cube(basis=[L2] * 3)

    @jax.jit
    def one_point_jvp(point, params, tangent):
        def local(p):
            return phantom.embed_points(point, fit_params=p, **embed_kwargs)[1]
        return jax.vmap(lambda t: jax.jvp(local, (params,), (t,))[1],
                        in_axes=1, out_axes=-1)(tangent)

    ele_map = jnp.array(mesh.ele_map)
    node_jacobian = jax.jacfwd(moved_params)(probe)
    (elements, _) = mesh.embed_points(target, fit_params=moved_params(probe).ravel(),
                                      **embed_kwargs)[0]
    gather = ele_map[elements].astype(int)
    assembled = jax.vmap(one_point_jvp)(
        target,
        moved_params(probe).ravel()[gather],
        node_jacobian.reshape(-1, probe.shape[0])[gather],
    ).reshape(-1, probe.shape[0])

    reference = np.asarray(jax.jacfwd(residual)(probe))
    np.testing.assert_allclose(np.asarray(assembled), reference, atol=1e-5)


############################################### the closed-form JVP rule

"""The rule in `HOMER.embedding` writes the implicit-function-theorem
derivative out by hand rather than handing the stationarity condition to
`jax.jacobian` and `jax.jvp`.  Two identities make that possible, and both are
worth pinning: `H` splits into `W' M W` minus a curvature term, where `W` is
the Jacobian the refinement already computed at the converged point; and the
embedding is *linear* in the mesh parameters, so the parameter derivatives are
the same kernel evaluated at the tangent parameters rather than derivatives at
all.  Get either wrong and the derivative is quietly off, not broken -- so
these compare against central differences on cases where the truth is
computable.
"""


def _interior_problem(basis):
    """Points just off the mesh, well inside it, so the derivative is smooth.

    Points *outside* the mesh project onto a boundary, where xi is clamped and
    the derivative has a kink -- central differences across one are
    meaningless, so they are no use as a reference.
    """
    mesh = cube(basis=[basis] * 3)
    rng = np.random.default_rng(0)
    xi = rng.random((30, 3)) * 0.6 + 0.2
    on_mesh = arr(mesh.evaluate_embeddings_ele_xi_pair(np.zeros(30, int), xi))
    target = jnp.asarray(on_mesh + rng.standard_normal((30, 3)) * 0.01)
    base = jnp.asarray(np.asarray(mesh.true_param_array).reshape(-1, 3))
    return mesh, base, target


@pytest.mark.parametrize("basis", [L2, L3])
def test_the_parameter_derivative_matches_central_differences(basis):
    """Covers the linear-in-parameters identity for both w and its Jacobian.

    L2 and L3 tabulate no second derivative, so the curvature term has to come
    from differentiating the basis expressions -- these bases are the reason it
    is autodiffed rather than read from a table.
    """
    mesh, base, target = _interior_problem(basis)

    def embedded(shift):
        moved = base.at[:, 0].add(shift[0]).at[:, 1].add(shift[1]).at[:, 2].add(shift[2])
        (_, xi), res = mesh.embed_points(target, fit_params=moved.ravel(),
                                         return_residual=True, iterations=30, tol=0)
        return jnp.concatenate([xi.ravel(), res.ravel()])

    analytic = arr(jax.jacfwd(embedded)(jnp.zeros(3)))
    numeric = central_difference(embedded, np.zeros(3), step=2e-3)

    np.testing.assert_allclose(analytic, numeric, atol=2e-3)
    assert np.abs(analytic).max() > 0.1   #not silently zero


def test_the_point_derivative_matches_central_differences():
    """dxi/dx and dr/dx go through the same H, but with the x_dot term."""
    mesh, _, target = _interior_problem(L2)
    params = jnp.asarray(mesh.optimisable_param_array)

    def embedded(shift):
        moved = target + jnp.asarray(shift)
        (_, xi), res = mesh.embed_points(moved, fit_params=params,
                                         return_residual=True, iterations=30, tol=0)
        return jnp.concatenate([xi.ravel(), res.ravel()])

    analytic = arr(jax.jacfwd(embedded)(jnp.zeros(3)))
    numeric = central_difference(embedded, np.zeros(3), step=2e-3)

    np.testing.assert_allclose(analytic, numeric, atol=2e-3)
    assert np.abs(analytic).max() > 0.1


@pytest.mark.parametrize("n_active", [2, 1, 0])
def test_a_masked_residual_still_differentiates(n_active):
    """`dim_mask` under-determines xi; at n_active=0 H is exactly singular.

    Not checked against central differences, deliberately: with fewer active
    dimensions than parametric ones, every point of a null manifold is an
    equally good minimum, so xi is not a function of the parameters and the
    two finite-difference evaluations are free to land on different answers
    (they do -- xi moves by 0.05 for a 2e-3 step).  What *is* well defined is
    that the derivative exists, that the masked-off residual components carry
    no derivative at all, and that the wholly degenerate case comes back a
    clean zero rather than the NaN a plain solve would give.
    """
    mesh, base, target = _interior_problem(L2)
    dim_mask = np.array([True] * n_active + [False] * (3 - n_active))

    def embedded(shift):
        moved = base.at[:, 0].add(shift[0])
        (_, xi), res = mesh.embed_points(target, fit_params=moved.ravel(),
                                         return_residual=True, dim_mask=dim_mask,
                                         iterations=30, tol=0)
        return res

    analytic = arr(jax.jacfwd(embedded)(jnp.zeros(1)))[..., 0]

    assert np.all(np.isfinite(analytic))
    #masked components are documented to be exactly zero in the Jacobian
    np.testing.assert_array_equal(analytic[:, n_active:],
                                  np.zeros_like(analytic[:, n_active:]))
    if n_active == 0:
        np.testing.assert_array_equal(analytic, np.zeros_like(analytic))


def test_the_returned_jacobian_is_taken_at_the_converged_point():
    """The rule reuses the refinement's Jacobian, so it must track the answer.

    Checked through the derivative it feeds: `dr/dx = m - W dxi/dx`, an
    identity that only holds if W is dw/dxi at the *same* (elem, xi) the solve
    returned.  A Jacobian left over from a rejected proposal would break it.
    """
    mesh, _, target = _interior_problem(L3)
    params = jnp.asarray(mesh.optimisable_param_array)

    def solve(pts):
        (ele, xi), res = mesh.embed_points(pts, fit_params=params,
                                           return_residual=True, iterations=30, tol=0)
        return xi, res, ele

    ele = np.asarray(solve(target)[2])
    d_xi = arr(jax.jacfwd(lambda p: solve(p)[0])(target))
    d_res = arr(jax.jacfwd(lambda p: solve(p)[1])(target))
    W = arr(mesh.evaluate_jacobians_ele_xi_pair(ele, np.asarray(solve(target)[0])))

    for i in range(len(ele)):
        #dr_i/dx_i = I - W_i dxi_i/dx_i
        predicted = np.eye(3) - W[i] @ d_xi[i, :, i, :]
        np.testing.assert_allclose(d_res[i, :, i, :], predicted, atol=1e-4)
