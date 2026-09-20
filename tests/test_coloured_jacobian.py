"""The coloured sparse Jacobians, against the dense one they stand in for.

:func:`~HOMER.jacobian_evaluator.make_jac_for_mesh_func` and its static
sibling reconstruct a Jacobian from one jvp per mesh colour rather than one
per parameter.  That is only worth anything if the result is the Jacobian, so
every test here compares against ``jax.jacfwd`` on the same residual.

The static maker freezes its sparsity at a probe point, which makes two things
worth pinning separately: that the frozen pattern still gives the right matrix
once the parameters and the per-call data have moved away from that point, and
that probe data which masks entries out costs them permanently -- the failure
its docstring warns about, which is silent in use.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

from HOMER.basis_definitions import L1
from HOMER.geometry import cube
from HOMER.jacobian_evaluator import (make_jac_for_mesh_func,
                                      make_static_jac_for_mesh_func)

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


@pytest.mark.parametrize("fields_separable", [False, True])
def test_static_jacobian_matches_the_dense_one(problem, fields_separable):
    mesh, residual, params, _, unit, _ = problem
    jac = make_static_jac_for_mesh_func(mesh, residual, params, fields_separable,
                                        further_args=unit)

    assert np.allclose(jac(params, **unit).toarray(),
                       dense_jac(residual, params, **unit), atol=EXACT)


@pytest.mark.parametrize("fields_separable", [False, True])
def test_dynamic_jacobian_matches_the_dense_one(problem, fields_separable):
    mesh, residual, params, _, unit, _ = problem
    jac = make_jac_for_mesh_func(mesh, residual, fields_separable)

    assert np.allclose(jac(params, **unit).toarray(),
                       dense_jac(residual, params, **unit), atol=EXACT)


def test_the_frozen_pattern_holds_as_params_and_data_move(problem):
    """The whole premise of the static maker: probe once, stay correct."""
    mesh, residual, params, moved, unit, data = problem
    jac = make_static_jac_for_mesh_func(mesh, residual, params, False, further_args=unit)

    for kwargs in data:
        for point in (params, moved):
            assert np.allclose(jac(point, **kwargs).toarray(),
                               dense_jac(residual, point, **kwargs), atol=EXACT)


def test_both_makers_return_something_scipy_can_use(problem):
    mesh, residual, params, _, unit, _ = problem
    n_res = residual(params, **unit).shape[0]

    for jac in (make_static_jac_for_mesh_func(mesh, residual, params, False, further_args=unit),
                make_jac_for_mesh_func(mesh, residual, False)):
        out = jac(params, **unit)
        assert isinstance(out, scipy.sparse.coo_array)
        assert out.shape == (n_res, params.shape[0])
        assert out.nnz > 0


def test_probe_data_that_masks_entries_loses_them_for_good(problem):
    """The silent failure the docstring warns about, made loud here."""
    mesh, residual, params, _, unit, _ = problem
    masked = {"targets": unit["targets"],
              "weights": jnp.zeros_like(unit["weights"]).at[:5].set(1.0)}

    honest = make_static_jac_for_mesh_func(mesh, residual, params, False, further_args=unit)
    starved = make_static_jac_for_mesh_func(mesh, residual, params, False, further_args=masked)

    assert starved(params, **unit).nnz < honest(params, **unit).nnz
    assert not np.allclose(starved(params, **unit).toarray(),
                           dense_jac(residual, params, **unit), atol=EXACT)
