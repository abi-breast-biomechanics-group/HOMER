"""Properties every 1-D basis must satisfy.

These are the checks that used to be implicit: a mesh "looked wrong" when a
basis was wrong.  Stated directly they are cheap, and they localise a fault
to the basis rather than to the mesh built on it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from HOMER.basis_definitions import (B3Basis, H3Basis, L1Basis, L2Basis,
                                     L3Basis, L4Basis)

ALL_BASES = [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis, B3Basis]
LAGRANGE = [L1Basis, L2Basis, L3Basis, L4Basis]

SAMPLE = jnp.linspace(0.0, 1.0, 11)


def tol(basis):
    """Everything here is exact in exact arithmetic; the slack is float32.

    The quartic weights are the worst-conditioned of the set -- they carry
    coefficients into the hundreds that then cancel -- and lose about two
    digits more than the rest.
    """
    return 1e-4 if basis is L4Basis else 1e-6


@pytest.mark.parametrize("basis", ALL_BASES, ids=lambda b: b.__name__)
def test_weight_names_match_the_evaluated_columns(basis):
    assert basis.fn(SAMPLE).shape == (len(SAMPLE), len(basis.weights))


@pytest.mark.parametrize("basis", LAGRANGE + [B3Basis], ids=lambda b: b.__name__)
def test_partition_of_unity(basis):
    """A constant field must be reproduced exactly, at every xi."""
    np.testing.assert_allclose(np.asarray(basis.fn(SAMPLE)).sum(-1), 1.0, atol=tol(basis))


def test_hermite_splits_into_value_and_derivative_weights():
    """H3 is not a partition of unity; only its two value weights are."""
    weights = np.asarray(H3Basis.fn(SAMPLE))
    value_cols = [i for i, w in enumerate(H3Basis.weights) if not w.startswith('d')]
    deriv_cols = [i for i, w in enumerate(H3Basis.weights) if w.startswith('d')]

    np.testing.assert_allclose(weights[:, value_cols].sum(-1), 1.0, atol=1e-6)
    #the derivative weights vanish at both ends, so nodal values alone fix the endpoints
    np.testing.assert_allclose(weights[[0, -1]][:, deriv_cols], 0.0, atol=1e-6)


@pytest.mark.parametrize("basis", LAGRANGE, ids=lambda b: b.__name__)
def test_lagrange_bases_are_interpolatory(basis):
    """Evaluating at the node locations must give the identity."""
    at_nodes = np.asarray(basis.fn(jnp.array(basis.node_locs)))
    np.testing.assert_allclose(at_nodes, np.eye(len(basis.node_locs)), atol=1e-6)
    assert basis.interpolatory


def test_hermite_is_interpolatory_in_value_and_slope():
    at_nodes = np.asarray(H3Basis.fn(jnp.array([0.0, 1.0])))
    slope_at_nodes = np.asarray(H3Basis.deriv[1](jnp.array([0.0, 1.0])))

    #columns are [x0, dx0, x1, dx1]
    np.testing.assert_allclose(at_nodes, [[1, 0, 0, 0], [0, 0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(slope_at_nodes, [[0, 1, 0, 0], [0, 0, 0, 1]], atol=1e-6)
    assert H3Basis.interpolatory


def test_b3_is_flagged_as_a_control_net():
    """B3 parameters are control points, not values at ``node_locs``.

    Refinement relies on this flag to decide whether a fixed nodal value may
    be carried across verbatim, so it is worth pinning.
    """
    assert not B3Basis.interpolatory
    at_nodes = np.asarray(B3Basis.fn(jnp.array([0.0, 1.0])))
    assert np.abs(at_nodes - np.eye(4)[:2]).max() > 0.1


@pytest.mark.parametrize("basis", ALL_BASES, ids=lambda b: b.__name__)
@pytest.mark.parametrize("order", [1, 2])
def test_tabulated_derivatives_agree_with_autodiff(basis, order):
    if order >= len(basis.deriv):
        pytest.skip(f"{basis.__name__} tabulates only {len(basis.deriv) - 1} derivative(s)")

    def one_point(t):
        return basis.fn(jnp.atleast_1d(t))[0]

    autodiff = one_point
    for _ in range(order):
        autodiff = jax.jacfwd(autodiff)

    for x in (0.0, 0.13, 0.5, 0.87, 1.0):
        np.testing.assert_allclose(np.asarray(basis.deriv[order](jnp.array([x]))[0]),
                                   np.asarray(autodiff(x)), atol=1e-4)


@pytest.mark.parametrize("basis", LAGRANGE, ids=lambda b: b.__name__)
def test_lagrange_reproduces_polynomials_up_to_its_order(basis):
    """The defining accuracy claim: order-k Lagrange is exact on x^k."""
    nodes = np.array(basis.node_locs)
    x = np.asarray(SAMPLE)
    for power in range(basis.order + 1):
        got = np.asarray(basis.fn(SAMPLE)) @ (nodes ** power)
        np.testing.assert_allclose(got, x ** power, atol=tol(basis))


@pytest.mark.parametrize("basis", LAGRANGE, ids=lambda b: b.__name__)
def test_lagrange_is_not_exact_one_order_higher(basis):
    """Guards the previous test against a basis that is silently too rich."""
    nodes = np.array(basis.node_locs)
    power = basis.order + 1
    got = np.asarray(basis.fn(SAMPLE)) @ (nodes ** power)
    assert np.abs(got - np.asarray(SAMPLE) ** power).max() > 1e-4


@pytest.mark.parametrize("basis", ALL_BASES, ids=lambda b: b.__name__)
def test_derivative_of_a_constant_field_is_zero(basis):
    """Follows from partition of unity, and catches a mis-scaled derivative."""
    if basis is H3Basis:
        coeffs = np.array([1.0, 0.0, 1.0, 0.0])   #constant 1, zero slope
    else:
        coeffs = np.ones(len(basis.weights))
    np.testing.assert_allclose(np.asarray(basis.deriv[1](SAMPLE)) @ coeffs, 0.0, atol=1e-4)
