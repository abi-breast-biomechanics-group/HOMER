"""Properties every 1-D basis must satisfy.

These are the checks that used to be implicit: a mesh "looked wrong" when a
basis was wrong.  Stated directly they are cheap, and they localise a fault
to the basis rather than to the mesh built on it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dataclasses import replace

from HOMER.basis_definitions import (B3, BasisGroup, H3, L1,
                                     L2, L3, L4, Lagrange,
                                     basis_by_name, registered_bases)

ALL_BASES = [L1, L2, L3, L4, H3, B3]
LAGRANGE = [L1, L2, L3, L4]

SAMPLE = jnp.linspace(0.0, 1.0, 11)


def tol(basis):
    """Everything here is exact in exact arithmetic; the slack is float32.

    The quartic weights are the worst-conditioned of the set -- they carry
    coefficients into the hundreds that then cancel -- and lose about two
    digits more than the rest.
    """
    return 1e-4 if basis is L4 else 1e-6


@pytest.mark.parametrize("basis", ALL_BASES, ids=lambda b: b.__name__)
def test_weight_names_match_the_evaluated_columns(basis):
    assert basis.fn(SAMPLE).shape == (len(SAMPLE), len(basis.weights))


@pytest.mark.parametrize("basis", LAGRANGE + [B3], ids=lambda b: b.__name__)
def test_partition_of_unity(basis):
    """A constant field must be reproduced exactly, at every xi."""
    np.testing.assert_allclose(np.asarray(basis.fn(SAMPLE)).sum(-1), 1.0, atol=tol(basis))


def test_hermite_splits_into_value_and_derivative_weights():
    """H3 is not a partition of unity; only its two value weights are."""
    weights = np.asarray(H3.fn(SAMPLE))
    value_cols = [i for i, w in enumerate(H3.weights) if not w.startswith('d')]
    deriv_cols = [i for i, w in enumerate(H3.weights) if w.startswith('d')]

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
    at_nodes = np.asarray(H3.fn(jnp.array([0.0, 1.0])))
    slope_at_nodes = np.asarray(H3.deriv[1](jnp.array([0.0, 1.0])))

    #columns are [x0, dx0, x1, dx1]
    np.testing.assert_allclose(at_nodes, [[1, 0, 0, 0], [0, 0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(slope_at_nodes, [[0, 1, 0, 0], [0, 0, 0, 1]], atol=1e-6)
    assert H3.interpolatory


def test_b3_is_flagged_as_a_control_net():
    """B3 parameters are control points, not values at ``node_locs``.

    Refinement relies on this flag to decide whether a fixed nodal value may
    be carried across verbatim, so it is worth pinning.
    """
    assert not B3.interpolatory
    at_nodes = np.asarray(B3.fn(jnp.array([0.0, 1.0])))
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
    if basis is H3:
        coeffs = np.array([1.0, 0.0, 1.0, 0.0])   #constant 1, zero slope
    else:
        coeffs = np.ones(len(basis.weights))
    np.testing.assert_allclose(np.asarray(basis.deriv[1](SAMPLE)) @ coeffs, 0.0, atol=1e-4)


# --------------------------------------------------------------------------
# Bases as values: the algebra, identity, and the registry.
# --------------------------------------------------------------------------

def test_multiplication_repeats_a_basis_across_directions():
    assert tuple(H3 * 3) == (H3, H3, H3)
    assert tuple(3 * H3) == (H3, H3, H3)
    assert tuple(H3 ** 3) == (H3, H3, H3)


def test_multiplication_joins_directions_in_order():
    assert tuple(H3 * B3) == (H3, B3)
    assert tuple(H3 * B3**2) == (H3, B3, B3)
    assert tuple(H3**2 * B3) == (H3, H3, B3)
    assert tuple(H3 * H3 * B3) == (H3, H3, B3)
    assert tuple(H3 * 2 * B3) == (H3, H3, B3)
    assert tuple((H3 * L1)**2) == (H3, L1, H3, L1)
    assert tuple((H3 * L1) * 2) == (H3, L1, H3, L1)


def test_a_sequence_on_the_left_prepends_rather_than_appends():
    """``list * basis`` reaches ``Basis.__rmul__``, which must not be the
    mirror of ``__mul__``: the left operand stays on the left."""
    assert tuple([H3, L1] * B3) == (H3, L1, B3)
    assert tuple((L1,) * H3) == (L1, H3)
    assert tuple(B3 * [H3, L1]) == (B3, H3, L1)
    assert tuple([H3] * (L1 * B3)) == (H3, L1, B3)


def test_addition_is_gone_and_says_what_to_write_instead():
    """``BasisGroup`` subclasses ``tuple``, so a deleted ``__add__`` would be
    inherited and silently return a plain tuple.  It has to raise."""
    with pytest.raises(TypeError):
        H3 + L1
    for expr in (lambda: H3 + (L1 * B3),
                 lambda: (H3 * L1) + B3,
                 lambda: (H3 * L1) + (B3 * H3)):
        with pytest.raises(TypeError, match="no longer combines"):
            expr()


def test_a_count_has_to_be_an_actual_count():
    """``bool`` is an ``int`` subclass; ``H3 * True`` is a mistake."""
    for expr in (lambda: H3 * True, lambda: H3 * 'abc',
                 lambda: H3 ** 1.5, lambda: (H3 * L1) * None):
        with pytest.raises(TypeError):
            expr()


def test_a_group_is_a_tuple_so_the_old_list_spelling_still_works():
    """Everything downstream indexes, iterates and lens the basis group."""
    group = H3**2 * B3
    assert isinstance(group, tuple)
    assert group == (H3, H3, B3) == BasisGroup([H3, H3, B3])
    assert group[0] is H3 and len(group) == 3 and group.ndim == 3
    assert list(group) == [H3, H3, B3]


def test_group_interpolatory_is_the_and_of_its_directions():
    assert (H3 * 3).interpolatory
    assert not (H3**2 * B3).interpolatory


def test_a_group_repr_reads_back_as_the_expression_that_built_it():
    assert repr(H3**2 * B3) == "H3**2 * B3"
    assert repr(H3 * L1 * B3) == "H3 * L1 * B3"
    assert repr(BasisGroup()) == "BasisGroup()"
    #and the repr is the expression: it evaluates back to the group
    expr = repr(H3**2 * B3)
    assert tuple(eval(expr, {'H3': H3, 'B3': B3})) == (H3, H3, B3)


def test_bases_are_values_that_survive_copying():
    """Identity is the name, so a basis stays itself through a round-trip."""
    import copy
    import pickle
    assert copy.deepcopy(H3) is H3
    assert pickle.loads(pickle.dumps(H3)) is H3
    assert H3 == basis_by_name('H3')
    assert len({H3, H3, L1}) == 2


@pytest.mark.parametrize("basis", ALL_BASES, ids=lambda b: b.__name__)
def test_every_basis_registers_under_its_own_name(basis):
    """The registry is what lets HOMER.io resolve a basis out of JSON."""
    assert registered_bases()[basis.name] is basis
    assert basis.__name__ == basis.name


def test_an_unknown_basis_name_says_what_is_available():
    with pytest.raises(KeyError, match="L1"):
        basis_by_name('NoSuchBasis')


def test_lagrange_selects_a_basis_by_order():
    assert Lagrange(1) is L1 and Lagrange(4) is L4
    with pytest.raises(ValueError, match="No Lagrange basis of order 7"):
        Lagrange(7)


def test_a_basis_may_be_varied_without_subclassing():
    """`replace` is how you get a one-off variant now that bases are values."""
    degree7 = replace(L4, name='ADegree7Basis', order=7)
    assert degree7.order == 7 and degree7.fn is L4.fn
    assert degree7 != L4
    assert basis_by_name('ADegree7Basis') is degree7


def test_a_malformed_basis_fails_where_it_is_defined():
    """The checks that used to be implicit in "the mesh looked wrong"."""
    with pytest.raises(ValueError, match="weight names"):
        replace(L1, name='TooFewWeights', weights=('x0',))
    with pytest.raises(ValueError, match="deriv\\[0\\] must be fn"):
        replace(L1, name='WrongDeriv', deriv=(L2.fn,))


def test_a_name_may_not_be_reused_for_a_different_basis():
    """Names are the serialisation key; two meanings would corrupt a load."""
    with pytest.raises(ValueError, match="already registered"):
        replace(L1, name='H3')


def test_a_group_rejects_anything_that_is_not_a_basis():
    with pytest.raises(TypeError, match="pass H3, not H3"):
        BasisGroup([H3, 'H3'])
