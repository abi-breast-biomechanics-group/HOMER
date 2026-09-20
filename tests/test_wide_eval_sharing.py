"""One compiled wide evaluation, shared between fields.

``*_in_every_element`` used to rebuild its closure on every call, so JAX met a
function it had never seen and retraced from scratch each time -- identical
meshes paid the same ~9ms over and over.  The evaluators now take the field's
element map, scale factors and parameters as arguments instead of reading
them, which lets one jitted copy serve every field built on the same basis.

That only holds if nothing field-specific is still being read through the
closure, and a miss there would be silent: the shared copy would quietly
answer with whichever field happened to build it.  These tests compare shared
answers against each field's own, on fields deliberately made to differ.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import B3, H3, L1, L2, L3
from HOMER.geometry import cube
from HOMER.mesh_decorators import _WIDE_EVAL_JITS, _wide_eval_key

from _helpers import EXACT


XI = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 0.25, 0.75]])


def build(basis, res, scale=1.0):
    mesh = cube(basis=[basis] * 3)
    mesh.refine([res] * 3)
    mesh.generate_mesh()
    if scale != 1.0:
        mesh.update_from_params(np.asarray(mesh.true_param_array) * scale)
    return mesh


def elementwise(mesh, xi):
    """The same values, gathered one element at a time without the wide path."""
    return np.concatenate([np.asarray(mesh.evaluate_embeddings(np.array([e]), xi))
                           for e in range(len(mesh.elements))])


@pytest.mark.parametrize("basis", [L1, H3], ids=lambda b: b.__name__)
def test_fields_sharing_a_basis_share_one_compiled_evaluation(basis):
    a, b = build(basis, 2), build(basis, 2, scale=1.7)

    key_a = _wide_eval_key(a, "in_every_element", "evaluate_embeddings", 0, False)
    key_b = _wide_eval_key(b, "in_every_element", "evaluate_embeddings", 0, False)
    assert key_a == key_b, "same basis should reach the same entry"


@pytest.mark.parametrize("basis", [L1, H3], ids=lambda b: b.__name__)
def test_a_shared_evaluation_still_answers_for_each_field(basis):
    """The failure this guards: the shared copy answering with the wrong field."""
    a, b = build(basis, 2), build(basis, 2, scale=1.7)

    wide_a = np.asarray(a.evaluate_embeddings_in_every_element(XI))
    wide_b = np.asarray(b.evaluate_embeddings_in_every_element(XI))

    assert np.allclose(wide_a, elementwise(a, XI), atol=EXACT)
    assert np.allclose(wide_b, elementwise(b, XI), atol=EXACT)
    #and they are genuinely different fields, so a mix-up would have shown
    assert not np.allclose(wide_a, wide_b, atol=EXACT)


def test_order_of_use_does_not_decide_the_answer():
    """Whichever field builds the entry, both must still get their own values."""
    first, second = build(L1, 2), build(L1, 2, scale=0.4)
    forwards = (np.asarray(first.evaluate_embeddings_in_every_element(XI)),
                np.asarray(second.evaluate_embeddings_in_every_element(XI)))

    _WIDE_EVAL_JITS.clear()

    backwards = (np.asarray(second.evaluate_embeddings_in_every_element(XI)),
                 np.asarray(first.evaluate_embeddings_in_every_element(XI)))

    assert np.allclose(forwards[0], backwards[1], atol=EXACT)
    assert np.allclose(forwards[1], backwards[0], atol=EXACT)


def test_the_jacobian_path_shares_correctly_too():
    a, b = build(L1, 2), build(L1, 2, scale=2.5)

    jac_a = np.asarray(a.evaluate_jacobians_in_every_element(XI))
    jac_b = np.asarray(b.evaluate_jacobians_in_every_element(XI))

    #b is a uniformly scaled by 2.5, so its jacobians scale with it
    assert np.allclose(jac_b, 2.5 * jac_a, atol=EXACT)


def test_fixing_parameters_does_not_leak_between_fields():
    """The optimisable indices travel as an argument, so they must not stick."""
    free, fixed = build(L1, 2), build(L1, 2)
    fixed.nodes[0].fix_parameter('loc', inds=[2])
    fixed.generate_mesh()

    assert np.allclose(np.asarray(free.evaluate_embeddings_in_every_element(XI)),
                       elementwise(free, XI), atol=EXACT)
    assert np.allclose(np.asarray(fixed.evaluate_embeddings_in_every_element(XI)),
                       elementwise(fixed, XI), atol=EXACT)


@pytest.mark.parametrize("left,right", [(L1, L2), (L2, L3), (L3, H3), (H3, B3)],
                         ids=lambda b: b.__name__)
def test_different_bases_never_share_an_entry(left, right):
    """Every basis is an instance of the same ``Basis`` class.

    Keying on the class therefore said nothing at all, and H3 and B3 -- same
    order, same element count -- would have been handed each other's compiled
    evaluation.  The key has to name the basis.
    """
    a, b = build(left, 2), build(right, 2)

    assert (_wide_eval_key(a, "in_every_element", "evaluate_embeddings", 0, False)
            != _wide_eval_key(b, "in_every_element", "evaluate_embeddings", 0, False))


def test_the_two_wide_variants_do_not_share_an_entry():
    """``in_every_element`` and ``ele_xi_pair`` are different computations."""
    mesh = build(L1, 2)

    assert (_wide_eval_key(mesh, "in_every_element", "evaluate_embeddings", 0, False)
            != _wide_eval_key(mesh, "ele_xi_pair", "evaluate_embeddings", 0, False))


@pytest.mark.parametrize("basis", [L1, H3], ids=lambda b: b.__name__)
def test_the_paired_variant_answers_for_each_field(basis):
    a, b = build(basis, 2), build(basis, 2, scale=1.9)
    eles = np.arange(len(a.elements))
    xis = np.tile(np.array([0.25, 0.5, 0.75]), (len(eles), 1))

    pair_a = np.asarray(a.evaluate_embeddings_ele_xi_pair(eles, xis))
    pair_b = np.asarray(b.evaluate_embeddings_ele_xi_pair(eles, xis))

    direct_a = np.concatenate([np.asarray(a.evaluate_embeddings(np.array([e]), xis[i:i + 1]))
                               for i, e in enumerate(eles)])
    assert np.allclose(pair_a, direct_a, atol=EXACT)
    assert not np.allclose(pair_a, pair_b, atol=EXACT)
