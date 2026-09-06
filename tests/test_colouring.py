"""Jacobian colouring: which parameters can be perturbed together.

``test_matrix_get_colour.py`` printed the colour count and drew the colours on
the nodes.  The property that makes a colouring *correct* is that no two
parameters sharing a colour ever influence the same output -- that is
checkable directly against the element map, and it is what the sparse
Jacobian evaluation relies on.

The script also unpacked two return values from a call that returns three,
so it could not have run as written.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis, L3Basis
from HOMER.geometry import cube

from _helpers import arr


def element_columns(mesh, fields_separable):
    """For each element, the optimisable-parameter columns it depends on."""
    optimisable = np.asarray(mesh.optimisable_param_bool)
    #position of each true parameter within the optimisable subset, or -1
    position = np.full(optimisable.shape[0], -1, dtype=int)
    position[optimisable] = np.arange(optimisable.sum())

    stride = mesh.fdim if fields_separable else 1
    groups = []
    for ele_map in np.asarray(mesh.ele_map):
        for offset in range(stride):
            cols = position[ele_map.astype(int)[offset::stride]]
            groups.append(set(cols[cols >= 0].tolist()))
    return groups


@pytest.fixture(scope="module")
def coloured_mesh():
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine([3, 3, 3])
    return mesh


@pytest.mark.parametrize("fields_separable", [False, True])
def test_no_two_parameters_of_one_element_share_a_colour(coloured_mesh, fields_separable):
    """The defining property: same colour implies never in the same output."""
    colours = coloured_mesh.get_colouring_dict(fields_seperable=fields_separable)

    for group in element_columns(coloured_mesh, fields_separable):
        used = [colours[c] for c in group]
        assert len(set(used)) == len(used)


def test_every_optimisable_parameter_gets_a_colour(coloured_mesh):
    colours = coloured_mesh.get_colouring_dict()

    assert set(colours) == set(range(len(coloured_mesh.optimisable_param_array)))


def test_treating_fields_as_separable_needs_fewer_colours(coloured_mesh):
    """Each coordinate of a vector field is an independent output, so the
    three components of a node can share a colour."""
    joint = coloured_mesh.get_colouring_dict(fields_seperable=False)
    separable = coloured_mesh.get_colouring_dict(fields_seperable=True)

    assert max(separable.values()) + 1 < max(joint.values()) + 1


def test_the_colouring_is_a_real_compression(coloured_mesh):
    n_colours = max(coloured_mesh.get_colouring_dict(fields_seperable=True).values()) + 1

    assert n_colours < len(coloured_mesh.optimisable_param_array)


def test_seed_matrices_come_back_as_a_triple(coloured_mesh):
    """Returns (colouring, value seed, index seed) -- not the pair the old
    script tried to unpack."""
    colours, seed_values, seed_indices = coloured_mesh.get_colouring_dict(
        fields_seperable=True, seed_matrix=True)

    n_params = max(colours) + 1
    n_colours = max(colours.values()) + 1
    assert seed_values.shape == (n_params, n_colours)
    assert seed_indices.shape == (n_params, n_colours)


def test_the_seed_matrix_has_exactly_one_entry_per_parameter(coloured_mesh):
    colours, seed_values, seed_indices = coloured_mesh.get_colouring_dict(
        fields_seperable=True, seed_matrix=True)

    values = np.asarray(seed_values.todense())
    indices = np.asarray(seed_indices.todense())

    np.testing.assert_array_equal(values.sum(axis=1), np.ones(values.shape[0]))
    for parameter, colour in colours.items():
        assert values[parameter, colour] == 1
        assert indices[parameter, colour] == parameter


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, H3Basis], ids=lambda b: b.__name__)
def test_colouring_works_for_every_basis(basis):
    mesh = cube(basis=[basis] * 3)
    mesh.refine(2)

    colours = mesh.get_colouring_dict(fields_seperable=True)

    for group in element_columns(mesh, True):
        used = [colours[c] for c in group]
        assert len(set(used)) == len(used)


def test_fixed_parameters_are_left_out_of_the_colouring():
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(2)
    mesh.nodes[0].fix_parameter('loc')
    mesh.generate_mesh()

    colours = mesh.get_colouring_dict()

    assert len(colours) == len(mesh.optimisable_param_array)
