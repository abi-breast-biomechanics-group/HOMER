"""Node numbering after a manipulation is predictable, and only the numbering.

Refining and rebasing rebuild the node list, and the order that falls out of
the algorithm is an artefact of the sweep.  These tests pin the two halves of
the fix: the ordering is the mesh's parametric lattice, and applying it changes
nothing else -- not the geometry, not the fields, not the constraints.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import B3Basis, H3Basis, L1Basis, L2Basis
from HOMER.geometry import basic_surfaceMN, cube, cubeMNO
from HOMER.mesh import reordering
from HOMER.mesh.reordering import (apply_node_permutation, element_lattice_coords,
                                   node_permutation, preserving_permutation,
                                   reorder_nodes, resolve_strategy)

from _helpers import CLOSE, EXACT, arr, node_locs, unit_hex


def is_lexicographic(locs, decimals=4):
    """Are these positions already sorted with the last coordinate slowest?"""
    rounded = np.round(np.asarray(locs, dtype=float), decimals)
    return np.array_equal(np.lexsort(rounded.T), np.arange(len(rounded)))


def sample(mesh, res=5):
    """Points on the mesh, as a description of its shape that ignores numbering."""
    grid = mesh.xi_grid(res=res)
    eles = np.repeat(np.arange(len(mesh.elements)), len(grid))
    return arr(mesh.evaluate_embeddings_ele_xi_pair(eles, np.tile(grid, (len(mesh.elements), 1))))


####################################### the ordering itself

@pytest.mark.parametrize("res", [[2, 2, 2], [3, 1, 2], [2, 3, 4]])
def test_refining_a_cube_numbers_the_nodes_along_the_lattice(res):
    """The whole point: a refined axis-aligned cube comes out in (z, y, x) order."""
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(by_xi_refinement=[np.linspace(0, 1, r + 1) for r in res])

    assert is_lexicographic(node_locs(mesh))


def test_the_lattice_ordering_reproduces_what_cubeMNO_sorts_by_hand():
    """`cubeMNO` sorts on coordinates; `refine` now gets there from the topology."""
    by_hand = node_locs(cubeMNO([3, 3, 3], basis=[L1Basis] * 3))

    from_refine = cube(basis=[L1Basis] * 3)
    from_refine.refine(3)

    np.testing.assert_allclose(node_locs(from_refine), by_hand, atol=EXACT)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, H3Basis, B3Basis],
                         ids=lambda b: b.__name__)
def test_every_basis_refines_into_lattice_order(basis):
    """Including B3, whose control points sit outside the element that owns them.

    The check is that the ordering is a fixed point -- reordering again would
    do nothing -- rather than that the coordinates are sorted.  A control net
    is fitted rather than interpolated, so its points carry the fit's noise and
    are not exactly co-planar; the lattice ordering is unbothered by that,
    which is the reason it does not look at coordinates.
    """
    mesh = cube(basis=[basis] * 3)
    mesh.refine(2)

    np.testing.assert_array_equal(node_permutation(mesh, 'lattice'),
                                  np.arange(len(mesh.nodes)))
    if basis.interpolatory:
        assert is_lexicographic(node_locs(mesh))


def test_rebasing_to_a_denser_basis_numbers_the_new_nodes_along_the_lattice():
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3).rebase([L2Basis] * 3)

    assert len(mesh.nodes) == 5 ** 3
    assert is_lexicographic(node_locs(mesh))


def test_rebasing_at_the_same_node_count_keeps_the_numbering_it_had():
    """An operation that does not change the node set must not renumber it.

    A rebase rebuilds the node list from the new basis, so left alone it comes
    back in an order that has nothing to do with the order the mesh had.  L1
    and H3 have the same nodes, so the old numbering is reproduced -- including
    a deliberately awkward one, which is the case that shows it is preservation
    and not the lattice ordering agreeing by luck.
    """
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    reorder_nodes(mesh, 'bandwidth')            #an order no strategy would pick
    before = node_locs(mesh)
    assert not is_lexicographic(before)

    rebased = mesh.rebase([H3Basis] * 3)

    assert len(rebased.nodes) == 3 ** 3
    np.testing.assert_allclose(node_locs(rebased), before, atol=EXACT)


def test_the_same_mesh_built_two_ways_gets_the_same_numbering():
    """One factor-4 refinement, or two factor-2 ones, must number alike."""
    one_step = cube(basis=[L1Basis] * 3)
    one_step.refine(4)

    two_steps = cube(basis=[L1Basis] * 3)
    two_steps.refine(2)
    two_steps.refine(2)

    np.testing.assert_allclose(node_locs(one_step), node_locs(two_steps), atol=EXACT)


def test_a_surface_mesh_is_ordered_by_its_parametric_directions():
    """xi_0 fastest: the 2-D lattice is walked a xi_1 row at a time."""
    mesh = basic_surfaceMN([3, 2], basis=[L1Basis] * 2)
    #the default patch lies on x = 0, with xi_0 along z and xi_1 along y
    locs = node_locs(mesh)

    assert is_lexicographic(locs)               #basic_surfaceMN asks for 'spatial'

    from_refine = basic_surfaceMN([3, 2], basis=[L1Basis] * 2)
    reorder_nodes(from_refine, 'lattice')
    yz = node_locs(from_refine)[:, [2, 1]]      #(xi_0, xi_1) as coordinates
    assert np.array_equal(np.lexsort(np.round(yz, 4).T), np.arange(len(yz)))


def test_rebasing_back_and_forth_returns_the_numbering_it_started_with():
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    reorder_nodes(mesh, 'bandwidth')
    before = node_locs(mesh)

    round_trip = mesh.rebase([H3Basis] * 3).rebase([L1Basis] * 3)

    np.testing.assert_allclose(node_locs(round_trip), before, atol=EXACT)


def test_a_refinement_that_adds_no_nodes_keeps_the_numbering():
    """A factor of one in every direction: the same mesh, so the same indices."""
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    reorder_nodes(mesh, 'bandwidth')
    before = node_locs(mesh)

    mesh.refine(by_xi_refinement=[np.linspace(0, 1, 2)] * 3)

    np.testing.assert_allclose(node_locs(mesh), before, atol=EXACT)


def test_a_refinement_that_does_add_nodes_renumbers():
    """The old nodes are only a subset of the new ones, so there is no old order.

    Preserving their indices would strand every added node in the arbitrary
    order the sweep produced, which is the thing being fixed.
    """
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    reorder_nodes(mesh, 'bandwidth')

    mesh.refine(2)

    assert len(mesh.nodes) == 5 ** 3
    assert is_lexicographic(node_locs(mesh))


def test_an_iso_rebase_of_a_field_keeps_the_fields_numbering_too():
    mesh = unit_hex(basis=[L1Basis] * 3)
    field = mesh.rebase([L1Basis] * 3, reorder_nodes=False)   #a plain copy to work from
    reorder_nodes(field, 'bandwidth')
    before = node_locs(field)

    np.testing.assert_allclose(node_locs(field.rebase([H3Basis] * 3)), before, atol=EXACT)


def test_preservation_is_declined_when_the_correspondence_is_not_one_to_one():
    """The unit test of the rule itself, on maps a real operation would produce."""
    #every new node on a distinct old one, none left over
    np.testing.assert_array_equal(preserving_permutation([2, 0, 1], 3), [1, 2, 0])
    #a node with no old counterpart
    assert preserving_permutation([0, -1, 2], 3) is None
    #two new nodes on the same old one
    assert preserving_permutation([0, 0, 2], 3) is None
    #more new nodes than old
    assert preserving_permutation([0, 1, 2, 3], 3) is None


####################################### it renumbers, and nothing else

@pytest.mark.parametrize("strategy", ['lattice', 'spatial', 'bandwidth'])
def test_a_reorder_is_a_permutation_and_nothing_more(strategy):
    mesh = cubeMNO([2, 2, 2], basis=[H3Basis] * 3)
    before_locs = node_locs(mesh)
    before_shape = sample(mesh)

    perm = reorder_nodes(mesh, strategy)

    assert sorted(perm.tolist()) == list(range(len(mesh.nodes)))
    np.testing.assert_allclose(node_locs(mesh), before_locs[perm], atol=0)
    np.testing.assert_allclose(sample(mesh), before_shape, atol=EXACT)
    assert mesh.get_volume() == pytest.approx(1.0, abs=CLOSE)


@pytest.mark.parametrize("strategy", ['lattice', 'spatial', 'bandwidth'])
def test_the_elements_still_point_at_the_nodes_they_did(strategy):
    """The renumbering has to be pushed through every element's node list."""
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    before = [[tuple(np.round(mesh.nodes[n].loc, 6)) for n in e.nodes] for e in mesh.elements]

    reorder_nodes(mesh, strategy)

    after = [[tuple(np.round(mesh.nodes[n].loc, 6)) for n in e.nodes] for e in mesh.elements]
    assert after == before


def test_reordering_carries_the_fixed_parameters_with_the_node():
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    pinned = min(range(len(mesh.nodes)), key=lambda i: tuple(mesh.nodes[i].loc))
    pinned_loc = np.array(mesh.nodes[pinned].loc)
    mesh.nodes[pinned].fix_parameter('loc', inds=[0, 2])
    mesh.generate_mesh()

    reorder_nodes(mesh, 'bandwidth')

    still_fixed = [n for n in mesh.nodes if n.fixed_params]
    assert len(still_fixed) == 1
    np.testing.assert_allclose(still_fixed[0].loc, pinned_loc, atol=0)
    np.testing.assert_array_equal(still_fixed[0].fixed_params['loc'], [0, 2])
    #and the parameter vector agrees with the node it now sits at
    assert (~mesh.optimisable_param_bool).sum() == 2


def test_a_refined_mesh_keeps_its_fields_lined_up_with_the_geometry():
    """Every field is renumbered from its own topology, so co-location holds."""
    mesh = unit_hex(basis=[L1Basis] * 3)
    pts = np.array(np.meshgrid(*[np.linspace(0.05, 0.95, 4)] * 3)).reshape(3, -1).T
    mesh.new_field('height', field_dimension=1, new_basis=[L1Basis] * 3,
                   field_locs=pts, field_values=pts[:, 2])

    mesh.refine(2)

    where, _ = mesh.embed_points(pts)
    xis = arr(_)
    values = arr(mesh['height'].evaluate_embeddings_ele_xi_pair(np.asarray(where), xis)).ravel()
    np.testing.assert_allclose(values, pts[:, 2], atol=CLOSE)


####################################### the off switch

def test_refine_leaves_the_raw_ordering_alone_when_asked():
    ordered = cube(basis=[L1Basis] * 3)
    ordered.refine(2)
    raw = cube(basis=[L1Basis] * 3)
    raw.refine(2, reorder_nodes=False)

    assert not is_lexicographic(node_locs(raw))
    #same mesh either way, just numbered differently
    assert sorted(map(tuple, np.round(node_locs(raw), 4))) == \
           sorted(map(tuple, np.round(node_locs(ordered), 4)))


def test_rebase_leaves_the_raw_ordering_alone_when_asked():
    base = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)

    assert is_lexicographic(node_locs(base.rebase([L2Basis] * 3)))
    assert not is_lexicographic(node_locs(base.rebase([L2Basis] * 3, reorder_nodes=False)))


def test_a_mesh_refine_passes_the_switch_down_to_its_fields():
    mesh = unit_hex(basis=[L1Basis] * 3)
    mesh.new_field('flat', field_dimension=1, new_basis=[L1Basis] * 3)
    before = [list(e.nodes) for e in mesh['flat'].elements]

    mesh.refine(2, reorder_nodes=False)

    after = [list(e.nodes) for e in mesh['flat'].elements]
    assert after != before          #it did refine
    assert not is_lexicographic(node_locs(mesh))


def test_the_default_ordering_can_be_turned_off_for_the_session(monkeypatch):
    monkeypatch.setattr(reordering, 'DEFAULT_NODE_ORDERING', False)
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(2)

    assert not is_lexicographic(node_locs(mesh))


def test_an_unknown_ordering_is_rejected():
    mesh = cube(basis=[L1Basis] * 3)
    with pytest.raises(ValueError, match="unknown node ordering"):
        reorder_nodes(mesh, 'nearest-neighbour')


####################################### the pieces

def test_element_lattice_coords_lay_a_refined_cube_out_on_a_grid():
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(by_xi_refinement=[np.linspace(0, 1, r + 1) for r in (2, 3, 4)])

    component, coords = element_lattice_coords(mesh._topo_lookup)

    assert set(component) == {0}
    assert coords.min(axis=0).tolist() == [0, 0, 0]
    assert coords.max(axis=0).tolist() == [1, 2, 3]
    assert len({tuple(c) for c in coords}) == len(coords)   #a coordinate each


def test_two_separate_blocks_are_ordered_a_block_at_a_time():
    left = cube(basis=[L1Basis] * 3)
    right = cube(basis=[L1Basis] * 3)
    right.transform(np.array([[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float))
    pair = left + right
    pair.refine(2)

    component, _ = element_lattice_coords(pair._topo_lookup)
    assert set(component) == {0, 1}

    #each block is contiguous in the node numbering, and lexicographic within
    xs = node_locs(pair)[:, 0]
    first_block = xs < 2.5
    assert first_block[:first_block.sum()].all()
    assert is_lexicographic(node_locs(pair)[first_block])
    assert is_lexicographic(node_locs(pair)[~first_block])


def test_the_permutation_is_returned_so_an_index_list_can_follow_it():
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    watched = [4, 11, 26]
    watched_locs = node_locs(mesh)[watched]

    perm = node_permutation(mesh, 'bandwidth')
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    apply_node_permutation(mesh, perm)

    np.testing.assert_allclose(node_locs(mesh)[inverse[watched]], watched_locs, atol=0)


def test_a_disabled_reorder_reports_that_it_did_nothing():
    mesh = cube(basis=[L1Basis] * 3)
    assert reorder_nodes(mesh, False) is None
    assert node_permutation(mesh, None) is None


def test_resolve_strategy_reports_what_a_switch_will_do():
    """refine/rebase check this before doing the work of a parent-node map."""
    assert resolve_strategy(True) == 'lattice'
    assert resolve_strategy('spatial') == 'spatial'
    assert resolve_strategy(False) is None
    assert resolve_strategy(None) is None
