"""Refinement and basis conversion must not move the geometry.

That is the whole claim the old ``refine_mesh_test.py`` / ``rebase_mesh_test.py``
scripts were making by drawing the before and after on top of each other.
Stated numerically: sample the original surface, then require every sample to
still lie on the converted mesh.

Constraint bookkeeping across the same two operations is covered separately,
in ``test_fixed_param_preservation.py``.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import (B3Basis, H3Basis, L1Basis, L2Basis,
                                     L3Basis)
from HOMER.geometry import basic_surface, cube

from _helpers import CLOSE, EXACT, arr, hermite_cube, node_locs, unit_hex

#(basis, nodes per direction added by a factor-r refinement of one element)
NODE_COUNT = {L1Basis: lambda r: r + 1,
              H3Basis: lambda r: r + 1,
              L2Basis: lambda r: 2 * r + 1,
              L3Basis: lambda r: 3 * r + 1,
              B3Basis: lambda r: r + 3}


def sample_surface(mesh, res=6):
    """Points on the mesh surface, as an independent description of its shape."""
    return arr(mesh.get_surface(just_faces=True, res=res)) if mesh.ndim == 3 \
        else arr(mesh.get_surface(res=res))


def max_distance_to(mesh, points):
    """How far the given points sit off *mesh*, via the embedding solve."""
    _, residual = mesh.embed_points(points, return_residual=True, iterations=25)
    return np.linalg.norm(arr(residual), axis=-1).max()


def mean_distance_to(mesh, points):
    _, residual = mesh.embed_points(points, return_residual=True, iterations=25)
    return np.linalg.norm(arr(residual), axis=-1).mean()


############################################### refinement

@pytest.mark.parametrize("basis", list(NODE_COUNT), ids=lambda b: b.__name__)
@pytest.mark.parametrize("factor", [2, 3])
def test_refine_produces_the_expected_element_and_node_counts(basis, factor):
    mesh = unit_hex(basis=[basis] * 3)

    mesh.refine(factor)

    assert len(mesh.elements) == factor ** 3
    assert len(mesh.nodes) == NODE_COUNT[basis](factor) ** 3


def test_refine_accepts_a_per_direction_factor():
    mesh = unit_hex(basis=[L1Basis] * 3)

    mesh.refine([2, 3, 2])

    assert len(mesh.elements) == 12
    assert len(mesh.nodes) == 3 * 4 * 3
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, H3Basis], ids=lambda b: b.__name__)
def test_refine_preserves_a_curved_surface(basis):
    """The visual check made numeric: the refined mesh still passes through
    every point of the original."""
    mesh = hermite_cube().rebase([basis] * 3)
    before = sample_surface(mesh)

    mesh.refine(2)

    assert max_distance_to(mesh, before) < EXACT


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, H3Basis], ids=lambda b: b.__name__)
def test_uniform_refine_reaches_float32_round_off(basis):
    """Subdivision is exact in exact arithmetic; the question is whether the
    solve keeps it.

    The refined space contains the coarse one, so the least-squares system is
    consistent and an exact answer exists.  It used to be lost to conditioning
    -- a refined tricubic Hermite mesh has ``cond(W) = 4.4e4`` and float32
    gives up four to five digits at that point.  Collocating over the whole
    child element and equilibrating the columns recovers them, so the refined
    surface now sits at the same round-off floor as the unrefined one.
    """
    mesh = hermite_cube().rebase([basis] * 3)
    before = sample_surface(mesh)

    floor = max_distance_to(mesh, before)          #no refinement: pure solver noise
    mesh.refine(2)

    assert floor < 1e-5
    assert max_distance_to(mesh, before) < 1e-5


def test_uniform_refine_of_a_control_net_is_exact_away_from_the_domain_edge():
    """B3 is the one basis that does not quite reach the floor everywhere.

    Its outermost control points lie outside the domain and are constrained by
    only part of their support, so the error concentrates at the boundary: the
    worst samples sit 2-5x closer to a domain edge than average, while the
    mean is at round-off.  Still 40x better than before the equilibrated solve.
    """
    mesh = hermite_cube().rebase([B3Basis] * 3)
    before = sample_surface(mesh)

    mesh.refine(2)

    assert mean_distance_to(mesh, before) < 1e-5
    assert max_distance_to(mesh, before) < 1e-4


def test_refine_by_non_uniform_xi_breaks_is_exact_for_lagrange():
    """Lagrange nodal parameters are values, which do not care about the split."""
    for basis in (L1Basis, L2Basis, L3Basis):
        mesh = hermite_cube().rebase([basis] * 3)
        before = sample_surface(mesh)

        mesh.refine(by_xi_refinement=[np.array([0, 0.3, 1.0]),
                                      np.array([0, 0.65, 1.0]),
                                      np.array([0, 0.5, 1.0])])

        assert len(mesh.elements) == 8, basis
        assert max_distance_to(mesh, before) < 1e-5, basis


@pytest.mark.parametrize("basis", [H3Basis, B3Basis], ids=lambda b: b.__name__)
def test_non_uniform_refine_of_a_derivative_basis_cannot_be_exact(basis):
    """A representability limit, not a solver one -- so it is pinned, not fixed.

    ``H3``: the node between two children of different widths carries one
    ``du``, but each child needs ``dx/deta = h * dx/dxi`` with its own ``h``.
    One shared parameter cannot serve both.  (``MeshElement.scale_factors``
    is the hook that would let it; it is never populated.)

    ``B3``: an interior knot makes the knot vector non-uniform, which a
    uniform B-spline basis has no way to express.

    Both show the same signature -- at the floor when the two halves are equal,
    two orders of magnitude worse the moment they are not, and no better in
    float64, because the least-squares system cannot match its own targets.
    """
    even = hermite_cube().rebase([basis] * 3)
    before = sample_surface(even)
    even.refine(by_xi_refinement=[np.array([0, 0.5, 1.0])] * 3)

    uneven = hermite_cube().rebase([basis] * 3)
    uneven.refine(by_xi_refinement=[np.array([0, 0.49, 1.0]),      #2% uneven is enough
                                    np.array([0, 0.5, 1.0]),
                                    np.array([0, 0.5, 1.0])])

    assert max_distance_to(even, before) < 1e-4
    assert max_distance_to(uneven, before) > 5e-5
    assert max_distance_to(uneven, before) < 5e-3                  #but bounded


def test_get_volume_is_unchanged_by_refinement():
    """Subdividing an element must not change how much space it encloses.

    ``get_volume`` used to quadrature at the basis order -- a single Gauss
    point for L1 -- which under-integrates det(J) on any hexahedron that is
    not affine.  The symptom was a volume that crept upwards with every
    refinement, because smaller elements are individually closer to affine.
    The rule is now chosen to integrate det(J) exactly, so the answer is the
    same at every resolution.
    """
    mesh = hermite_cube().rebase([L1Basis] * 3)

    volumes = [float(mesh.get_volume())]
    for _ in range(3):
        mesh.refine(2)
        volumes.append(float(mesh.get_volume()))

    np.testing.assert_allclose(volumes, volumes[0], rtol=EXACT)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, H3Basis], ids=lambda b: b.__name__)
def test_refine_preserves_the_volume(basis):
    mesh = hermite_cube().rebase([basis] * 3)
    before = float(mesh.get_volume())

    mesh.refine(2)

    #refinement itself is a least-squares re-fit, so the geometry moves a
    #little; the volume follows it and nothing more
    assert mesh.get_volume() == pytest.approx(before, rel=1e-3)


############################################### rebasing

@pytest.mark.parametrize("target", [L2Basis, L3Basis, H3Basis], ids=lambda b: b.__name__)
def test_rebase_to_a_richer_basis_is_exact_on_a_trilinear_cube(target):
    """L2, L3 and H3 all contain the trilinear space, so nothing may move."""
    source = unit_hex(basis=[L1Basis] * 3)
    xi = source.xi_grid(5)
    reference = arr(source.evaluate_embeddings_ele_xi_pair(np.zeros(len(xi), int), xi))

    out = source.rebase([target] * 3)

    got = arr(out.evaluate_embeddings_ele_xi_pair(np.zeros(len(xi), int), xi))
    np.testing.assert_allclose(got, reference, atol=EXACT)


def test_rebase_to_a_control_net_is_exact_on_a_trilinear_cube():
    """B3 can represent the geometry, and now reaches it.

    The control-net weight matrix is the worst-conditioned HOMER builds
    (``cond(W) = 1.6e5``), which used to leave this fit 5.8e-4 off.  The
    equilibrated solve brings it to the same floor as every other basis.
    """
    source = unit_hex(basis=[L1Basis] * 3)
    xi = source.xi_grid(5)
    reference = arr(source.evaluate_embeddings_ele_xi_pair(np.zeros(len(xi), int), xi))

    out = source.rebase([B3Basis] * 3)

    got = arr(out.evaluate_embeddings_ele_xi_pair(np.zeros(len(xi), int), xi))
    np.testing.assert_allclose(got, reference, atol=EXACT)


def test_rebase_to_the_same_basis_is_a_no_op():
    source = hermite_cube()

    out = source.rebase([H3Basis] * 3)

    np.testing.assert_allclose(arr(out.true_param_array), arr(source.true_param_array), atol=CLOSE)


def test_rebase_returns_a_new_mesh_by_default():
    source = unit_hex(basis=[L1Basis] * 3)

    out = source.rebase([L2Basis] * 3)

    assert out is not source
    assert len(source.nodes) == 8
    assert len(out.nodes) == 27


def test_rebase_in_place_mutates_the_original_and_returns_it():
    mesh = unit_hex(basis=[L1Basis] * 3)

    returned = mesh.rebase([L2Basis] * 3, in_place=True)

    assert returned is mesh
    assert len(mesh.nodes) == 27
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)


def test_rebase_in_place_keeps_secondary_fields():
    from HOMER import MeshField

    mesh = unit_hex(basis=[L1Basis] * 3)
    mesh.new_field('scalar', field_dimension=1, field_params=np.arange(8, dtype=float))

    mesh.rebase([L2Basis] * 3, in_place=True)

    assert 'scalar' in mesh.fields
    assert isinstance(mesh['scalar'], MeshField)


def test_rebase_down_is_a_least_squares_fit_not_a_corner_interpolation():
    """H3 -> L1 cannot hold the bowed edges, and does not privilege the corners.

    The trilinear fit balances the error over the whole element: the centre
    comes back exactly and the corners move by an amount set by the curvature
    being discarded.  Anyone rebasing down to pin landmarks needs to know
    that, which is why it is asserted rather than assumed.
    """
    curved = hermite_cube()
    corner_xi = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
    centre_xi = np.array([[0.5, 0.5, 0.5]])
    corners = arr(curved.evaluate_embeddings(0, corner_xi))
    centre = arr(curved.evaluate_embeddings(0, centre_xi))

    flat = curved.rebase([L1Basis] * 3)

    np.testing.assert_allclose(arr(flat.evaluate_embeddings(0, centre_xi)), centre, atol=EXACT)
    assert np.abs(arr(flat.evaluate_embeddings(0, corner_xi)) - corners).max() > 0.1


def test_multi_element_rebase_keeps_the_surface():
    """Shared nodes are the interesting part: neighbours must stay stitched."""
    mesh = cube(basis=[L1Basis] * 3)
    mesh.refine(2)
    before = sample_surface(mesh)

    out = mesh.rebase([H3Basis] * 3)

    assert max_distance_to(out, before) < EXACT


def test_rebase_of_a_surface_mesh_keeps_it_in_plane():
    mesh = basic_surface(basis=[L2Basis] * 2)

    out = mesh.rebase([B3Basis] * 2)

    surface = arr(out.get_surface(res=8))
    np.testing.assert_allclose(surface[:, 0], 0.0, atol=CLOSE)
