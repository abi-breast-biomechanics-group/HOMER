"""The convenience factories in :mod:`HOMER.geometry`.

Nothing here was covered before, yet every other test (and most of the
applications) starts from one of these four functions.
"""

import numpy as np
import pytest

from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis, L3Basis
from HOMER.geometry import basic_surface, basic_surfaceMN, cube, cubeMNO

from _helpers import EXACT, node_locs

BASES = [L1Basis, L2Basis, L3Basis, H3Basis]


@pytest.mark.parametrize("basis", BASES, ids=lambda b: b.__name__)
def test_cube_is_a_unit_cube_whatever_the_basis(basis):
    mesh = cube(basis=[basis] * 3)

    assert mesh.ndim == 3
    assert len(mesh.elements) == 1
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)
    np.testing.assert_allclose(node_locs(mesh).min(0), -0.5, atol=EXACT)
    np.testing.assert_allclose(node_locs(mesh).max(0), 0.5, atol=EXACT)


def test_cube_honours_scale_and_centre():
    mesh = cube(scale=3.0, centre=np.array([1.0, -2.0, 0.5]), basis=[L1Basis] * 3)

    assert mesh.get_volume() == pytest.approx(27.0, rel=1e-5)
    np.testing.assert_allclose(node_locs(mesh).mean(0), [1.0, -2.0, 0.5], atol=EXACT)


def test_cube_accepts_an_unset_scale():
    """``cubeMNO`` forwards its own defaults straight through, so None must work."""
    assert cube(scale=None, centre=None, basis=[L1Basis] * 3).get_volume() == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("res", [[1, 1, 1], [2, 2, 2], [3, 1, 2]])
def test_cubeMNO_subdivides_without_changing_the_volume(res):
    mesh = cubeMNO(res, basis=[L1Basis] * 3)

    assert len(mesh.elements) == int(np.prod(res))
    assert len(mesh.nodes) == int(np.prod([r + 1 for r in res]))
    assert mesh.get_volume() == pytest.approx(1.0, abs=EXACT)


def test_cubeMNO_orders_nodes_lexicographically():
    """The re-ordering is the whole point of MNO over a plain refine."""
    mesh = cubeMNO([2, 2, 2], basis=[L1Basis] * 3)
    locs = np.round(node_locs(mesh), 4)

    order = np.lexsort((locs[:, 0], locs[:, 1], locs[:, 2]))
    np.testing.assert_array_equal(order, np.arange(len(locs)))


@pytest.mark.parametrize("basis", BASES, ids=lambda b: b.__name__)
def test_basic_surface_is_a_flat_unit_patch(basis):
    mesh = basic_surface(basis=[basis] * 2)

    assert mesh.ndim == 2
    #the default corners all lie on x = 0
    np.testing.assert_allclose(node_locs(mesh)[:, 0], 0.0, atol=EXACT)
    corners = np.asarray(mesh.evaluate_embeddings(0, np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])))
    np.testing.assert_allclose(np.sort(corners, axis=0),
                               np.sort(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], float), axis=0),
                               atol=EXACT)


def test_basic_surface_accepts_custom_corners():
    corners = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [2, 3, 0]], dtype=float)

    mesh = basic_surface(corner_locs=corners, basis=[L1Basis] * 2)

    np.testing.assert_allclose(np.sort(node_locs(mesh), axis=0), np.sort(corners, axis=0), atol=EXACT)


@pytest.mark.parametrize("res", [[1, 1], [2, 3]])
def test_basic_surfaceMN_subdivides_the_patch(res):
    mesh = basic_surfaceMN(res, basis=[L1Basis] * 2)

    assert len(mesh.elements) == int(np.prod(res))
    assert len(mesh.nodes) == int(np.prod([r + 1 for r in res]))
