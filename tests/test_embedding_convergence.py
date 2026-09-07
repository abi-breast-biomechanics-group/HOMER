"""The convergence exit in the Newton-Raphson refinement.

``embed_points`` stops refining once the residual is inside ``tol`` instead of
always running ``iterations`` steps.  The refinement is vectorised over the
query points, so a single ``lax.while_loop`` trip count is shared by the whole
batch -- which is the thing worth testing.  A converged point must be *frozen*,
not merely ignored: if the batched loop let finished lanes keep stepping until
the slowest one caught up, an answer would depend on which other points
happened to be embedded alongside it, and the same point would come back
differently from a solo call, a reordered batch, or a different ``chunk_size``.

The tolerance itself is checked to be what it claims -- an early exit at
float32 round-off, not an accuracy setting.
"""

import numpy as np
import pytest

from HOMER.embedding import DEFAULT_EMBED_TOL
from HOMER.geometry import cubeMNO

from _helpers import EXACT, arr


@pytest.fixture(scope="module")
def block():
    """A 2x2x2 tricubic-Hermite block: curved elements, real topology."""
    return cubeMNO((2, 2, 2))


@pytest.fixture(scope="module")
def on_mesh(block):
    """Points taken off the mesh, so they converge in two or three steps."""
    rng = np.random.default_rng(0)
    ele = rng.integers(0, len(block.elements), 24)
    xi = rng.random((24, 3))
    return arr(block.evaluate_embeddings_ele_xi_pair(ele, xi))


@pytest.fixture(scope="module")
def unreachable():
    """Points well outside the mesh, which never reach the tolerance.

    These are what hold a batch open: they keep iterating to the cap, so any
    lane coupling in the loop would show up on the points sharing their batch.
    """
    rng = np.random.default_rng(7)
    return arr(rng.random((6, 3))) * 6.0 - 2.5


def embed(mesh, points, **kw):
    (ele, xi), res = mesh.embed_points(np.atleast_2d(points),
                                       return_residual=True, **kw)
    return arr(ele), arr(xi), arr(res)


############################################### batch composition

def test_a_point_embeds_the_same_alone_as_in_a_batch(block, on_mesh):
    ele, xi, _ = embed(block, on_mesh)

    for i, point in enumerate(on_mesh):
        solo_ele, solo_xi, _ = embed(block, point)
        assert solo_ele[0] == ele[i]
        np.testing.assert_array_equal(solo_xi[0], xi[i])


def test_a_point_that_cannot_converge_does_not_disturb_the_others(block, on_mesh,
                                                                 unreachable):
    """The case the shared trip count makes possible.

    The unreachable points iterate to the cap, so the batch stays open long
    after the on-mesh points have finished.  Those points must come back
    exactly as they do without such company.
    """
    ele, xi, _ = embed(block, on_mesh)
    mixed_ele, mixed_xi, _ = embed(block, np.concatenate([on_mesh, unreachable]))

    np.testing.assert_array_equal(mixed_ele[:len(on_mesh)], ele)
    np.testing.assert_array_equal(mixed_xi[:len(on_mesh)], xi)


def test_lane_position_does_not_change_the_answer(block, on_mesh, unreachable):
    points = np.concatenate([on_mesh, unreachable])
    ele, xi, _ = embed(block, points)
    rev_ele, rev_xi, _ = embed(block, points[::-1])

    np.testing.assert_array_equal(rev_ele[::-1], ele)
    np.testing.assert_array_equal(rev_xi[::-1], xi)


def test_chunking_does_not_change_the_answer(block, on_mesh, unreachable):
    """Chunking re-splits the batch, so each chunk's trip count differs."""
    points = np.concatenate([on_mesh, unreachable])
    ele, xi, _ = embed(block, points)
    chunk_ele, chunk_xi, _ = embed(block, points, chunk_size=5)

    np.testing.assert_array_equal(chunk_ele, ele)
    np.testing.assert_array_equal(chunk_xi, xi)


############################################### what the tolerance costs

def test_the_early_exit_agrees_with_running_every_iteration(block, on_mesh):
    ele, xi, _ = embed(block, on_mesh)
    full_ele, full_xi, _ = embed(block, on_mesh, tol=0)

    np.testing.assert_array_equal(ele, full_ele)
    np.testing.assert_allclose(xi, full_xi, atol=EXACT)


def test_points_on_the_mesh_still_embed_to_round_off(block, on_mesh):
    _, _, res = embed(block, on_mesh)

    assert np.linalg.norm(res, axis=-1).max() < DEFAULT_EMBED_TOL


def test_a_tolerance_of_zero_runs_the_iteration_cap(block, on_mesh):
    """``tol=0`` is unreachable, so the cap is what stops the loop."""
    _, one_step, _ = embed(block, on_mesh, tol=0, iterations=1)
    _, many, _ = embed(block, on_mesh, tol=0, iterations=15)

    assert not np.array_equal(one_step, many)


def test_an_unreachable_point_reports_its_residual(block, unreachable):
    """Not converging is reported, not silently exited from."""
    _, _, res = embed(block, unreachable)

    assert np.linalg.norm(res, axis=-1).max() > DEFAULT_EMBED_TOL
