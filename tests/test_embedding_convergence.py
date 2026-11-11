"""The convergence exit in the Newton-Raphson refinement.

``embed_points`` stops refining once the residual is inside ``tol`` instead of
always running ``iterations`` steps.  The refinement is vectorised over the
query points, so a single ``lax.while_loop`` trip count is shared by the whole
batch -- which is the thing worth testing.  A converged point must be *frozen*,
not merely ignored: if the batched loop let finished lanes keep stepping until
the slowest one caught up, an answer would depend on which other points
happened to be embedded alongside it.

What the batch is *made of* must not matter at all, and that is asserted
exactly: reordering it, or adding points that never converge, reproduces every
answer bit for bit.  How *many* points are in it is a weaker guarantee.  The
loop stops as soon as the residual is inside ``tol``, and a batch of one and a
batch of two dozen are different XLA programs that can cross that line on
iterates a rounding apart -- both inside the tolerance, which is all the
tolerance claims.  So batch size is held to the position the answer names
rather than to the bits of the answer.

The tolerance itself is checked to be what it claims -- an early exit at
float32 round-off, not an accuracy setting.
"""

import numpy as np
import pytest

from HOMER.embedding import DEFAULT_EMBED_TOL
from HOMER.geometry import cubeMNO

from _helpers import EXACT, arr

#one float32 rounding of a parametric coordinate.  Not an accuracy budget: it
#is the width of the last bit, and the observed spread is half of it.
XI_ROUNDING = float(np.finfo(np.float32).eps)


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


@pytest.fixture(scope="module")
def near_mesh(block, on_mesh):
    """Points just outside the block: a closest point to find, but no zero residual there.

    A point merely nudged inside a volume mesh still embeds exactly, so these
    are set beyond one face of it.
    """
    points = on_mesh.copy()
    top = max(node.loc[0] for node in block.nodes)
    points[:, 0] = top + 0.01 + 0.02 * np.random.default_rng(3).random(len(points))
    return points


def embed(mesh, points, **kw):
    (ele, xi), res = mesh.embed_points(np.atleast_2d(points),
                                       return_residual=True, **kw)
    return arr(ele), arr(xi), arr(res)


############################################### batch composition

def test_a_point_embeds_the_same_alone_as_in_a_batch(block, on_mesh):
    """Batch size may move the last bit of xi.  It may not move the point.

    ``(element, xi)`` is a *way of writing* a position, and a point on a shared
    face has more than one: the same place is xi = 1 of one element and xi = 0
    of its neighbour, and a rounding in the refinement is enough to decide
    which of the two comes back.  Neither is wrong, so the position is what is
    compared.  Where the two calls do agree on the element, the parametric
    coordinate has to agree to a float32 rounding of itself.
    """
    ele, xi, _ = embed(block, on_mesh)
    batched = arr(block.evaluate_embeddings_ele_xi_pair(ele, xi))

    for i, point in enumerate(on_mesh):
        solo_ele, solo_xi, _ = embed(block, point)
        solo = arr(block.evaluate_embeddings_ele_xi_pair(solo_ele, solo_xi)).reshape(-1)

        np.testing.assert_allclose(solo, batched[i], atol=EXACT)
        if solo_ele[0] == ele[i]:
            np.testing.assert_allclose(solo_xi[0], xi[i], atol=XI_ROUNDING)


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


def test_a_point_off_the_mesh_stops_once_it_stops_moving(block, near_mesh):
    """Its residual never reaches the tolerance, so it is the step that stops it.

    Stopping before the cap is what makes a larger cap change nothing.  Waiting
    for the line search to stall instead took three steps here, so a cap of two
    tells the two apart.
    """
    capped = embed(block, near_mesh, iterations=2)
    uncapped = embed(block, near_mesh, iterations=100)

    for a, b in zip(capped, uncapped):
        np.testing.assert_array_equal(a, b)
    assert (np.linalg.norm(capped[2], axis=-1) > DEFAULT_EMBED_TOL).all()


def test_stopping_off_the_mesh_costs_nothing_in_accuracy(block, near_mesh):
    ele, xi, res = embed(block, near_mesh)
    full_ele, full_xi, full_res = embed(block, near_mesh, tol=0)

    np.testing.assert_array_equal(ele, full_ele)
    np.testing.assert_allclose(xi, full_xi, atol=EXACT)
    np.testing.assert_allclose(np.linalg.norm(res, axis=-1), np.linalg.norm(full_res, axis=-1),
                               atol=DEFAULT_EMBED_TOL)


def test_the_early_exits_give_up_on_no_point_full_iterations_would_embed():
    """A degenerate element can damp the step to nothing however far a point has to go.

    The example cube has zero nodal derivatives at some nodes, so its Jacobian
    collapses on edges there; a seed on one takes a vanishing first step and
    then creeps off as the element recovers.  A short step must not read as
    having arrived.
    """
    from HOMER.examples import hermite_cube
    mesh = hermite_cube()
    mesh.refine(4)
    rng = np.random.default_rng(11)
    points = arr(mesh.evaluate_embeddings_ele_xi_pair(rng.integers(len(mesh.elements), size=4000),
                                                      rng.random((4000, 3))))

    _, _, early = embed(mesh, points)
    _, _, full = embed(mesh, points, tol=0)

    embedded_in_full = np.linalg.norm(full, axis=-1) < 1e-5
    assert (np.linalg.norm(early, axis=-1)[embedded_in_full] < 1e-5).all()


def test_points_inside_a_curved_block_all_embed():
    """The example cube's zero nodal derivatives collapse some faces of its elements.

    On such a face the Jacobian column across it vanishes, so a seed there gets
    no first-order sign of the point inside, and a lock decided once at the seed
    held it on the face: 7% of the points failed.  Locks are now re-decided as
    the point moves, never along a direction the Jacobian has collapsed in, and
    a blind direction is probed off the face instead.
    """
    from HOMER.examples import hermite_cube
    mesh = hermite_cube()
    mesh.refine(2)
    rng = np.random.default_rng(12)
    points = arr(mesh.evaluate_embeddings_ele_xi_pair(rng.integers(len(mesh.elements), size=4000),
                                                      rng.random((4000, 3))))

    _, _, res = embed(mesh, points)

    assert np.mean(np.linalg.norm(res, axis=-1) > 1e-5) < 1e-3


def test_an_unreachable_point_reports_its_residual(block, unreachable):
    """Not converging is reported, not silently exited from."""
    _, _, res = embed(block, unreachable)

    assert np.linalg.norm(res, axis=-1).max() > DEFAULT_EMBED_TOL
