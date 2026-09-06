"""Regression: the coarse seed must honour ``dim_mask`` for multi-state fields.

A Morton code describes a whole coordinate vector, so the Z-curve coarse
search cannot express "ignore these components for this query".  It used to
be fed the query with masked components replaced by that dimension's mean
(``_coarse_search_points``), which leaves the substituted values contributing
to the match distance.

For a single-state field that is harmless -- every state agrees on one
surface point, so the argmin is unique and Newton-Raphson recovers it from a
mediocre seed.  For a masked multi-state field the minimised quantity

    d(elem, xi) = sum_over_active_states || target_s - field_s(elem, xi) ||^2

is a compromise across states with many competing local minima, and the seed
decides which one the local solver lands in.  A mis-steered seed then returns
a point embedded on the wrong part of the surface, not merely an imprecise
one -- and raising ``grid_res`` made it worse, by offering more seeds to a
landscape with more spurious basins.

The check is one-sided and cannot produce a false positive: the brute-force
value is itself only an upper bound on the true minimum, so a cold
``embed_points`` result coming back *above* it has provably missed the
global argmin.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from HOMER import cube
from HOMER.utils import (rodrigues_exp, masked_closest_indices,
                         aknn_closest_indices)
from HOMER.embedding import _coarse_nn_mode

N_PTS = 200
NOISE = 0.02
BRUTE_RES = 24        # brute-force samples per axis, per face
TOL = 1e-6            # slack, so solver noise is never counted as a failure


def _surface_grid(res):
    """Dense (elem, xi) candidates covering all six faces of the unit cube."""
    a = np.linspace(0.0, 1.0, res)
    u, v = np.meshgrid(a, a, indexing="ij")
    xis = []
    for axis in range(3):
        for side in (0.0, 1.0):
            xi = np.empty((res * res, 3))
            free = [k for k in range(3) if k != axis]
            xi[:, free[0]] = u.ravel()
            xi[:, free[1]] = v.ravel()
            xi[:, axis] = side
            xis.append(xi)
    xi = np.concatenate(xis)
    return np.zeros(len(xi), dtype=int), xi


def _build(n_states, rng):
    """Cube mesh carrying a 3*n_states field whose states genuinely differ."""
    mesh = cube(scale=2.0)
    mesh.generate_mesh()
    gen = np.asarray(mesh.optimisable_param_array).reshape(-1, 3)
    mesh.new_field("f", field_dimension=3 * n_states,
                   field_params=np.tile(gen, (1, n_states)).ravel())
    blocks = [gen @ np.asarray(rodrigues_exp(jnp.array(rng.normal(0, 0.35, 3)))).T
              for _ in range(n_states)]
    # field layout is node-major: [state0 xyz, state1 xyz, ...] per coordinate
    block = np.stack(blocks).transpose(1, 0, 2).reshape(-1)
    return mesh["f"], jnp.array(block)


def _queries(field, fp, n_states, rng):
    """Noisy multi-state measurements of points that lie on the surface.

    Each state is perturbed independently, so no single (elem, xi) reproduces
    every state -- which is what makes the argmin a genuine compromise.
    """
    elem, xi = _surface_grid(BRUTE_RES)
    pick = rng.integers(0, len(xi), N_PTS)
    clean = np.asarray(field.evaluate_embeddings_ele_xi_pair(
        jnp.array(elem[pick]), jnp.array(xi[pick]), fit_params=fp))
    tgt = clean + rng.normal(0, NOISE, clean.shape)

    # every point keeps at least one state and drops the others entirely
    mask = np.ones((N_PTS, n_states), dtype=bool)
    if n_states > 1:
        for i in range(N_PTS):
            drop = rng.random(n_states) < 0.25
            if drop.all():
                drop[rng.integers(n_states)] = False
            mask[i] = ~drop
    return tgt, np.repeat(mask, 3, axis=1)


def _residual(field, tgt, fp, dim_mask, grid_res, iterations, init=None):
    kw = {"init_elexi": init} if init is not None else {}
    _, r = field.embed_points(jnp.array(tgt), fit_params=fp, surface_embed=True,
                              grid_res=grid_res, iterations=iterations,
                              dim_mask=jnp.array(dim_mask),
                              return_residual=True, **kw)
    return np.sqrt(np.sum((np.asarray(r) * dim_mask) ** 2, axis=1))


def _brute_force_seed(field, fp, tgt, dim_mask):
    """Global argmin of the masked multi-state distance over a dense grid."""
    elem, xi = _surface_grid(BRUTE_RES)
    cand = field.evaluate_embeddings_ele_xi_pair(
        jnp.array(elem), jnp.array(xi), fit_params=fp)
    best = np.asarray(masked_closest_indices(cand, jnp.array(tgt),
                                             jnp.array(dim_mask)))
    return jnp.array(elem[best]), jnp.array(xi[best])


@pytest.mark.parametrize("n_states", [1, 2, 3, 4])
@pytest.mark.parametrize("grid_res", [5, 9, 20])
def test_cold_embed_finds_the_global_argmin(n_states, grid_res):
    rng = np.random.default_rng(0)
    field, fp = _build(n_states, rng)
    tgt, dim_mask = _queries(field, fp, n_states, rng)

    ref = _residual(field, tgt, fp, dim_mask, 5, 60,
                    init=_brute_force_seed(field, fp, tgt, dim_mask))
    cold = _residual(field, tgt, fp, dim_mask, grid_res, 15)

    excess = cold - ref
    assert np.max(excess) <= TOL, (
        f"{np.mean(excess > TOL):.1%} of points beat by brute force; "
        f"worst excess {excess.max():.3e}"
    )


def test_masked_closest_indices_ignores_masked_dimensions():
    """The masked lookup must be exact, and blind to inactive components."""
    rng = np.random.default_rng(1)
    A = jnp.array(rng.normal(size=(200, 12)))
    B = np.asarray(rng.normal(size=(50, 12)))
    mask = rng.random((50, 12)) < 0.7

    got = np.asarray(masked_closest_indices(A, jnp.array(B), jnp.array(mask)))
    d = ((np.asarray(A)[None] - B[:, None]) ** 2 * mask[:, None, :]).sum(-1)
    assert np.array_equal(got, d.argmin(axis=1))

    # garbage in the masked components must not move the answer
    poisoned = np.where(mask, B, 1e6)
    assert np.array_equal(
        got,
        np.asarray(masked_closest_indices(A, jnp.array(poisoned), jnp.array(mask))),
    )


# ─────────────────────────────────────────────────────────────────────
# Unmasked high-dimensional fields: the Morton bit budget
# ─────────────────────────────────────────────────────────────────────

def test_coarse_search_mode_selection():
    """Each field shape must reach the search that can actually handle it."""
    full3 = jnp.ones((8, 3), dtype=bool)
    full12 = jnp.ones((8, 12), dtype=bool)
    partial = jnp.array(np.tile([True, False, True], (8, 4)))

    # 10 bits per dimension: the cheap Z-curve is adequate.
    assert _coarse_nn_mode(full3, 3) == "morton"
    # 2 bits per dimension: the Z-curve has stopped discriminating.
    assert _coarse_nn_mode(full12, 12) == "aknn"
    # No code can express a per-query subset of dimensions.
    assert _coarse_nn_mode(partial, 12) == "masked"


@pytest.mark.parametrize("fdim", [6, 12, 15])
def test_aknn_matches_exact_search(fdim):
    """The distance-ranked search must agree with an exact one."""
    rng = np.random.default_rng(2)
    A = jnp.array(rng.normal(size=(2400, fdim)), dtype=jnp.float32)
    B = jnp.array(rng.normal(size=(3000, fdim)), dtype=jnp.float32)

    got = np.asarray(aknn_closest_indices(A, B))
    ref = np.asarray(masked_closest_indices(A, B, jnp.ones(B.shape, dtype=bool)))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("n_states", [2, 4])
def test_unmasked_offsurface_embed_on_refined_mesh(n_states):
    """Unmasked, high-fdim, off-surface points on a multi-element mesh.

    This is where the Morton seed used to break down: an on-surface query has
    one basin and Newton-Raphson reaches it from almost any seed, but a point
    well off the surface of a refined mesh can be seeded into the wrong
    element, and the refinement stays where it was put.
    """
    rng = np.random.default_rng(0)
    mesh = cube(scale=2.0)
    mesh.refine(2)
    mesh.generate_mesh()
    gen = np.asarray(mesh.optimisable_param_array).reshape(-1, 3)
    mesh.new_field("f", field_dimension=3 * n_states,
                   field_params=np.tile(gen, (1, n_states)).ravel())
    field = mesh["f"]
    blocks = [gen @ np.asarray(rodrigues_exp(jnp.array(rng.normal(0, 0.35, 3)))).T
              for _ in range(n_states)]
    fp = jnp.array(np.stack(blocks).transpose(1, 0, 2).reshape(-1))

    def candidates(res):
        g = mesh.xi_grid(res=res, dim=3, surface=True).reshape(3, 2, -1, 3)
        e, x = [], []
        for face in mesh.faces:
            gd = g[face[1], face[2]]
            e.append(np.full(gd.shape[0], face[0]))
            x.append(gd)
        e, x = np.concatenate(e).astype(int), np.concatenate(x)
        return e, x, field.evaluate_embeddings_ele_xi_pair(
            jnp.array(e), jnp.array(x), fit_params=fp)

    e_b, x_b, c_b = candidates(16)
    pick = rng.integers(0, len(x_b), N_PTS)
    # noise far larger than NOISE: these points sit well off the surface
    tgt = np.asarray(c_b)[pick] + rng.normal(0, 0.30, (N_PTS, 3 * n_states))
    full = np.ones(tgt.shape, dtype=bool)

    seed = np.asarray(masked_closest_indices(c_b, jnp.array(tgt), jnp.array(full)))
    ref = _residual(field, tgt, fp, full, 5, 60,
                    init=(jnp.array(e_b[seed]), jnp.array(x_b[seed])))
    cold = _residual(field, tgt, fp, full, 5, 15)

    excess = cold - ref
    assert np.max(excess) <= TOL, (
        f"{np.mean(excess > TOL):.1%} of points beat by brute force; "
        f"worst excess {excess.max():.3e}"
    )
