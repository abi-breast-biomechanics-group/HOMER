"""
embedding.py – JAX-compiled point embedding for HOMER meshes.

Hoists the embedding closures (Newton–Raphson solver, coarse nearest-
neighbour search, ``mesh_embed_points`` with its custom JVP) out of
:meth:`Mesh.embed_points` into module-level functions that are built
once per ``generate_mesh()`` call and reused across every
``embed_points`` invocation.

Performance notes
-----------------
* **B1** – The JIT-compiled ``mesh_embed_points`` and its JVP are
  created once in :func:`build_embedding_fn` (called from
  ``generate_mesh``) instead of being redefined on every
  ``embed_points`` call.  This eliminates redundant XLA retracing.
* **B2** – The Newton–Raphson iteration count is passed as a traced
  ``jnp.int32`` value so that changing ``iterations`` between calls
  does *not* trigger a retrace.
* **C1** – For 2-D meshes the coarse NN and NR refinement are fused
  into a single traced block that XLA can pipeline end-to-end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from HOMER.closed_form_matrix_solves import explicit_solve_2x2, explicit_solve_3x3
from HOMER.utils import (approx_closest_indices_Morton_nd, masked_closest_indices,
                         aknn_closest_indices)

if TYPE_CHECKING:
    from HOMER.mesher import MeshField


# ─────────────────────────────────────────────────────────────────────
# Linear-system solve dispatch (ndim is a compile-time constant)
# ─────────────────────────────────────────────────────────────────────

def _linear_solve(A, b, ndim: int):
    """Dispatch to the explicit solver matching the mesh parametric dimension."""
    if ndim == 2:
        return explicit_solve_2x2(A, b)
    elif ndim == 3:
        return explicit_solve_3x3(A, b)
    else:
        c, lower = jax.scipy.linalg.cho_factor(A, lower=True)
        return jax.scipy.linalg.cho_solve((c, lower), b)


# ─────────────────────────────────────────────────────────────────────
# Newton–Raphson single-point solver  (vmap'd over query points)
# ─────────────────────────────────────────────────────────────────────

def _make_nr_solver(mesh: MeshField, ndim: int, robust_init_est: bool):
    """Return a single-point NR function closed over *mesh*.

    The returned function has signature::

        nr_solve(elem, xi0, x_target, lbound, dim_mask,
                 iterations, fit_params) -> ((elem, xi), residual)

    ``iterations`` is a traced ``jnp.int32`` so the XLA trace is
    reused regardless of the Python-level value (B2).
    """
    A_size = ndim  # size of the linear system

    def nr_solve(elem, xi0, x_target, lbound, dim_mask, iterations, fit_params):
        dim_mask = jnp.asarray(dim_mask, dtype=bool)

        # ── body of fori_loop ────────────────────────────────────
        def body_fun(i, state):
            elem, xi, r, r_mag_sq, delta_xi, stepsize = state
            xi_prop = xi + stepsize * delta_xi

            elem_prop, xi_mapped, valid = mesh.topomap(elem, xi_prop)

            x_prop = mesh.evaluate_embeddings(elem_prop, xi_mapped, fit_params=fit_params)[0]
            r_prop = jnp.where(dim_mask, x_target - x_prop, 0.0)
            r_mag_prop_sq = jnp.sum(jnp.square(r_prop))

            accept = r_mag_prop_sq < r_mag_sq

            next_elem = jnp.where(accept, elem_prop, elem)
            next_xi = jnp.where(accept, xi_mapped, xi)
            next_r = jnp.where(accept, r_prop, r)
            next_r_mag_sq = jnp.where(accept, r_mag_prop_sq, r_mag_sq)

            J = mesh.evaluate_jacobians(next_elem, next_xi, fit_params=fit_params)[0]
            J = jnp.where(dim_mask[:, None], J, 0.0)

            Jt = J.T
            A = Jt @ J + jnp.eye(A_size) * 1e-5

            mask = jnp.where(lbound, jnp.zeros_like(next_xi), jnp.ones_like(next_xi))
            diag_mask = jnp.where(lbound, jnp.ones_like(next_xi), jnp.zeros_like(next_xi))

            A_free = A * mask[:, None] * mask[None, :] + jnp.diag(diag_mask)
            b_free = (Jt @ next_r) * mask

            new_delta_xi = _linear_solve(A_free, b_free, ndim)

            next_delta_xi = jnp.where(accept, new_delta_xi, delta_xi)
            next_stepsize = jnp.where(accept, 1.0, stepsize * 0.5)

            return (next_elem, next_xi, next_r, next_r_mag_sq, next_delta_xi, next_stepsize)

        # ── initial state ────────────────────────────────────────
        init_x = mesh.evaluate_embeddings(elem, xi0, fit_params=fit_params)[0]
        init_r = jnp.where(dim_mask, x_target - init_x, 0.0)
        init_r_mag_sq = jnp.sum(jnp.square(init_r))

        if not robust_init_est:
            init_J = mesh.evaluate_jacobians_ele_xi_pair(elem, xi0, fit_params=fit_params)
        else:
            init_J = mesh.eval_numeric_jac_ele_xi_pair(elem, xi0, fit_params=fit_params, step=1e-2)
        init_J = jnp.where(dim_mask[:, None], init_J, 0.0)

        init_Jt = init_J.T
        init_A = init_Jt @ init_J + jnp.eye(A_size) * 1e-7

        init_mask = jnp.where(lbound, jnp.zeros_like(xi0), jnp.ones_like(xi0))
        init_diag_mask = jnp.where(lbound, jnp.ones_like(xi0), jnp.zeros_like(xi0))

        init_A_free = (init_A * init_mask[:, None] * init_mask[None, :]) + jnp.diag(init_diag_mask)
        init_b_free = (init_Jt @ init_r) * init_mask

        init_delta_xi = _linear_solve(init_A_free, init_b_free, ndim)

        # B2: iterations is a *traced* jnp.int32 → one XLA trace for
        # any iteration count, instead of recompiling per Python int.
        init_state = (elem.astype(int), xi0, init_r, init_r_mag_sq, init_delta_xi, 1.0)
        final_state = jax.lax.fori_loop(0, iterations, body_fun, init_state)
        elem_f, xi_f, r_f, _, _, _ = final_state

        return (elem_f, xi_f), r_f

    return nr_solve


# ─────────────────────────────────────────────────────────────────────
# Coarse search helpers
# ─────────────────────────────────────────────────────────────────────

# Distance from 0 or 1 within which a xi coordinate counts as sitting *on*
# that parametric bound.
_BOUND_TOL = 1e-6


def _projected_xi_step(mesh: MeshField, elem_num, xi, residual, fit_params,
                       dim_mask=None):
    """The xi-space step the Newton refinement would take from ``xi``.

    Solved through the normal equations rather than a direct
    ``jnp.linalg.solve`` on the Jacobian: ``J`` has shape
    ``(n_pts, fdim, ndim)`` and is only square when the field dimension
    happens to equal the parametric dimension, so a direct solve raises for
    every field wider than the mesh (a 9-dimensional field on a 3-D mesh,
    say).  The ridge matches the one ``nr_solve`` uses for its own first
    step, so this predicts the direction the refinement actually takes.

    The Jacobian is the analytic one for the same reason.  Seeds for a
    surface embedding sit *on* a parametric bound by construction, and
    ``eval_numeric_jac_ele_xi_pair`` differences across that bound: at
    ``xi = (0.5, 0, 0)`` on a plain cube it disagrees with the analytic
    Jacobian in both magnitude and sign, which is enough to aim the step
    the wrong way and freeze a bound that should have been released.
    """
    J = mesh.evaluate_jacobians_ele_xi_pair(elem_num, xi, fit_params=fit_params)
    if dim_mask is not None:
        residual = jnp.where(dim_mask, residual, 0.0)
        J = jnp.where(dim_mask[..., None], J, 0.0)
    Jt = jnp.swapaxes(J, -1, -2)
    A = Jt @ J + jnp.eye(xi.shape[-1]) * 1e-7
    return jnp.squeeze(jnp.linalg.solve(A, Jt @ residual[..., None]), -1)


def _resolve_active_bounds(mesh: MeshField, elem_num, xi, residual, fit_params,
                           surface: bool, dim_mask=None):
    """Decide which xi directions the NR refinement must hold fixed.

    A seed sitting on an external boundary of its element cannot move
    through that boundary, so ``nr_solve`` freezes the corresponding xi
    direction for the whole refinement.  Which directions those are is a
    property of *where the seed is*, not of where it came from -- and that
    distinction is what used to break surface embedding.

    The surface coarse grid is built face by face with the face perimeters
    included, so a candidate lying on a shared edge or corner is emitted
    once per face that touches it, each copy tagged with a different locked
    direction.  The nearest-neighbour search returns whichever copy it
    reached first, which resolves ties by the order of ``mesh.faces`` and
    not by geometry.  Freezing the wrong direction is unrecoverable: the
    refinement can no longer reach the face the point actually lies on, and
    converges cleanly to the wrong local minimum with no diagnostic.  On a
    single cube at ``grid_res=5`` roughly a third of seeds land on an edge
    or corner, and about half of those were locked to the wrong face.

    So the lock is derived here instead.  ``topo_chain_check`` reports every
    external boundary the seed sits on -- both faces of an edge, all three
    of a corner -- and a boundary is then released when the projected Newton
    step moves *inward* through it, because the refinement wants to leave
    that face and nothing stops it.  This is the same test the volume path
    already applied via its nudged ``delta_xi``, now shared by both.

    A surface embedding additionally needs at least one boundary to stay
    active; releasing them all drops the point into the element interior and
    it stops being a surface embedding.  When every candidate would be
    released, the one the seed is closest to remaining on -- the smallest
    inward step -- is kept.
    """
    at_lo = xi < _BOUND_TOL
    at_hi = xi > 1 - _BOUND_TOL
    external = jax.vmap(mesh.topo_chain_check)(elem_num, xi, at_lo, at_hi)

    step = _projected_xi_step(mesh, elem_num, xi, residual, fit_params, dim_mask)
    # How far the step travels *into* the element through each bound.
    # Positive means the refinement is trying to leave that boundary.
    inward = jnp.where(at_lo, step, 0.0) + jnp.where(at_hi, -step, 0.0)

    keep = external & ~(inward > 0.0)
    if not surface:
        return keep

    stranded = jnp.any(external, axis=-1) & ~jnp.any(keep, axis=-1)
    nearest = jax.nn.one_hot(
        jnp.argmin(jnp.where(external, inward, jnp.inf), axis=-1),
        xi.shape[-1],
    ).astype(bool)
    return jnp.where(stranded[:, None], external & nearest, keep)


def _coarse_search_3d(mesh: MeshField, points, search_pts, fit_params,
                      grid_res: int, dim_mask=None, window_size: int = 16,
                      nn_mode: str = "morton"):
    """Coarse search for 3-D volume meshes (non-surface).

    *search_pts* seeds the nearest-neighbour lookup (masked components
    neutralised, see :func:`_coarse_search_points`); *points* are the real
    query points and drive the residual used to resolve active bounds.
    *nn_mode* selects the coarse lookup (see :func:`_coarse_nn`); outside
    ``"morton"`` the lookup reads *points* directly and *search_pts* is unused.
    """
    xis = jnp.array(mesh.xi_grid(grid_res, 3, boundary_points=True))
    coarse_pts = mesh.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
    i = _coarse_nn(coarse_pts, points, search_pts, dim_mask, nn_mode,
                   window_size)
    test_res = points - coarse_pts[i]
    elem_num = jnp.array(i // xis.shape[0])
    init_xi = xis[i % jnp.array(xis.shape[0])]

    mf_pt = _resolve_active_bounds(mesh, elem_num, init_xi, test_res,
                                   fit_params, surface=False, dim_mask=dim_mask)
    return elem_num, init_xi, test_res, mf_pt


def _coarse_search_3d_surface(mesh: MeshField, points, search_pts, fit_params,
                              grid_res: int, dim_mask=None,
                              nn_mode: str = "morton", window_size: int = 32):
    """Coarse search restricted to the surface faces of a 3-D mesh.

    *nn_mode* selects the coarse lookup (see :func:`_coarse_nn`); outside
    ``"morton"`` the lookup reads *points* directly and *search_pts* is unused.
    """
    face_pts, elem_pts, xi_pts = [], [], []
    xi3grid = mesh.xi_grid(res=grid_res, dim=3, surface=True).reshape(3, 2, -1, 3)
    for face in mesh.faces:
        grid_def = xi3grid[face[1], face[2]]
        elem_pts.append(np.ones(grid_def.shape[0]) * face[0])
        xi_pts.append(grid_def)
        face_pts.append(mesh.evaluate_embeddings(jnp.array([face[0]]), grid_def, fit_params=fit_params))

    coarse_pts = jnp.concatenate(face_pts, axis=0)
    elems = jnp.concatenate(elem_pts, axis=0)
    xis = jnp.concatenate(xi_pts, axis=0)
    i = _coarse_nn(coarse_pts, points, search_pts, dim_mask, nn_mode,
                   window_size)
    test_res = points - coarse_pts[i]
    elem_num = elems[i].astype(int)
    init_xi = xis[i]

    # Deliberately *not* taken from the face that contributed this candidate
    # -- see _resolve_active_bounds for why that mis-locks edge seeds.
    mf_pt = _resolve_active_bounds(mesh, elem_num, init_xi, test_res,
                                   fit_params, surface=True, dim_mask=dim_mask)
    return elem_num, init_xi, test_res, mf_pt


# ─────────────────────────────────────────────────────────────────────
# dim_mask normalisation
# ─────────────────────────────────────────────────────────────────────

def _coarse_search_points(points, dim_mask):
    """Query points with masked components neutralised for the coarse search.

    The Newton-Raphson refinement respects ``dim_mask``, but the coarse
    nearest-neighbour search that seeds it does not: it matches the *whole*
    query vector against the candidate grid.  Callers normally park a
    sentinel (``-1``, ``NaN``-like magic numbers) in the components a point
    does not constrain, and those sentinels then dominate the coarse
    distance — the initial element/xi is chosen by the sentinel pattern
    rather than the geometry, and the NR iterations cannot always recover.
    The visible symptom is a residual that jumps discontinuously as
    parameters move and correspondences flip.

    Replacing each masked component with the mean of that dimension over the
    points which *do* constrain it keeps the query inside the coordinate
    range, so the coarse distance is driven by the active dimensions.  The
    residual itself is untouched — masked components are zeroed in
    ``nr_solve`` regardless — so this only changes the initial guess.
    """
    counts = jnp.sum(dim_mask, axis=0)
    totals = jnp.sum(jnp.where(dim_mask, points, 0.0), axis=0)
    col_mean = jnp.where(counts > 0, totals / jnp.maximum(counts, 1), 0.0)
    return jnp.where(dim_mask, points, col_mean[None, :])


def _as_dim_mask(dim_mask, points):
    """Normalise *dim_mask* to a bool array shaped like *points*.

    ``dim_mask`` states which physical dimensions of each query point
    participate in the embedding residual.  It is a *static* statement
    about the problem, not a differentiable quantity, so it is forced to
    ``bool`` here: JAX gives boolean inputs a ``float0`` tangent, which
    makes "no tangent with respect to the mask" the only representable
    answer and stops a float mask from silently leaking a derivative
    into the custom JVP rule.

    Accepts ``None`` (all dimensions active) or anything broadcastable
    to ``points.shape`` — a per-dimension ``(fdim,)`` vector is broadcast
    across every point.
    """
    if dim_mask is None:
        return jnp.ones(points.shape, dtype=bool)
    dim_mask = jnp.asarray(dim_mask).astype(bool)
    return jnp.broadcast_to(dim_mask, points.shape)


# A Morton code spends a fixed 32-bit budget across the field dimension, so
# it keeps 32 // fdim bits per dimension.  At fdim <= 3 that is >= 10 bits and
# the Z-curve is adequate (and much the cheapest option, which matters because
# that is also the case with the largest candidate sets).  Beyond it, measured
# recall against an exact search falls away fast -- 88.9% at fdim 3, 49.1% at
# 6, 21.0% at 12 -- with unbounded misses, so the distance-ranked search takes
# over.  See `aknn_closest_indices`.
MORTON_MAX_FDIM = 3


def _coarse_nn(coarse_pts, points, search_pts, dim_mask, mode, window_size):
    """Run whichever coarse nearest-neighbour search *mode* selects.

    ``mode`` is a compile-time string chosen by :func:`_coarse_nn_mode`:

    ``"masked"``
        Exact search under ``dim_mask``.  A Z-curve code describes a whole
        coordinate vector, so it cannot express "these components only".
    ``"aknn"``
        Distance-ranked search, for high-dimensional fields where the Morton
        bit budget has run out.
    ``"morton"``
        The cheap Z-curve search, for low-dimensional fields.
    """
    if mode == "masked":
        return masked_closest_indices(coarse_pts, points, dim_mask)
    if mode == "aknn":
        return aknn_closest_indices(coarse_pts, points)
    return approx_closest_indices_Morton_nd(coarse_pts, search_pts,
                                            window_size=window_size)


def _coarse_nn_mode(dim_mask, fdim: int) -> str:
    """Pick the coarse search for this call.  Static; see :func:`_coarse_nn`."""
    if _needs_masked_nn(dim_mask):
        return "masked"
    return "aknn" if fdim > MORTON_MAX_FDIM else "morton"


def _needs_masked_nn(dim_mask) -> bool:
    """Whether the coarse search has to run the exact masked lookup.

    A Morton code describes a whole coordinate vector, so the Z-curve search
    cannot express "ignore these components for this query".  With every
    dimension active there is nothing to ignore and the cheap approximate
    search is exact enough; as soon as any component is masked off, the
    substitute value parked there (see :func:`_coarse_search_points`) steers
    the match, and for a multi-state field the wrong seed is a different
    minimum rather than a slightly worse one.

    ``dim_mask`` is documented as a *static* statement about the problem, so
    reading its value here is legitimate.  If a caller none the less traces
    over it, the value is not available at trace time -- fall back to the
    exact search, which is correct for either mask.
    """
    try:
        return not bool(jnp.all(dim_mask))
    except jax.errors.TracerBoolConversionError:
        return True


# ─────────────────────────────────────────────────────────────────────
# JVP helpers (for differentiable embedding)
# ─────────────────────────────────────────────────────────────────────

def _make_jvp_helpers(mesh: MeshField, approx_jac: bool):
    """Build the ``D``, ``g``, and ``embed_single_jvp`` functions."""

    @jax.jit
    def D(eles, xis, x, params, dim_mask):
        diff = x - mesh.evaluate_embeddings([eles], xis, fit_params=params)
        diff = jnp.where(dim_mask, diff, 0.0)
        return jnp.sum(diff ** 2) / 2

    @jax.jit
    def g(eles, xis, x, params, dim_mask):
        return jax.grad(D, argnums=1)(eles, xis, x, params, dim_mask)

    @jax.jit
    def embed_single_jvp(ele, xi, point, params, point_dot, param_dot, dim_mask):
        local_H = jax.jacobian(g, argnums=1)(ele, xi, point, params, dim_mask)
        comb_product = jax.jvp(
            lambda w, p: g(ele, xi, w, p, dim_mask),
            (point, params), (point_dot, param_dot)
        )[1]

        active_mask = jnp.isclose(xi, 1.0, atol=1e-5) | jnp.isclose(xi, 0, atol=1e-5)
        free_mask = ~active_mask
        masked_H = (
            jnp.where(free_mask[:, None] * free_mask[None, :], local_H, 0.0)
            + jnp.diag(jnp.where(active_mask, 1.0, 0.0))
        )
        masked_comb_product = jnp.where(active_mask, 0.0, comb_product)

        # `dim_mask` can switch off enough residual components to leave
        # the xi solve under-determined — in the limit an all-False row
        # makes D identically zero, so local_H is exactly zero and a
        # plain solve returns NaN.  Those null directions carry no
        # information about how xi moves, so take the minimum-norm
        # (pseudo-inverse) solution, which puts nothing along them and
        # agrees with `solve` whenever masked_H is invertible.
        xi_dot = -(jnp.linalg.pinv(masked_H, hermitian=True)
                   @ masked_comb_product) * (0 if approx_jac else 1)
        w_dot = jax.jvp(
            lambda x, p: mesh.evaluate_embeddings([ele], x, fit_params=p),
            (xi, params), (xi_dot, param_dot)
        )[1][0]

        r_dot = jnp.where(dim_mask, point_dot - w_dot, 0.0)
        return xi_dot, r_dot

    return embed_single_jvp


# ─────────────────────────────────────────────────────────────────────
# Public builder – called once from generate_mesh()
# ─────────────────────────────────────────────────────────────────────

def build_embedding_fn(mesh: MeshField, *, approx_jac: bool = False,
                       robust_init_est: bool = False):
    """Create a JIT-compiled embedding function closed over *mesh*.

    Called once from :meth:`MeshField.generate_mesh`.  Returns a
    callable with signature::

        embed(points, fit_params, dim_mask, init_elexi, surface_embed,
              grid_res, iterations)
            -> ((elem_num, embedded), residual)

    The callable delegates to a ``@jax.custom_jvp`` function that is
    JIT-compiled once and reused, avoiding repeated XLA retracing.

    Parameters
    ----------
    mesh :
        The :class:`~HOMER.mesher.MeshField` instance.
    approx_jac :
        If ``True``, drops the sliding term from the residual gradient
        estimation for the custom JVP (see ``embed_points`` docs).
    robust_init_est :
        If ``True``, uses a numeric Jacobian for the initial NR step.
    """
    ndim = mesh.ndim

    # Build the NR solver (closed over mesh)
    nr_solve = _make_nr_solver(mesh, ndim, robust_init_est)

    # Build the JVP helpers
    embed_single_jvp = _make_jvp_helpers(mesh, approx_jac)

    # ── Coarse-search + NR (one traced block per branch) ─────────
    #
    # C1: For 2-D meshes the coarse search and NR refinement are
    # merged into a single function so XLA traces the full pipeline
    # (coarse eval → Morton NN → NR) as one fused block, avoiding
    # the materialisation boundary that used to separate them.

    def _run_coarse_2d(points, fit_params, dim_mask, grid_res, iterations,
                       window_size, nn_mode):
        """C1 fused path for 2-D meshes."""
        xis = jnp.asarray(mesh.xi_grid(grid_res, 2, boundary_points=False))
        coarse_pts = mesh.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
        search_pts = (_coarse_search_points(points, dim_mask)
                      if nn_mode == "morton" else None)
        i = _coarse_nn(coarse_pts, points, search_pts, dim_mask, nn_mode,
                       window_size)
        elem_num = i // xis.shape[0]
        init_xi = xis[i % xis.shape[0]]

        at_lo, at_hi = init_xi < 1e-6, init_xi > 1 - 1e-6
        mf_pt = jax.vmap(mesh.topo_chain_check)(elem_num, init_xi, at_lo, at_hi)

        (en, emb), res = jax.vmap(
            lambda elem, xi, target, lbound, lmask: nr_solve(
                elem, xi, target, lbound, lmask, iterations, fit_params
            )
        )(elem_num, init_xi, points, mf_pt, dim_mask)
        return (en, emb), res

    def _run_coarse_3d(points, fit_params, dim_mask, grid_res, iterations,
                       surface_embed, window_size, nn_mode):
        """Coarse + NR path for 3-D meshes."""
        search_pts = (_coarse_search_points(points, dim_mask)
                      if nn_mode == "morton" else None)
        if surface_embed:
            elem_num, init_xi, _, mf_pt = _coarse_search_3d_surface(
                mesh, points, search_pts, fit_params, grid_res,
                dim_mask=dim_mask, nn_mode=nn_mode,
            )
        else:
            elem_num, init_xi, _, mf_pt = _coarse_search_3d(
                mesh, points, search_pts, fit_params, grid_res,
                dim_mask=dim_mask, window_size=window_size,
                nn_mode=nn_mode,
            )

        (en, emb), res = jax.vmap(
            lambda elem, xi, target, lbound, lmask: nr_solve(
                elem, xi, target, lbound, lmask, iterations, fit_params
            )
        )(elem_num, init_xi, points, mf_pt, dim_mask)
        return (en, emb), res

    # ── The core JIT function (takes only JAX arrays) ────────────
    #
    # Python-level branching (use_init_elexi, surface_embed, grid_res)
    # happens *outside* this function.  Only `iterations` is traced
    # inside the JIT (B2: different iteration counts share one trace).

    def _make_jit_embed(use_init_elexi: bool, surface_embed: bool,
                        grid_res: int, window_size: int, nn_mode: str):
        """Return a JIT-compiled ``@custom_jvp`` function for one branch."""

        @jax.custom_jvp
        @jax.jit
        def _embed_jit(points, fit_params, dim_mask,
                       init_elexi_elem, init_elexi_xi, iterations):
            # dim_mask is normalised to a bool (n_pts, fdim) array by
            # `embed` before it ever reaches here (see _as_dim_mask).
            points = jnp.atleast_2d(points)

            if use_init_elexi:
                elem_num = jnp.atleast_1d(init_elexi_elem)
                init_xi = jnp.atleast_2d(init_elexi_xi)
                test_res = points - mesh.evaluate_embeddings_ele_xi_pair(
                    elem_num, init_xi, fit_params=fit_params)

                # A caller-supplied seed is as free to land on an edge or a
                # corner as a coarse one, so it gets the same geometric
                # resolution rather than locking every bound it touches.
                mf_pt = _resolve_active_bounds(mesh, elem_num, init_xi, test_res,
                                               fit_params, surface=surface_embed,
                                               dim_mask=dim_mask)

                (en, emb), res = jax.vmap(
                    lambda elem, xi, target, lbound, lmask: nr_solve(
                        elem, xi, target, lbound, lmask, iterations, fit_params
                    )
                )(elem_num, init_xi, points, mf_pt, dim_mask)
                return (en, emb), res
            else:
                if ndim == 2:
                    return _run_coarse_2d(points, fit_params, dim_mask,
                                         grid_res, iterations, window_size,
                                         nn_mode)
                else:
                    return _run_coarse_3d(points, fit_params, dim_mask,
                                         grid_res, iterations,
                                         surface_embed, window_size,
                                         nn_mode)

        @_embed_jit.defjvp
        def _embed_jvp(primal, tangent):
            primal_out = _embed_jit(*primal)
            (ele, xi), _ = primal_out

            points, params, dm = primal[0], primal[1], primal[2]
            point_dot, param_dot = tangent[0], tangent[1]
            # tangent[2] is the dim_mask tangent.  The mask is a static
            # statement about *which residual components exist*, not a
            # differentiable quantity, so it carries no meaningful
            # tangent and is deliberately ignored.  Normalising it to
            # bool in `embed` makes JAX hand us a float0 zero here.

            points = jnp.atleast_2d(points)
            dm = _as_dim_mask(dm, points)

            # dm is per-point, shape (n_pts, fdim): it must be mapped
            # over the point axis exactly like the primal path does,
            # otherwise every point sees the whole mask and r_dot comes
            # back with a spurious leading (n_pts,) axis.
            xi_dot, r_dot = jax.vmap(
                lambda e, x, w, w_dot, m: embed_single_jvp(
                    e, x, w, params, w_dot, param_dot, m
                )
            )(ele, xi, points, point_dot, dm)

            tangent_out = (
                (jnp.zeros_like(ele, dtype=jax.float0), xi_dot),
                r_dot,
            )
            return primal_out, tangent_out

        return _embed_jit

    # ── Pre-compile the default branch (coarse search, no surface) ──
    # Additional branches are compiled on demand and cached.
    _jit_cache: dict[tuple[bool, bool, int, int, str], object] = {}

    def _get_jit(use_init: bool, surface: bool, gres: int, wsize: int,
                 nn_mode: str):
        key = (use_init, surface, gres, wsize, nn_mode)
        fn = _jit_cache.get(key)
        if fn is None:
            fn = _make_jit_embed(use_init, surface, gres, wsize, nn_mode)
            _jit_cache[key] = fn
        return fn

    # ── Public entry point ───────────────────────────────────────
    def embed(points, fit_params, dim_mask,
              init_elexi, surface_embed, grid_res, iterations,
              chunk_size=None, window_size=16):
        """Dispatch to the appropriate JIT-compiled embedding function.

        Parameters that control Python-level branching
        (``init_elexi``, ``surface_embed``, ``grid_res``,
        ``window_size``) select the cached JIT trace.  ``iterations``
        is passed as a traced ``jnp.int32`` so it shares one trace
        across all values (B2).

        When *chunk_size* is set and the number of query points exceeds
        it, the point set is split into chunks that are processed
        sequentially and concatenated.  This bounds peak memory to
        ``O(chunk_size)`` instead of ``O(n_pts)``, preventing swap
        usage on large inputs.
        """
        points = jnp.atleast_2d(points)
        n_pts = points.shape[0]

        # Resolve the mask once, up front, so every downstream path
        # (chunked or not, JIT primal or custom JVP) sees the same
        # bool (n_pts, fdim) array and never has to re-handle None.
        dim_mask = _as_dim_mask(dim_mask, points)

        use_init = init_elexi is not None

        # Only the coarse search cares -- a caller-supplied seed skips it.
        nn_mode = ("morton" if use_init
                   else _coarse_nn_mode(dim_mask, points.shape[-1]))

        if chunk_size is not None and n_pts > chunk_size:
            en_parts, emb_parts, res_parts = [], [], []
            for start in range(0, n_pts, chunk_size):
                end = min(start + chunk_size, n_pts)
                p_chunk = points[start:end]

                if use_init:
                    ie_elem = jnp.atleast_1d(jnp.asarray(init_elexi[0][start:end]))
                    ie_xi = jnp.atleast_2d(jnp.asarray(init_elexi[1][start:end]))
                else:
                    ie_elem = jnp.zeros(p_chunk.shape[0], dtype=jnp.int32)
                    ie_xi = jnp.zeros((p_chunk.shape[0], ndim))

                dm_chunk = dim_mask[start:end]

                fn = _get_jit(use_init, bool(surface_embed),
                              int(grid_res), int(window_size), nn_mode)
                (en, emb), res = fn(p_chunk, fit_params, dm_chunk,
                                    ie_elem, ie_xi, jnp.int32(iterations))
                en_parts.append(en)
                emb_parts.append(emb)
                res_parts.append(res)

            return ((jnp.concatenate(en_parts),
                     jnp.concatenate(emb_parts)),
                    jnp.concatenate(res_parts))

        if use_init:
            ie_elem = jnp.atleast_1d(jnp.asarray(init_elexi[0]))
            ie_xi = jnp.atleast_2d(jnp.asarray(init_elexi[1]))
        else:
            ie_elem = jnp.zeros(n_pts, dtype=jnp.int32)
            ie_xi = jnp.zeros((n_pts, ndim))

        fn = _get_jit(use_init, bool(surface_embed),
                      int(grid_res), int(window_size), nn_mode)
        return fn(points, fit_params, dim_mask,
                  ie_elem, ie_xi, jnp.int32(iterations))

    return embed
