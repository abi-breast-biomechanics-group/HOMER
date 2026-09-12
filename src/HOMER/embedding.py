"""
embedding.py – JAX-compiled point embedding for HOMER meshes.

Hoists the embedding closures (Newton–Raphson solver, coarse nearest-
neighbour search, ``mesh_embed_points`` with its custom JVP) out of
:meth:`Mesh.embed_points` into module-level functions that are built
once per ``generate_mesh()`` call and reused across every
``embed_points`` invocation.

**Performance notes**

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
    from HOMER.mesh import MeshField


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


def _solve_xi_tangent(H, b, ndim: int, partial_mask: bool):
    """Solve ``H xi_dot = b`` for the custom JVP, as robustly as the mask needs.

    ``dim_mask`` can switch off enough residual components to leave the xi
    solve under-determined -- in the limit an all-False row makes ``D``
    identically zero, so ``H`` is exactly zero and a plain solve returns NaN.
    Those null directions carry no information about how xi moves, so the
    masked case takes the minimum-norm (pseudo-inverse) solution, which puts
    nothing along them.

    With every dimension active there are no such directions to worry about,
    and the pseudo-inverse is pure overhead -- an eigendecomposition per query
    point, which measured as more than half the whole tangent computation
    (24.2 ms of a 20k-point batch, against 12.5 ms for the direct solve).  So
    that case gets the explicit solve, with the same ridge the refinement
    itself uses to cover a degenerate element.  Checked against the
    pseudo-inverse on a plain and a surface embedding and on the derivative
    with respect to the query points, the two agree to float32 round-off; it
    is only the masked cases that separate them, by ~1e-3, and those keep the
    pseudo-inverse.
    """
    if partial_mask:
        return jnp.linalg.pinv(H, hermitian=True) @ b
    return _linear_solve(H + jnp.eye(ndim) * 1e-9, b, ndim)


# ─────────────────────────────────────────────────────────────────────
# Newton–Raphson single-point solver  (vmap'd over query points)
# ─────────────────────────────────────────────────────────────────────

#: A step this small cannot move ``xi`` at float32 resolution (``xi`` lives in
#: ``[0, 1]``, where the spacing is ~1.2e-7), so the refinement has frozen: the
#: proposal equals the current point, the comparison rejects it, and the step
#: halves again forever.  Iterating past that point is a provable no-op, which
#: is what lets :func:`_make_nr_solver` leave the loop without changing the
#: answer.
_STEP_FLOOR = 1e-9

#: Default embedding tolerance, as a fraction of the mesh's extent.
#:
#: The refinement converges quadratically, so one Newton step takes a coarse
#: seed from ~1e-2 to float32 round-off in a single jump and there is nothing
#: in between: sweeping this constant on a 27-element tricubic hexahedral block
#: and on a bilinear patch, every value from 3e-7 to 1e-5 produces the *same*
#: answer in the *same* time, and the residual it leaves (~2e-7 of the mesh
#: extent, against ~9e-8 for running all 15 iterations) is float32 round-off
#: either way.  1e-6 sits in the middle of that plateau -- an order of
#: magnitude clear of the noise floor, so a borderline point still trips it,
#: and an order tighter than the tightest tolerance the test suite calls exact.
#:
#: Pass ``tol=0`` to :meth:`~HOMER.mesh.field.MeshField.embed_points` to
#: iterate unconditionally instead.
DEFAULT_EMBED_TOL = 1e-6


def _reference_length(mesh: MeshField) -> float:
    """A characteristic length for *mesh*, used to scale the default tolerance.

    Taken from the span of the nodal coordinates, so the tolerance means the
    same thing on a mesh measured in metres as on one measured in microns.
    Deliberately read once, at build time, from the stored geometry rather
    than from each call's ``fit_params``: it is a fixed reference length for
    interpreting a relative tolerance, not a measurement of the current
    configuration, and a traced one would make the stopping rule wobble as a
    fit moved the mesh.  Degenerate cases (a single node, all nodes
    coincident) fall back to 1.0.
    """
    locs = np.asarray([np.asarray(node.loc) for node in mesh.nodes], dtype=float)
    if locs.size == 0:
        return 1.0
    span = float(np.max(np.ptp(locs, axis=0))) if locs.shape[0] > 1 else 0.0
    if not np.isfinite(span) or span <= 0.0:
        return float(np.max(np.abs(locs))) or 1.0
    return span


def _make_nr_solver(mesh: MeshField, ndim: int, robust_init_est: bool):
    """Return a single-point NR function closed over *mesh*.

    The returned function has signature::

        nr_solve(elem, xi0, x_target, lbound, dim_mask,
                 iterations, tol_sq, fit_params) -> ((elem, xi), residual)

    ``iterations`` and ``tol_sq`` are traced values so the XLA trace is
    reused regardless of their Python-level values (B2).

    ``iterations`` is an upper bound rather than a fixed count: the loop also
    stops once the squared residual is within ``tol_sq`` or the line search
    has frozen (see :data:`_STEP_FLOOR`).  Because this runs under
    :func:`jax.vmap`, the whole batch iterates until its *last* point is
    finished -- so the saving is real when the points converge together, as
    they do whenever they lie on or near the mesh, and nil (but not negative)
    for a batch holding a point that never converges.

    The returned Jacobian is the one at the converged ``(elem, xi)``.  It is
    carried through the loop rather than recomputed: the iteration already
    evaluates it at every proposal it accepts, so the last accepted one is
    exactly the Jacobian at the point the loop settled on, and the custom JVP
    in :func:`build_embedding_fn` gets it for nothing.
    """
    A_size = ndim  # size of the linear system

    def nr_solve(elem, xi0, x_target, lbound, dim_mask, iterations, tol_sq,
                 fit_params):
        dim_mask = jnp.asarray(dim_mask, dtype=bool)

        # ── body of the refinement loop ──────────────────────────
        def body_fun(state):
            i, elem, xi, r, r_mag_sq, delta_xi, stepsize, jac = state
            xi_prop = xi + stepsize * delta_xi

            elem_prop, xi_mapped, valid = mesh.topomap(elem, xi_prop)

            # The Jacobian is taken at the *proposal*, not at the accepted
            # point.  It only ever feeds `new_delta_xi`, which is discarded
            # unless the proposal is accepted -- and when it is accepted the
            # two points are the same one.  So this is the same arithmetic as
            # evaluating it at (next_elem, next_xi), with the value and the
            # Jacobian now sharing one parameter gather and one
            # tensor-product contraction chain instead of four.
            x_prop, J = mesh.evaluate_embeddings_and_jacobians(
                elem_prop, xi_mapped, fit_params=fit_params)
            x_prop, J = x_prop[0], J[0]
            r_prop = jnp.where(dim_mask, x_target - x_prop, 0.0)
            r_mag_prop_sq = jnp.sum(jnp.square(r_prop))

            accept = r_mag_prop_sq < r_mag_sq

            next_elem = jnp.where(accept, elem_prop, elem)
            next_xi = jnp.where(accept, xi_mapped, xi)
            next_r = jnp.where(accept, r_prop, r)
            next_r_mag_sq = jnp.where(accept, r_mag_prop_sq, r_mag_sq)

            # The step is solved from the mask-zeroed Jacobian, but the one
            # carried out of the loop is the plain dw/dxi: it is the mesh's
            # Jacobian at a point, not a statement about this call's mask, and
            # the JVP rule applies the mask itself where it needs to.
            J_masked = jnp.where(dim_mask[:, None], J, 0.0)

            Jt = J_masked.T
            A = Jt @ J_masked + jnp.eye(A_size) * 1e-5

            mask = jnp.where(lbound, jnp.zeros_like(next_xi), jnp.ones_like(next_xi))
            diag_mask = jnp.where(lbound, jnp.ones_like(next_xi), jnp.zeros_like(next_xi))

            A_free = A * mask[:, None] * mask[None, :] + jnp.diag(diag_mask)
            b_free = (Jt @ r_prop) * mask

            new_delta_xi = _linear_solve(A_free, b_free, ndim)

            next_delta_xi = jnp.where(accept, new_delta_xi, delta_xi)
            next_stepsize = jnp.where(accept, 1.0, stepsize * 0.5)

            # `jac` tracks (next_elem, next_xi) for the same reason `next_r`
            # does: the accepted proposal is where the loop now stands, so
            # this is the Jacobian at the point that will be returned.
            next_jac = jnp.where(accept, J, jac)

            return (i + 1, next_elem, next_xi, next_r, next_r_mag_sq,
                    next_delta_xi, next_stepsize, next_jac)

        def cond_fun(state):
            i, _, _, _, r_mag_sq, _, stepsize, _ = state
            return ((i < iterations)
                    & (r_mag_sq > tol_sq)
                    & (stepsize > _STEP_FLOOR))

        # ── initial state ────────────────────────────────────────
        # `analytic_J` is what the loop carries and returns; `init_J` is only
        # what the first step is taken from, which `robust_init_est` swaps for
        # a numeric estimate.  Keeping them apart means the returned Jacobian
        # is the analytic one either way, including for a point that never
        # accepts a step.
        init_x, analytic_J = mesh.evaluate_embeddings_and_jacobians(
            elem, xi0, fit_params=fit_params)
        init_x, analytic_J = init_x[0], analytic_J[0]
        if robust_init_est:
            init_J = mesh.eval_numeric_jac_ele_xi_pair(elem, xi0, fit_params=fit_params, step=1e-2)
        else:
            init_J = analytic_J
        init_r = jnp.where(dim_mask, x_target - init_x, 0.0)
        init_r_mag_sq = jnp.sum(jnp.square(init_r))

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
        init_state = (jnp.int32(0), elem.astype(int), xi0, init_r,
                      init_r_mag_sq, init_delta_xi, 1.0, analytic_J)
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
        _, elem_f, xi_f, r_f, _, _, _, jac_f = final_state

        return (elem_f, xi_f), r_f, jac_f

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


def _coarse_nn_mode(partial_mask: bool, fdim: int) -> str:
    """Pick the coarse search for this call.  Static; see :func:`_coarse_nn`."""
    if partial_mask:
        return "masked"
    return "aknn" if fdim > MORTON_MAX_FDIM else "morton"


def _mask_is_partial(dim_mask) -> bool:
    """Whether *dim_mask*, as the caller gave it, switches anything off.

    Read from the caller's own argument, *before*
    :func:`_as_dim_mask` normalises it, and that ordering is the whole point.
    ``None`` is a Python-level statement that every dimension is active, and
    it stays one however the call is wrapped; the normalised form is a JAX
    array, which inside an enclosing ``jax.jit`` is a tracer whose value
    cannot be read at all.  Deciding from the normalised mask therefore made
    every jitted caller look masked, including the great majority who never
    passed a mask -- which silently swapped the cheap Z-curve coarse search
    for the exact one and, in the custom JVP, the direct solve for a
    pseudo-inverse.  ``jax.jit(f)`` and ``f`` did not agree on which
    algorithm to run.

    A mask that *is* supplied is documented as static, so its value is read
    here.  A caller who none the less traces over it gets the conservative
    answer, which is correct for either mask.
    """
    if dim_mask is None:
        return False
    try:
        return not bool(jnp.all(jnp.asarray(dim_mask).astype(bool)))
    except jax.errors.TracerBoolConversionError:
        return True


# ─────────────────────────────────────────────────────────────────────
# JVP helpers (for differentiable embedding)
# ─────────────────────────────────────────────────────────────────────

def _make_jvp_helpers(mesh: MeshField, approx_jac: bool, partial_mask: bool):
    """Build the per-point ``embed_single_jvp`` rule.

    The embedding minimises ``D = |m * (x - w(e, xi; p))|^2 / 2`` over ``xi``,
    so at the point the refinement settles on ``g = dD/dxi = -W' r`` is zero
    and the implicit function theorem gives the tangent::

        H xi_dot = -(dg/dx . x_dot + dg/dp . p_dot)

    with ``W = dw/dxi``, ``r = m * (x - w)`` and ``H = dg/dxi``.  Writing
    those three derivatives out, rather than handing ``g`` to
    :func:`jax.jacobian` and :func:`jax.jvp` as this used to, buys two
    things.

    ``H = W' M W - sum_d r_d d2w_d/dxi2``.  ``W`` is the Jacobian at the
    converged point, which :func:`_make_nr_solver` has already computed and
    now returns, so the first term is a matmul on a value that cost nothing.
    That leaves only the curvature term to differentiate, and contracting it
    with ``r`` *first* makes it the Hessian of the scalar ``r . w(xi)``
    instead of a ``(fdim, ndim, ndim)`` tensor -- taken over the element's
    own kernel, with its parameters already gathered, so nothing
    differentiates through the mesh-wide parameter scatter.  Measured on a
    27-element tricubic Hermite mesh at 20k points, ``H`` drops from 25.6 ms
    to 8.0 ms.

    And ``w`` is *linear* in ``p``, so the parameter derivatives are not
    derivatives at all::

        dw/dp . p_dot = w(e, xi; p_dot)
        dW/dp . p_dot = W(e, xi; p_dot)

    -- one more call to the same fused value-and-Jacobian kernel, evaluated
    at the tangent parameters.  This is why *p_dot* must be widened with
    zeros in the non-optimisable slots: it is being passed as though it were
    a parameter vector, and a fixed parameter contributes no motion.

    *partial_mask* says whether this trace's ``dim_mask`` switches anything
    off, and selects how ``xi_dot`` is solved for -- see
    :func:`_solve_xi_tangent`.  Like the coarse-search mode it is a
    compile-time property of the call, so the two solves never coexist in one
    trace.
    """
    ndim = mesh.ndim

    def embed_single_jvp(ele, xi, point, r, W, point_dot,
                         full_params, full_param_dot, dim_mask):
        eles = jnp.atleast_1d(ele)
        xis = jnp.atleast_2d(xi)

        # w_p = dw/dp . p_dot and W_p = dW/dp . p_dot, by linearity in p.
        w_p, W_p = mesh.evaluate_embeddings_and_jacobians(
            eles, xis, fit_params=full_param_dot)
        w_p, W_p = w_p[0], W_p[0]

        masked_W = jnp.where(dim_mask[:, None], W, 0.0)

        # sum_d r_d d2w_d/dxi2, as the Hessian of a scalar.  `r` is the
        # primal residual, already masked, so the mask needs no repeating.
        elem_params = mesh._element_params(eles, full_params)[0]
        curvature = jax.hessian(
            lambda x: jnp.dot(r, mesh.elem_evals(elem_params, x[None, :]))
        )(xi)
        local_H = W.T @ masked_W - curvature

        # dg/dx . x_dot + dg/dp . p_dot, from g = -W' r.
        comb_product = (W.T @ jnp.where(dim_mask, w_p - point_dot, 0.0)
                        - W_p.T @ r)

        active_mask = jnp.isclose(xi, 1.0, atol=1e-5) | jnp.isclose(xi, 0, atol=1e-5)
        free_mask = ~active_mask
        masked_H = (
            jnp.where(free_mask[:, None] * free_mask[None, :], local_H, 0.0)
            + jnp.diag(jnp.where(active_mask, 1.0, 0.0))
        )
        masked_comb_product = jnp.where(active_mask, 0.0, comb_product)

        xi_dot = -_solve_xi_tangent(masked_H, masked_comb_product, ndim,
                                    partial_mask) * (0 if approx_jac else 1)

        # w_dot = W xi_dot + dw/dp . p_dot, again with W reused.
        w_dot = W @ xi_dot + w_p

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

    :param mesh:
        The :class:`~HOMER.mesh.field.MeshField` instance.
    :param approx_jac:
        If ``True``, drops the sliding term from the residual gradient
        estimation for the custom JVP (see ``embed_points`` docs).
    :param robust_init_est:
        If ``True``, uses a numeric Jacobian for the initial NR step.
    """
    ndim = mesh.ndim
    reference_length = _reference_length(mesh)

    # Build the NR solver (closed over mesh)
    nr_solve = _make_nr_solver(mesh, ndim, robust_init_est)

    # Build the JVP helpers.  `partial_mask` changes how the xi tangent is
    # solved for, and is a compile-time property of the call, so there is one
    # rule per value of it rather than a branch inside the rule.
    _jvp_rules: dict[bool, object] = {}

    def _jvp_rule(partial_mask: bool):
        rule = _jvp_rules.get(partial_mask)
        if rule is None:
            rule = _make_jvp_helpers(mesh, approx_jac, partial_mask)
            _jvp_rules[partial_mask] = rule
        return rule

    # ── Coarse-search + NR (one traced block per branch) ─────────
    #
    # C1: For 2-D meshes the coarse search and NR refinement are
    # merged into a single function so XLA traces the full pipeline
    # (coarse eval → Morton NN → NR) as one fused block, avoiding
    # the materialisation boundary that used to separate them.

    def _run_coarse_2d(points, fit_params, dim_mask, grid_res, iterations,
                       tol_sq, window_size, nn_mode):
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

        (en, emb), res, jac = jax.vmap(
            lambda elem, xi, target, lbound, lmask: nr_solve(
                elem, xi, target, lbound, lmask, iterations, tol_sq, fit_params
            )
        )(elem_num, init_xi, points, mf_pt, dim_mask)
        return (en, emb), res, jac

    def _run_coarse_3d(points, fit_params, dim_mask, grid_res, iterations,
                       tol_sq, surface_embed, window_size, nn_mode):
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

        (en, emb), res, jac = jax.vmap(
            lambda elem, xi, target, lbound, lmask: nr_solve(
                elem, xi, target, lbound, lmask, iterations, tol_sq, fit_params
            )
        )(elem_num, init_xi, points, mf_pt, dim_mask)
        return (en, emb), res, jac

    # ── The core JIT function (takes only JAX arrays) ────────────
    #
    # Python-level branching (use_init_elexi, surface_embed, grid_res)
    # happens *outside* this function.  Only `iterations` is traced
    # inside the JIT (B2: different iteration counts share one trace).

    def _make_jit_embed(use_init_elexi: bool, surface_embed: bool,
                        grid_res: int, window_size: int, nn_mode: str,
                        partial_mask: bool):
        """Return a JIT-compiled ``@custom_jvp`` function for one branch."""
        embed_single_jvp = _jvp_rule(partial_mask)

        @jax.jit
        def _solve(points, fit_params, dim_mask,
                   init_elexi_elem, init_elexi_xi, iterations, tol_sq):
            """The embedding solve, returning the converged Jacobian too.

            Split out from ``_embed_jit`` so the custom JVP rule can have the
            Jacobian at the converged point, which the refinement has already
            computed.  A ``custom_jvp`` function has to hand back a tangent
            for everything it returns, and the Jacobian's own tangent would be
            another order of derivative nobody wants -- so the Jacobian is
            *not* part of the differentiable output.  It leaves through here
            instead, and ``_embed_jit`` drops it.
            """
            # dim_mask is normalised to a bool (n_pts, fdim) array by
            # `embed` before it ever reaches here (see _as_dim_mask).
            points = jnp.atleast_2d(points)

            # Widen the parameters to a full vector once per call.  Every
            # evaluator would otherwise scatter the optimisable entries back
            # into the mesh-wide parameter array on each invocation -- a
            # scatter over the whole mesh, repeated for every Newton
            # iteration of every point.  Downstream evaluators pass a
            # full-length vector straight through, so this is transparent.
            fit_params = mesh.expand_fit_params(fit_params)

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

                (en, emb), res, jac = jax.vmap(
                    lambda elem, xi, target, lbound, lmask: nr_solve(
                        elem, xi, target, lbound, lmask, iterations, tol_sq,
                        fit_params
                    )
                )(elem_num, init_xi, points, mf_pt, dim_mask)
                return (en, emb), res, jac
            else:
                if ndim == 2:
                    return _run_coarse_2d(points, fit_params, dim_mask,
                                         grid_res, iterations, tol_sq,
                                         window_size, nn_mode)
                else:
                    return _run_coarse_3d(points, fit_params, dim_mask,
                                         grid_res, iterations, tol_sq,
                                         surface_embed, window_size,
                                         nn_mode)

        @jax.custom_jvp
        @jax.jit
        def _embed_jit(points, fit_params, dim_mask,
                       init_elexi_elem, init_elexi_xi, iterations, tol_sq):
            (en, emb), res, _ = _solve(points, fit_params, dim_mask,
                                       init_elexi_elem, init_elexi_xi,
                                       iterations, tol_sq)
            return (en, emb), res

        @_embed_jit.defjvp
        def _embed_jvp(primal, tangent):
            (ele, xi), res, jac = _solve(*primal)
            primal_out = ((ele, xi), res)

            points, params, dm = primal[0], primal[1], primal[2]
            point_dot, param_dot = tangent[0], tangent[1]
            # tangent[2] is the dim_mask tangent.  The mask is a static
            # statement about *which residual components exist*, not a
            # differentiable quantity, so it carries no meaningful
            # tangent and is deliberately ignored.  Normalising it to
            # bool in `embed` makes JAX hand us a float0 zero here.

            points = jnp.atleast_2d(points)
            dm = _as_dim_mask(dm, points)

            # Widen both the parameters and their tangent to full vectors
            # once, here, rather than inside the per-point rule.  The
            # tangent is widened with *zeros* in the non-optimisable slots:
            # `expand_fit_params` fills those from the stored geometry, which
            # is the right constant for a value and exactly wrong for a
            # derivative -- a fixed parameter does not move.
            full_params = mesh.expand_fit_params(params)
            param_dot = jnp.asarray(param_dot)
            if param_dot.shape[-1] == full_params.shape[-1]:
                full_param_dot = param_dot
            else:
                full_param_dot = jnp.zeros_like(full_params).at[
                    mesh.optimisable_param_bool].set(param_dot)

            # dm is per-point, shape (n_pts, fdim): it must be mapped
            # over the point axis exactly like the primal path does,
            # otherwise every point sees the whole mask and r_dot comes
            # back with a spurious leading (n_pts,) axis.
            xi_dot, r_dot = jax.vmap(
                lambda e, x, w, r, W, w_dot, m: embed_single_jvp(
                    e, x, w, r, W, w_dot, full_params, full_param_dot, m
                )
            )(ele, xi, points, res, jac, point_dot, dm)

            tangent_out = (
                (jnp.zeros_like(ele, dtype=jax.float0), xi_dot),
                r_dot,
            )
            return primal_out, tangent_out

        return _embed_jit

    # ── Pre-compile the default branch (coarse search, no surface) ──
    # Additional branches are compiled on demand and cached.
    _jit_cache: dict[tuple[bool, bool, int, int, str, bool], object] = {}

    def _get_jit(use_init: bool, surface: bool, gres: int, wsize: int,
                 nn_mode: str, partial_mask: bool):
        key = (use_init, surface, gres, wsize, nn_mode, partial_mask)
        fn = _jit_cache.get(key)
        if fn is None:
            fn = _make_jit_embed(use_init, surface, gres, wsize, nn_mode,
                                 partial_mask)
            _jit_cache[key] = fn
        return fn

    # ── Public entry point ───────────────────────────────────────
    def embed(points, fit_params, dim_mask,
              init_elexi, surface_embed, grid_res, iterations,
              chunk_size=None, window_size=16, tol=None):
        """Dispatch to the appropriate JIT-compiled embedding function.

        Parameters that control Python-level branching
        (``init_elexi``, ``surface_embed``, ``grid_res``,
        ``window_size``) select the cached JIT trace.  ``iterations``
        and ``tol`` are passed as traced values so they share one trace
        across all values (B2).

        ``tol`` is the residual norm at which the refinement stops;
        ``None`` uses :data:`DEFAULT_EMBED_TOL` scaled by the mesh's own
        extent, and ``0`` runs the full ``iterations`` unconditionally.

        When *chunk_size* is set and the number of query points exceeds
        it, the point set is split into chunks that are processed
        sequentially and concatenated.  This bounds peak memory to
        ``O(chunk_size)`` instead of ``O(n_pts)``, preventing swap
        usage on large inputs.
        """
        points = jnp.atleast_2d(points)
        n_pts = points.shape[0]

        # Traced, like `iterations`, so changing it shares one XLA trace.
        if tol is None:
            tol = DEFAULT_EMBED_TOL * reference_length
        tol_sq = jnp.float32(tol) ** 2

        # Whether anything is masked off, read from the caller's argument
        # before it is normalised -- see :func:`_mask_is_partial`.  Both the
        # coarse search and the JVP's xi solve branch on it, so it has to be
        # settled while `None` is still `None`.
        partial_mask = _mask_is_partial(dim_mask)

        # Resolve the mask once, up front, so every downstream path
        # (chunked or not, JIT primal or custom JVP) sees the same
        # bool (n_pts, fdim) array and never has to re-handle None.
        dim_mask = _as_dim_mask(dim_mask, points)

        use_init = init_elexi is not None

        # Only the coarse search cares -- a caller-supplied seed skips it.
        nn_mode = ("morton" if use_init
                   else _coarse_nn_mode(partial_mask, points.shape[-1]))

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
                              int(grid_res), int(window_size), nn_mode,
                              partial_mask)
                (en, emb), res = fn(p_chunk, fit_params, dm_chunk,
                                    ie_elem, ie_xi, jnp.int32(iterations),
                                    tol_sq)
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
                      int(grid_res), int(window_size), nn_mode, partial_mask)
        return fn(points, fit_params, dim_mask,
                  ie_elem, ie_xi, jnp.int32(iterations), tol_sq)

    return embed
