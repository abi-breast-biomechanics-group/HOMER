"""
jacobian_evaluator.py – Sparse-Jacobian helper for JAX cost functions.

Provides :func:`jacobian`, which wraps an arbitrary JAX-compatible cost
function and returns:

1. A JIT-compiled version of the cost function.
2. A sparse Jacobian function based on :mod:`sparsejac` (forward-mode AD
   with sparsity exploitation), or a dense Jacobian for small problems.

Used by :func:`~HOMER.fitting.point_cloud_fit` and can be called directly
when building custom fitting problems::

    from HOMER.jacobian_evaluator import jacobian

    fitting_fn, jac_fn = jacobian(my_cost_function,
                                  init_estimate=mesh.optimisable_param_array)

When the Jacobian is too large to hold -- or dense in the blocks that matter,
so that a colouring saves nothing -- :func:`matrix_free_jacobian` gives scipy a
``LinearOperator`` over ``jvp``/``vjp`` and a column-equilibrating scaling to
go with it, for ``least_squares(..., tr_solver='lsmr')``.
"""

from functools import partial
from time import time
from typing import Callable, Optional
import jax
import jax.numpy as jnp
import sparsejac
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from HOMER.mesh import Mesh, MeshField
from HOMER.mesh.topology import _colour_ranks

def jacobian(
    cost_function: Optional[Callable] = None, 
    init_estimate: Optional[jax.typing.ArrayLike] = None, 
    sparsity: Optional[jax.typing.ArrayLike | Callable] =None, 
    further_args=None, 
    sparse: bool = True, 
    return_sparsity=False
):
    """
    Given a jax compatible callable, returns both a compiled jax function, but also 
    the autodifferentiated jacobian of the function.

    :param cost_function:
        The JAX-compatible residual function, differentiated with respect to
        its first argument.  ``None`` returns a partially-applied
        :func:`jacobian`, so it can be used as a decorator factory.
    :param init_estimate:
        A representative parameter vector, used to probe the sparsity
        pattern.  Required unless *sparsity* is given.
    :param sparsity:
        A known pattern as a ``BCOO``, which skips the probing step.  A
        callable raises: patterns that change between calls are not
        supported.
    :param further_args:
        Extra keyword arguments bound into *cost_function* before the
        sparsity is estimated.
    :param sparse:
        When ``True`` (default), build the Jacobian with :mod:`sparsejac`
        forward-mode AD over the pattern.  ``False`` gives a dense
        ``jax.jacfwd``, which is the better choice for small problems.
    :param return_sparsity:
        Also return the pattern that was used or estimated.

    :raises ValueError:
        If neither *init_estimate* nor *sparsity* is given, or if *sparsity*
        is a callable.

    :returns:
        ``(fwd_func, jac_func)``, both taking the parameter vector and
        suitable for ``scipy.optimize.least_squares``; the Jacobian returns a
        ``scipy.sparse.coo_array``.  With *return_sparsity*, the pattern is
        appended.
    """
    if init_estimate is None and sparsity is None:
        raise ValueError("Code needs an initial estimate for meaningful sparsity estimation")

    if cost_function is None:
        return partial(
            jacobian, 
            init_estimate=init_estimate, 
            sparsity=sparsity, 
            sparse=sparse, 
            return_sparsity=return_sparsity
        )

    if further_args is None:
        further_args = {}
    
    fwd_func = jax.jit(cost_function)

    if sparse:
        if isinstance(sparsity, Callable):
            raise ValueError("Non-static sparsities are not yet supported")

        else:
            if sparsity is None:
                sparsity = estimate_sparsity(partial(fwd_func, **further_args), init_estimate)
            
            @jax.jit
            def sparse_jacobian(params, **kwargs):
                with jax.ensure_compile_time_eval():
                    jacfwd = sparsejac.jacfwd(cost_function, sparsity=sparsity, argnums=0)
                return jacfwd(params, **kwargs)

            def scipy_sparse_jac(params, **kwargs):
                jax_sparse = sparse_jacobian(params, **kwargs)
                return scipy.sparse.coo_array(
                    (jax_sparse.data, (jax_sparse.indices[:, 0], jax_sparse.indices[:, 1])),
                    shape=jax_sparse.shape,
                )
    else:
        dense_jac_fwd = jax.jit(jax.jacfwd(cost_function, argnums=0))
        
        def scipy_sparse_jac(params, **kwargs):
            return np.asarray(dense_jac_fwd(params, **kwargs))

    if return_sparsity:
        return fwd_func, scipy_sparse_jac, sparsity
    return fwd_func, scipy_sparse_jac
    
def estimate_sparsity(callable_fn, init_estimate) -> jax.experimental.sparse.BCOO:
    """Probe which outputs of *callable_fn* depend on which inputs.

    Perturbs one input at a time by 1.0 and records the outputs that move by
    more than ``1e-8``.  The probe runs under :func:`jax.lax.scan` rather than
    as one batched call, so peak memory is ``O(N)`` rather than ``O(N**2)`` in
    the parameter count.

    This detects structural dependence, not the size of the derivative: an
    output that happens not to move under a unit step at *init_estimate* is
    recorded as independent, so pass an estimate representative of where the
    optimiser will actually work.

    :param callable_fn:
        A JAX-compatible function of one parameter vector, with any other
        arguments already bound.
    :param init_estimate:
        The parameter vector to perturb, shape ``(N,)``.

    :returns:
        The pattern as a ``BCOO`` of shape ``(M, N)`` with unit entries, in
        the form :mod:`sparsejac` expects.
    """
    init_estimate = jnp.asarray(init_estimate)
    init_val = callable_fn(init_estimate)
    
    N = init_estimate.shape[0]
    M = init_val.shape[0]

    @jax.jit
    def compute_diffs_scanned():
        def scan_body(_, i):
            # Create a single perturbation vector on the fly to save memory
            p = jnp.zeros(N).at[i].set(1.0)
            res = callable_fn(init_estimate + p)
            return None, res
        
        # Scan executes sequentially, keeping memory footprint to O(N) instead of O(N^2)
        _, new_vals = jax.lax.scan(scan_body, None, jnp.arange(N))
        return new_vals
    
    new_vals = compute_diffs_scanned()
    changed_mask = jnp.abs(new_vals - init_val) > 1e-8 
    input_idx, output_idx = jnp.where(changed_mask)
    inds = jnp.column_stack((output_idx, input_idx))
    
    return jax.experimental.sparse.BCOO(
        (jnp.ones(inds.shape[0]), inds), 
        shape=(M, N)
    )



def estimate_column_norms(vjp, n_res, probes=16, seed=0, dtype=np.float64):
    """The column norms of a jacobian, without ever forming a column.

    ``||J_j||^2 = E[(J^T z)_j^2]`` for ``z ~ N(0, I)``, so a single vjp prices
    every column at once and a handful of probes estimates the whole scaling.
    This is the matrix-free twin of
    :func:`~HOMER.mesh.parameters.column_equilibrated_lstsq`, which reads the
    same norms straight off an assembled matrix.

    :param vjp:
        ``w -> J^T w`` at the point the jacobian is wanted.
    :param n_res:
        Length of the residual vector, which is the length of a probe.
    :param probes:
        How many gaussian probes to average.  The estimate's relative error
        falls as ``1/sqrt(probes)``, and a scaling only has to be right to
        within a factor, so 16 is plenty.
    :param seed:
        Seed for the probes, so that a fit is reproducible.
    :param dtype:
        Dtype of the probes; must match what the residual is evaluated in.
    :returns:
        ``(n_par,)`` array of estimated column norms.
    """
    rng = np.random.default_rng(seed)
    probe = rng.normal(size=(probes, n_res)).astype(dtype)
    mean_square = np.mean([np.asarray(vjp(jnp.asarray(z))) ** 2 for z in probe],
                          axis=0)
    return np.sqrt(mean_square)


def matrix_free_jacobian(cost_function, init_estimate, probes=16, seed=0,
                         precondition=True):
    """A jacobian for ``least_squares`` that is never formed, only applied.

    For a residual of ~1e5 entries in ~1e4 parameters the jacobian costs
    gigabytes to store, and a sparsity colouring does not rescue it when the
    blocks that matter are dense -- every dof supporting a sample excites the
    same rows, so :func:`jacobian` is the wrong tool.  This returns a
    :class:`scipy.sparse.linalg.LinearOperator` over jax's ``jvp``/``vjp``
    instead: nothing larger than a residual vector is ever allocated, and
    ``least_squares(..., tr_solver='lsmr')`` -- the one scipy trust-region
    solver that takes an operator -- solves each subproblem from
    matrix-vector products alone.

    Everything is returned in the *scaled* variable ``u``, with
    ``params = u * scale`` and *scale* the reciprocal column norms, so that
    every column of the operator has unit length.  That preconditioning is not
    a refinement: a Hermite basis carries derivative dofs whose columns are
    orders of magnitude shorter than the value dofs', and lsmr on the raw
    operator never solves the subproblem well enough to take a Newton-sized
    step.  SciPy's own ``x_scale='jac'`` cannot do it here -- it squares the
    jacobian elementwise, which needs a real matrix.

    *scale* is estimated once, at *init_estimate*, and held fixed for the whole
    solve, exactly as the assembled equilibration is.

    :param cost_function:
        A JAX-compatible ``params -> residuals``, both 1-D.
    :param init_estimate:
        The parameters the fit starts from, and the point the column norms are
        estimated at.
    :param probes:
        Gaussian probes averaged for those norms.
    :param seed:
        Seed for the probes.
    :param precondition:
        ``False`` returns the bare operator and a *scale* of ones.  Useful for
        pricing what the scaling buys, not for fitting.
    :returns:
        fitting_function : Callable
            ``u -> residuals`` as numpy, for ``least_squares``.
        jacobian_operator : Callable
            ``u -> LinearOperator``, for its ``jac``.
        scale : np.ndarray
            ``(n_par,)``.  Divide the starting parameters by it, multiply the
            answer back.

    **Examples**

    ::

        fwd, jac, scale = matrix_free_jacobian(cost, p_start)
        result = least_squares(fwd, p_start / scale, jac=jac, tr_solver='lsmr')
        params = result.x * scale
    """
    init = jnp.asarray(init_estimate)
    start_residual = jax.jit(cost_function)(init)
    n_res, n_par = start_residual.shape[0], init.shape[0]
    dtype = np.dtype(start_residual.dtype)

    jvp = jax.jit(lambda p, v: jax.jvp(cost_function, (p,), (v,))[1])
    vjp = jax.jit(lambda p, w: jax.vjp(cost_function, p)[1](w)[0])

    if precondition:
        norms = estimate_column_norms(partial(vjp, init), n_res, probes, seed, dtype)
        #a dead column would come back as 1/0: leave it at unit scale instead
        scale = 1.0 / np.where(norms > 0, norms, 1.0)
    else:
        scale = np.ones(n_par)
    scale = scale.astype(dtype)
    scale_j = jnp.asarray(scale)

    scaled_cost = jax.jit(lambda u: cost_function(u * scale_j))

    def as_vector(v):
        #scipy probes the operator with an int8 vector and hands lsmr (n, 1)
        #columns; jax rejects both, so normalise on the way in
        return jnp.asarray(np.asarray(v, dtype=dtype).ravel())

    def fitting_function(params):
        return np.asarray(scaled_cost(as_vector(params)))

    def jacobian_operator(params):
        p = as_vector(params) * scale_j
        return LinearOperator(
            (n_res, n_par), dtype=dtype,
            matvec=lambda v: np.asarray(jvp(p, as_vector(v) * scale_j)),
            rmatvec=lambda w: np.asarray(vjp(p, as_vector(w)) * scale_j))

    return fitting_function, jacobian_operator, scale


def _colouring_seeds(mesh: Mesh | MeshField, fields_seperable: bool):
    """The mesh colouring as two tangent bases, shape ``(n_colours, n_par)``.

    Row *c* of the first carries a 1 in every parameter coloured *c*, and the
    second carries that parameter's rank within the colour.  A jvp against the
    pair returns each nonzero of the Jacobian alongside the same nonzero
    weighted by that rank, which is what :func:`_decode_compressed_columns`
    divides apart before reading it back through *members*.

    :returns:
        ``(seed_values, seed_ranks, members)``, the last being the
        ``(n_colours, widest colour)`` table turning a decoded rank back into
        a parameter.
    """
    colouring, seed_values, seed_ranks = mesh.get_colouring_dict(
        fields_seperable=fields_seperable, seed_matrix=True)

    colour_of = np.empty(len(colouring), dtype=int)
    colour_of[list(colouring)] = list(colouring.values())
    _, members = _colour_ranks(colour_of, int(colour_of.max()) + 1)

    return seed_values.todense().T, seed_ranks.todense().T, jnp.asarray(members)


@jax.jit
def _decode_compressed_columns(weighted, values, members, eps=1e-6, int_tol=1e-2):
    """Recover which column each compressed entry came from.

    ``weighted / values`` is the rank the parameter holds within its colour,
    because the colouring lets at most one parameter of a colour reach a given
    residual.  Row *c* came from colour *c*, so that rank and *members* give
    the column back.  Ranks run to the size of one colour rather than to the
    parameter count, and it is that smaller range which keeps the quotient
    sharp enough to round in float32.

    The quotient only means anything where the entry is a real nonzero, so the
    division is guarded and everything doubtful -- a value under *eps*, a
    quotient further than *int_tol* from an integer, a rank past the end of
    its colour -- is reported as invalid rather than scattered somewhere wrong.

    :returns:
        ``(columns, valid)``, both shaped like *values*.  Columns where *valid*
        is ``False`` are set to 0 and must not be read.
    """
    is_nonzero = jnp.abs(values) > eps
    quotient = weighted / jnp.where(is_nonzero, values, 1.0)
    ranks = jnp.round(quotient)

    in_range = (ranks >= 0) & (ranks < members.shape[1]) & jnp.isfinite(quotient)
    columns = jnp.take_along_axis(
        members, jnp.where(in_range, ranks, 0).astype(jnp.int32), axis=1)

    valid = (is_nonzero
             & (jnp.abs(quotient - ranks) < int_tol)
             & in_range
             & (columns >= 0))
    return jnp.where(valid, columns, 0).astype(jnp.int32), valid


def make_jac_for_mesh_func(initial_mesh: Mesh | MeshField, function: Callable,
                           fields_seperable: bool) -> Callable:
    """A sparse Jacobian that re-reads its own sparsity on every call.

    A mesh already knows which parameters can never touch the same output --
    that is its element map -- so the pattern needs no probing:
    :meth:`~HOMER.mesh.Mesh.get_colouring_dict` turns the topology into a
    colouring, and one jvp per colour then carries every nonzero of the
    Jacobian.  A second jvp, against the same seeds weighted by parameter
    index, says which column each of those nonzeros belongs in.

    Paying for that second pass buys a Jacobian whose *pattern* may move
    between calls, which is what a data-to-model term needs: a residual built
    on :meth:`~HOMER.mesh.Mesh.embed_points` re-embeds its data as the mesh
    deforms, so a point can land in a different element from one iteration to
    the next.  The colouring stays valid throughout -- every residual entry
    still draws on a single element's parameters -- while the decoded indices
    follow the data.  When the pattern does hold still, take
    :func:`make_static_jac_for_mesh_func` instead and halve the work.

    :param initial_mesh:
        The mesh whose colouring is used.  Only its topology and its
        optimisable-parameter mask are read, so the mesh may deform afterwards;
        fixing or freeing parameters invalidates it.
    :param function:
        A JAX-compatible ``(params, **kwargs) -> residuals``, both 1-D, taking
        the mesh's :attr:`optimisable_param_array` and differentiated with
        respect to it alone.
    :param fields_seperable:
        Treat each field component as its own output, which holds for
        embedding evaluation but not for a coupled quantity such as a local
        Jacobian determinant.  ``True`` needs markedly fewer colours, and so
        fewer jvps.

    :returns:
        ``(params, **kwargs) -> coo_array`` of shape
        ``(n_residuals, n_parameters)``, ready for
        ``scipy.optimize.least_squares``.  Keyword arguments are forwarded to
        *function* behind its parameter vector, which is how
        ``least_squares(..., kwargs=...)`` threads per-solve data through.

    .. warning::
        *fields_seperable* has to match how the residual actually couples, and
        a mismatch is quiet: entries the decode cannot resolve are dropped, so
        the Jacobian comes back sparse, plausible and wrong rather than raising.
        A residual differentiated through :meth:`~HOMER.mesh.Mesh.embed_points`
        is separable only with ``approx_jac=True``, which drops the sliding
        term and leaves the point at fixed ``(element, xi)``.  The exact
        embedding jvp moves the point as the mesh deforms, which makes every
        residual component depend on every component of the element's
        parameters -- separable in elements, not in fields, so it needs
        ``fields_seperable=False``.
    """
    seed_values, seed_ranks, members = _colouring_seeds(initial_mesh, fields_seperable)
    n_par = seed_values.shape[1]

    @jax.jit
    def compressed(params, **kwargs):
        jvp = jax.vmap(jax.linearize(lambda p: function(p, **kwargs), params)[1])
        values = jvp(seed_values)
        return (values,) + _decode_compressed_columns(jvp(seed_ranks), values, members)

    def dynamic_jac(params, **kwargs):
        values, columns, valid = compressed(params, **kwargs)
        n_res = values.shape[1]

        kept = np.flatnonzero(np.asarray(valid).ravel())
        rows = np.broadcast_to(np.arange(n_res)[None, :], values.shape).ravel()[kept]
        return scipy.sparse.coo_array(
            (np.asarray(values).ravel()[kept], (rows, np.asarray(columns).ravel()[kept])),
            shape=(n_res, n_par))

    return dynamic_jac


def make_static_jac_for_mesh_func(initial_mesh: Mesh | MeshField, function: Callable,
                                  init_estimate: jax.typing.ArrayLike,
                                  fields_seperable: bool,
                                  further_args: Optional[dict] = None) -> Callable:
    """The same coloured Jacobian, with the sparsity decoded once and frozen.

    :func:`make_jac_for_mesh_func` spends two jvps per colour on every call,
    one for the values and one to find out where they go.  When the pattern
    does not move -- anything evaluated at fixed ``(element, xi)`` pairs, which
    is most model-to-data fits -- the second pass returns the same answer every
    time.  This runs it once at *init_estimate*, keeps the indices, and leaves
    one jvp per colour to do per call, so it costs about half as much.

    :param initial_mesh:
        The mesh whose colouring is used.
    :param function:
        A JAX-compatible ``(params, **kwargs) -> residuals``, both 1-D, taking
        the mesh's :attr:`optimisable_param_array` and differentiated with
        respect to it alone.
    :param init_estimate:
        Where the pattern is read.  An entry that happens to vanish here is
        recorded as absent and stays absent, so pass parameters representative
        of where the optimiser will work -- the same care
        :func:`estimate_sparsity` needs, for the same reason.
    :param fields_seperable:
        As in :func:`make_jac_for_mesh_func`.
    :param further_args:
        Keyword arguments for *function* while the pattern is read.  They need
        not be the ones later calls pass, and often should not be: an entry the
        probe data masks to zero is recorded as absent for good, so probe with
        data that leaves every entry live -- unit weights rather than a
        visibility mask, say -- and pass the real data per call.

    :returns:
        ``(params, **kwargs) -> coo_array`` of shape
        ``(n_residuals, n_parameters)``, carrying the same indices every call,
        so only the values come back from the device.  Keyword arguments are
        forwarded as in :func:`make_jac_for_mesh_func`.

    .. warning::
        Frozen indices are wrong indices once the pattern moves.  A residual
        that re-embeds its data will keep returning a full-looking Jacobian
        with its entries in the columns they belonged to at *init_estimate*;
        that case wants :func:`make_jac_for_mesh_func`.  The restriction in its
        warning applies here too.
    """
    seed_values, seed_ranks, members = _colouring_seeds(initial_mesh, fields_seperable)
    n_par = seed_values.shape[1]

    def coloured_jvps(params, seeds, **kwargs):
        return jax.vmap(jax.linearize(lambda p: function(p, **kwargs), params)[1])(seeds)

    probe = {} if further_args is None else further_args
    init_estimate = jnp.asarray(init_estimate)
    start_values = coloured_jvps(init_estimate, seed_values, **probe)
    columns, valid = _decode_compressed_columns(
        coloured_jvps(init_estimate, seed_ranks, **probe), start_values, members)

    n_res = start_values.shape[1]
    kept = np.flatnonzero(np.asarray(valid).ravel())
    rows = np.broadcast_to(np.arange(n_res)[None, :], start_values.shape).ravel()[kept]
    cols = np.asarray(columns).ravel()[kept]
    kept = jnp.asarray(kept)

    @jax.jit
    def entries(params, **kwargs):
        return coloured_jvps(params, seed_values, **kwargs).ravel()[kept]

    def static_jac(params, **kwargs):
        #the pattern is frozen, so only the values cross back from the device
        return scipy.sparse.coo_array((np.asarray(entries(params, **kwargs)), (rows, cols)),
                                      shape=(n_res, n_par))

    return static_jac
