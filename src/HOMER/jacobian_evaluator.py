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
from jax.flatten_util import ravel_pytree

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


def _mesh_colours(mesh: Mesh | MeshField, fields_seperable: bool) -> np.ndarray:
    """The colour of each optimisable parameter of *mesh*, from its topology."""
    colouring = mesh.get_colouring_dict(fields_seperable=fields_seperable)
    colour_of = np.empty(len(colouring), dtype=int)
    colour_of[list(colouring)] = list(colouring.values())
    return colour_of


def _colouring_seeds(colour_of):
    """The colouring *colour_of* as two tangent bases, shape ``(n_colours, n_par)``.

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
    colour_of = np.asarray(colour_of, dtype=int)
    n_colours = int(colour_of.max()) + 1
    rank, members = _colour_ranks(colour_of, n_colours)

    seed_values = np.zeros((n_colours, colour_of.size), dtype=np.float32)
    seed_ranks = np.zeros((n_colours, colour_of.size), dtype=np.float32)
    seed_values[colour_of, np.arange(colour_of.size)] = 1.0
    seed_ranks[colour_of, np.arange(colour_of.size)] = rank
    return jnp.asarray(seed_values), jnp.asarray(seed_ranks), jnp.asarray(members)


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


#: How closely the cached Jacobian has to reproduce the probe jvp, relative to
#: the probe's own size, before it is trusted.  A moved pattern misses by the
#: size of the entries it misplaces; float32 round-off by five orders less.
_PROBE_TOL = 1e-3


def _along(values, columns, valid, direction):
    """``J @ direction`` for a Jacobian held compressed: ``values[..., c, r]`` at
    column ``columns[..., c, r]`` where ``valid``."""
    return jnp.sum(jnp.where(valid, values * direction[columns], 0.0), axis=-2)


def _agrees(values, columns, valid, direction, expected):
    """Whether the compressed Jacobian reproduces *expected*, the jvp along *direction*."""
    miss = jnp.abs(_along(values, columns, valid, direction) - expected).max()
    return bool(miss <= _PROBE_TOL * (jnp.abs(expected).max() + 1e-12))


@jax.jit
def _merge_patterns(columns, valid, held_columns, held_valid):
    """A fresh decode folded into the pattern held, and whether either moved.

    An entry is held at the column it was decoded at.  A value can fall under
    the decode's threshold on one call and rise over it on the next while its
    column stays put, so the held pattern is the union of the decodes:

    :returns:
        ``(columns, valid, conflict, grown)``.  *conflict* is an entry decoded
        at a column other than the one held -- the pattern has moved, and the
        union means nothing; *grown* an entry the held pattern lacked.
    """
    both = valid & held_valid
    conflict = jnp.any(both & (columns != held_columns))
    grown = jnp.any(valid & ~held_valid)
    return jnp.where(valid, columns, held_columns), valid | held_valid, conflict, grown


def _bucket(n):
    """*n* rounded up to within a sixteenth, so a layout of similar size reuses its compile."""
    step = 1 << max(n.bit_length() - 4, 0)
    return -(-n // step) * step


@partial(jax.jit, static_argnames='size')
def _csr_layout(valid, columns, size):
    """The kept entries of a ``(n_items, n_jvps, n_res)`` block in CSR order.

    Item *b*'s residual *r* is row ``b * n_res + r``.

    :returns:
        ``(order, indices, indptr)``: each entry's position in the block
        ravelled, its column, and the row pointer.  *order* and *indices* are
        padded to *size*; ``indptr[-1]`` entries are real.
    """
    n_items, n_jvps, n_res = valid.shape
    row_major = jnp.nonzero(jnp.swapaxes(valid, 1, 2).ravel(), size=size, fill_value=0)[0]
    row, jvp = row_major // n_jvps, row_major % n_jvps
    order = (row // n_res * n_jvps + jvp) * n_res + row % n_res
    indptr = jnp.concatenate([jnp.zeros(1, jnp.int32),
                              jnp.cumsum(valid.sum(1).ravel(), dtype=jnp.int32)])
    return order, columns.ravel()[order], indptr


def _is_leaf(x):
    return x is None or isinstance(x, (MeshField, np.ndarray, jax.Array))


def make_jac_for_mesh_func(initial_mesh, function: Callable,
                           fields_seperable: Optional[bool] = None,
                           batched: bool = False, init_estimate=None) -> tuple[Callable, Callable]:
    """A sparse Jacobian of a function of mesh parameters, from the meshes' colourings.

    A mesh already knows which parameters can never touch the same output --
    that is its element map -- so the pattern needs no probing:
    :meth:`~HOMER.mesh.Mesh.get_colouring_dict` turns the topology into a
    colouring, and one jvp per colour then carries every nonzero of the
    Jacobian.  Which column each nonzero belongs to is decoded from a second
    set of jvps, against the same seeds weighted by each parameter's rank in
    its colour.

    That decoding is paid only while the pattern moves.  Every call also takes
    one jvp along a fixed random direction and checks the Jacobian it holds
    against it: a pattern that has moved -- a residual on
    :meth:`~HOMER.mesh.Mesh.embed_points` whose data lands in a new element,
    an entry masked to zero when it was decoded -- misses that jvp by the size
    of what it misplaced, and is decoded afresh.  A pattern is decoded
    alongside the values on every call until a decode adds nothing to the
    entries already held -- values that cross the decode's threshold come and
    go while their columns stay put -- and from then on trusted while the
    probe agrees.  So a fixed
    pattern costs one set of jvps a call, and a moving one stays right.

    The parameters may be a pytree -- a mesh's and a statistical shape model's
    mode weights, several meshes, a pose -- with *initial_mesh* the same tree
    holding a mesh at each leaf to colour and ``None`` at each leaf small
    enough to take one jvp per entry.  Each leaf is a block of columns, in the
    order :func:`jax.flatten_util.ravel_pytree` lays them out.  The residual
    and Jacobian returned take that ravelled vector -- what a solver works on
    -- and unravel it for *function*, so ``x0, unravel = ravel_pytree(params)``
    gives the starting vector and reads the solution back.

    A leaf whose parameters are not a mesh's own -- a reparameterisation, say
    -- takes its colouring as an integer array instead, the colour of each of
    its parameters, flattened.  Any two parameters sharing a colour must never
    reach the same residual entry; the probe checks it as it checks a mesh's.
    A 2-D array holds candidate colourings, one per row, tried in order as a
    mesh's separable and coupled colourings are.

    The residual may be a pytree too: each of its leaves is a block of rows,
    differentiated on its own -- its own colouring tried, its own pattern
    kept -- so a term that couples a mesh's field components does not force
    the coupled colouring on one that does not, and a block whose data moves
    between elements does not unsettle one whose pattern is fixed.  Rows
    follow :func:`jax.flatten_util.ravel_pytree` of the output -- the order
    of the residual returned beside the Jacobian.  Work the blocks share is
    done once per block.

    With *batched*, *function* describes one item of a batch -- one subject of
    a cohort fitted on a shared template -- and every leaf of the parameters,
    and of the keyword arguments, carries a leading batch axis.  The items are
    differentiated side by side with the same tangents, so whatever does not
    depend on an item's own values is done once for the batch, and the
    Jacobian comes back block diagonal within each block of rows: rows follow
    ``ravel_pytree`` of the batched output, a block's rows item by item.

    :param initial_mesh:
        The mesh whose colouring is used, or a tree matching the parameters
        holding a mesh, a colour array (or a 2-D array of candidates), or
        ``None`` at each leaf.  With
        *batched*, the tree describes one item.  Only the topology and
        optimisable-parameter mask of a mesh are read, so the meshes may deform
        afterwards; fixing or freeing parameters invalidates them.
    :param function:
        A JAX-compatible ``(params, **kwargs) -> residuals``, the residuals an
        array or a pytree of them, differentiated with respect to *params*
        alone.  For a mesh leaf, every residual entry must draw on one
        element's parameters at a time.
    :param fields_seperable:
        Whether each residual component draws on only the matching field
        component of a mesh, which needs a third as many colours.  ``None``
        finds out, mesh by mesh and block by block: the separable colouring is
        tried first and kept if the probe confirms it.  An evaluation at fixed points is
        separable; a residual on :meth:`~HOMER.mesh.Mesh.embed_points` is not,
        unless its points are held with ``derivative='fixed'``.
    :param batched:
        Whether *function* is applied item by item over a leading batch axis.
    :param init_estimate:
        Parameters with the structure *function* takes -- a tree, a batch --
        from which the ravelled vector is unravelled.  Needed unless the
        parameters are one vector: a single, unbatched mesh or colour-array
        leaf.

    :returns:
        ``(residual, jacobian)``, both ``(x, **kwargs)`` for *x* the ravelled
        parameters: the compiled residual, ravelled to 1-D, and its Jacobian as a ``csr_array`` of
        shape ``(n_residuals, n_parameters)`` with the rows in the same order
        -- the pair ``scipy.optimize.least_squares`` takes.  Keyword arguments
        are forwarded to *function* behind its parameters, which is how
        ``least_squares(..., kwargs=...)`` threads per-solve data through.

    :raises ValueError:
        If the parameters are a tree or a batch and *init_estimate* is not
        given, or if no colouring of a leaf reproduces the probe for some block of rows:
        a residual entry there draws on two parameters of one colour, which a
        colouring cannot carry.
    """
    meshes, treedef = jax.tree_util.tree_flatten(initial_mesh, is_leaf=_is_leaf)
    modes = [True, False] if fields_seperable is None else [bool(fields_seperable)]
    if init_estimate is not None:
        unravel = ravel_pytree(init_estimate)[1]
    elif len(meshes) == 1 and meshes[0] is not None and not batched:
        unravel = lambda x: treedef.unflatten([x])
    else:
        raise ValueError("the parameters are a tree or a batch, so init_estimate must give "
                         "their structure for the ravelled vector to be unravelled")

    def leaf_seeds(leaf):
        """Each colouring the leaf may take, in the order they are tried."""
        if isinstance(leaf, MeshField):
            return {mode: _colouring_seeds(_mesh_colours(leaf, mode)) for mode in modes}
        candidates = np.atleast_2d(np.asarray(leaf))
        return {k: _colouring_seeds(colours) for k, colours in enumerate(candidates)}

    seeds = [None if m is None else leaf_seeds(m) for m in meshes]
    rng = np.random.default_rng(0)
    probes = [None if s is None else jnp.asarray(rng.standard_normal(next(iter(s.values()))[0].shape[1]))
              for s in seeds]
    decode_items = jax.vmap(_decode_compressed_columns, in_axes=(0, 0, None))

    def n_items(params):
        return jax.tree_util.tree_leaves(params)[0].shape[0]

    def row_block(index, name):
        """The Jacobian of output leaf *index* of *function*, with state of its own."""
        state = [None if s is None else {'mode': next(iter(s)), 'columns': None, 'valid': None,
                                         'settled': False}
                 for s in seeds]
        layout = {}

        @jax.jit
        def jvps(params, rows, kwargs):
            """One jvp per row for every item, *rows* holding each leaf's own tangent rows."""
            counts = [r.shape[0] for r in rows]

            def item(p, kw):
                leaves = treedef.flatten_up_to(p)
                tangents = [jnp.concatenate([r if i == j else jnp.zeros((k, jnp.size(leaf)), r.dtype)
                                             for i, (r, k) in enumerate(zip(rows, counts))])
                            .reshape((-1,) + jnp.shape(leaf))
                            for j, leaf in enumerate(leaves)]
                block = lambda q: jnp.ravel(jax.tree_util.tree_leaves(function(q, **kw))[index])
                linear = jax.linearize(block, p)[1]
                return jax.vmap(lambda *t: linear(treedef.unflatten(t)))(*tangents)

            return jax.vmap(item)(params, kwargs)

        def run(params, decoding, kwargs):
            """Each leaf's block of jvps for every item, all from one linearisation.

            A coloured leaf takes its colour seeds and the probe, and -- if it is
            in *decoding* -- the rank seeds too; any other leaf, one per entry.
            One pass, so values and ranks come from the same solve of anything
            inside *function*: an embedding re-run in a separate pass can put a
            point on an element boundary into the neighbouring element, and the
            decoded columns would belong to other values.
            """
            sizes = [int(np.size(leaf)) // n_items(params) for leaf in treedef.flatten_up_to(params)]
            rows = []
            for i, (m, s, st, probe, n) in enumerate(zip(meshes, seeds, state, probes, sizes)):
                if m is None:
                    rows.append(jnp.eye(n))
                    continue
                seed_values, seed_ranks, _ = s[st['mode']]
                rows.append(jnp.vstack([seed_values, probe[None]]
                                       + ([seed_ranks] if i in decoding else [])))
            out = jvps(params, rows, kwargs)                    #(n_items, n_jvps, n_res)
            bounds = np.cumsum([0] + [r.shape[0] for r in rows])
            blocks = []
            for i, (m, s, st, a, b) in enumerate(zip(meshes, seeds, state, bounds[:-1], bounds[1:])):
                block = out[:, a:b]
                if m is None:
                    blocks.append(block)
                    continue
                k = s[st['mode']][0].shape[0]
                blocks.append((block[:, :k], block[:, k], block[:, k + 1:] if i in decoding else None))
            return blocks, sizes

        def assemble(blocks, sizes):
            """The CSR matrix, its layout kept until some leaf's pattern changes.

            The layout -- where each kept entry sits among the jvps, in CSR order
            -- is found on the device, so only the nonzeros and their columns
            cross to the host.  A leaf taking one jvp per entry keeps every
            entry.  Leaf *j* of item *b* takes columns from ``n_items * offset_j
            + b * size_j``, the order :func:`jax.flatten_util.ravel_pytree` gives
            a batched tree.
            """
            n_b = (blocks[0] if meshes[0] is None else blocks[0][0]).shape[0]
            offsets = np.cumsum([0] + sizes)
            item = jnp.arange(n_b)[:, None, None]
            values, columns, valid = [], [], []
            for i, (m, st, b) in enumerate(zip(meshes, state, blocks)):
                first = n_b * offsets[i] + item * sizes[i]
                if m is None:
                    values.append(b)
                    columns.append(jnp.arange(b.shape[1])[None, :, None] + first + jnp.zeros(b.shape, int))
                    valid.append(jnp.ones(b.shape, bool))
                else:
                    values.append(b[0])
                    columns.append(st['columns'] + first)
                    valid.append(st['valid'])
            values = jnp.concatenate(values, axis=1)
            n_res = values.shape[2]
            if layout.get('n_res') != n_res:
                valid = jnp.concatenate(valid, axis=1)
                order, indices, indptr = _csr_layout(valid, jnp.concatenate(columns, axis=1),
                                                     _bucket(int(valid.sum())))
                nnz = int(indptr[-1])
                layout.update(n_res=n_res, order=order[:nnz],
                              indices=np.asarray(indices[:nnz]), indptr=np.asarray(indptr))
            data = np.asarray(jnp.take(values.ravel(), layout['order']))
            return scipy.sparse.csr_array((data, layout['indices'], layout['indptr']),
                                          shape=(n_b * n_res, n_b * offsets[-1]))

        def decode(i, blocks, params, kwargs):
            """Decode leaf *i*'s columns, trying its colourings in turn from the current one.

            A decode that agrees with the pattern held, column for column
            wherever both have an entry, is folded into it
            (:func:`_merge_patterns`).  The leaf is *settled* once a decode adds
            nothing: from then on it is trusted while the probe agrees, and
            until then each call decodes it in its first pass.  A decode that
            disagrees replaces the pattern.
            """
            st, tried = state[i], list(seeds[i])
            for mode in tried[tried.index(st['mode']):]:
                if mode != st['mode'] or blocks[i][2] is None:
                    st['mode'] = mode
                    blocks, _ = run(params, {i}, kwargs)
                values, along, ranked = blocks[i]
                columns, valid = decode_items(ranked, values, seeds[i][mode][2])
                if _agrees(values, columns, valid, probes[i], along):
                    break
            else:
                raise ValueError(f"no colouring reproduces the derivative of {name}: some residual "
                                 "entry draws on two parameters of one colour")
            settled = False
            if st['columns'] is not None and st['columns'].shape == columns.shape:
                merged_columns, merged_valid, conflict, grown = _merge_patterns(
                    columns, valid, st['columns'], st['valid'])
                if not conflict:
                    columns, valid, settled = merged_columns, merged_valid, not bool(grown)
            st.update(columns=columns, valid=valid, settled=settled)
            if not settled:
                layout.clear()
            return blocks

        def jac(params, kwargs):
            decoding = {i for i, st in enumerate(state) if st is not None and not st['settled']}
            blocks, sizes = run(params, decoding, kwargs)
            for i, st in enumerate(state):
                if st is None:
                    continue
                if i not in decoding:
                    values, along, _ = blocks[i]
                    if (st['columns'].shape == values.shape
                            and _agrees(values, st['columns'], st['valid'], probes[i], along)):
                        continue
                    st['settled'] = False
                    blocks, _ = run(params, {i}, kwargs)
                blocks = decode(i, blocks, params, kwargs)
            return assemble(blocks, sizes)

        return jac

    row_blocks = []

    def jac(x, **kwargs):
        params = unravel(jnp.asarray(x))
        if not batched:
            params, kwargs = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], (params, kwargs))
        if not row_blocks:
            shapes = jax.eval_shape(lambda p, kw: function(p, **kw),
                                    *jax.tree_util.tree_map(lambda x: x[0], (params, kwargs)))
            paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(shapes)]
            row_blocks.extend(row_block(i, f"the output {path}" if path else "the residual")
                              for i, path in enumerate(paths))
        parts = [block(params, kwargs) for block in row_blocks]
        return parts[0] if len(parts) == 1 else scipy.sparse.vstack(parts, format='csr')

    @jax.jit
    def residual(x, **kwargs):
        params = unravel(x)
        if not batched:
            return ravel_pytree(function(params, **kwargs))[0]
        return ravel_pytree(jax.vmap(lambda p, kw: function(p, **kw))(params, kwargs))[0]

    return residual, jac
