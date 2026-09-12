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
