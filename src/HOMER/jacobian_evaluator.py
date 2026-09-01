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
"""

from functools import partial
from time import time
from typing import Callable, Optional
import jax
import jax.numpy as jnp
import sparsejac
import numpy as np
import scipy

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

