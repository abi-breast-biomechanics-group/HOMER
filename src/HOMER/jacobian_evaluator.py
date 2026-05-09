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
from scipy.sparse import csr_matrix
from collections import defaultdict

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
            def scipy_sparse_jac(params, **kwargs):
                sparse_csr = sparsity(params)
                update_csr_jacobian(partial(fwd_func, **kwargs), params, sparse_csr)
                return sparse_csr
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
def _next_power_of_2(x):
    """Returns the smallest power of 2 greater than or equal to x."""
    return 1 if x == 0 else 2**(x - 1).bit_length()

def update_csr_jacobian(f, params, sparsity_csr, MAX_BATCH_SIZE=128):
    ts0 = time()
    M, N = sparsity_csr.shape
    indptr = sparsity_csr.indptr
    indices = sparsity_csr.indices
    
    if not np.issubdtype(sparsity_csr.data.dtype, np.floating):
        sparsity_csr.data = sparsity_csr.data.astype(np.float32)

    pattern_to_rows = defaultdict(list)
    exact_max_deps = 0
    
    for i in range(M):
        start, end = indptr[i], indptr[i+1]
        deps = indices[start:end]
        if len(deps) == 0:
            continue
            
        pattern_to_rows[tuple(deps)].append(i)
        if len(deps) > exact_max_deps:
            exact_max_deps = len(deps)

    if exact_max_deps == 0:
        sparsity_csr.data[:] = 0.0
        return sparsity_csr

    exact_max_groups = sum((len(rows) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE 
                           for rows in pattern_to_rows.values())

    # --- 2. Bucket to powers of 2 ---
    MAX_DEPS = _next_power_of_2(exact_max_deps)
    MAX_GROUPS = _next_power_of_2(exact_max_groups)

    # --- 3. Prepare static arrays ---
    row_indices = np.zeros((MAX_GROUPS, MAX_BATCH_SIZE), dtype=np.int32)
    row_masks = np.zeros((MAX_GROUPS, MAX_BATCH_SIZE), dtype=bool)
    col_indices = np.zeros((MAX_GROUPS, MAX_DEPS), dtype=np.int32)
    col_masks = np.zeros((MAX_GROUPS, MAX_DEPS), dtype=bool)
    data_loc_mapping = np.zeros((MAX_GROUPS, MAX_BATCH_SIZE, MAX_DEPS), dtype=np.int32)
    group_idx = 0
    
    # --- 4. FAST Vectorized Array Population ---
    for deps_tuple, rows in pattern_to_rows.items():
        deps_len = len(deps_tuple)
        deps_arr = np.array(deps_tuple, dtype=np.int32)
        rows_arr = np.array(rows, dtype=np.int32)
        n_rows = len(rows_arr)
        
        # Calculate chunks and padding
        num_chunks = (n_rows + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        pad_len = num_chunks * MAX_BATCH_SIZE - n_rows
        
        if pad_len > 0:
            rows_padded = np.pad(rows_arr, (0, pad_len), constant_values=0)
            masks_padded = np.pad(np.ones(n_rows, dtype=bool), (0, pad_len), constant_values=False)
        else:
            rows_padded = rows_arr
            masks_padded = np.ones(n_rows, dtype=bool)
            
        # Reshape to 2D blocks
        rows_chunked = rows_padded.reshape(num_chunks, MAX_BATCH_SIZE)
        masks_chunked = masks_padded.reshape(num_chunks, MAX_BATCH_SIZE)
        
        # Target indices in the static arrays
        g_start = group_idx
        g_end = group_idx + num_chunks
        
        row_indices[g_start:g_end, :] = rows_chunked
        row_masks[g_start:g_end, :] = masks_chunked
        
        col_indices[g_start:g_end, :deps_len] = deps_arr
        col_masks[g_start:g_end, :deps_len] = True
        
        starts = indptr[rows_chunked][..., np.newaxis] # Shape: (num_chunks, MAX_BATCH_SIZE, 1)
        offsets = np.arange(deps_len)[np.newaxis, np.newaxis, :] # Shape: (1, 1, deps_len)
        
        data_loc_mapping[g_start:g_end, :, :deps_len] = starts + offsets
        
        group_idx += num_chunks
    ts1 = time()
    print(f" to0k {ts1-ts0} s")
    batched_grads = extract_vmapped_sparse_jacobian(
        f, params, row_indices, row_masks, col_indices, col_masks
    )
    
    grads_np = np.asarray(batched_grads)
    combined_mask = row_masks[:, :, np.newaxis] & col_masks[:, np.newaxis, :]
    valid_grads = grads_np[combined_mask]
    valid_locs = data_loc_mapping[combined_mask]
    
    sparsity_csr.data[valid_locs] = valid_grads

    return sparsity_csr

@partial(jax.jit, static_argnums=0)
def extract_batched_sparse_jacobian(f, x_full, row_indices, row_masks, col_indices, col_masks):
    """
    row_indices/masks shape: (MAX_GROUPS, MAX_BATCH_SIZE)
    col_indices/masks shape: (MAX_GROUPS, MAX_DEPS)
    """
    y, vjp_fn = jax.vjp(f, x_full)
    M = y.shape[0]
    
    # We vmap the pullback so it can accept a matrix of cotangents
    # vjp_fn normally takes a 1D vector. Now it takes a 2D batch.
    batched_vjp = jax.vmap(vjp_fn)

    def scan_body(carry, i):
        r_idx = row_indices[i]  # shape (MAX_BATCH_SIZE,)
        r_mask = row_masks[i]   # shape (MAX_BATCH_SIZE,)
        c_idx = col_indices[i]  # shape (MAX_DEPS,)
        c_mask = col_masks[i]   # shape (MAX_DEPS,)
        
        # Build a batch of one-hot cotangent vectors
        e_batch = jnp.zeros((row_indices.shape[1], M))
        # Vectorized assignment: Place 1.0 at the correct row indices if mask is True
        e_batch = e_batch.at[jnp.arange(row_indices.shape[1]), r_idx].set(r_mask.astype(jnp.float32))
        
        # Execute the backward pass for the entire batch at once
        # dense_grads shape: (MAX_BATCH_SIZE, N)
        dense_grads = batched_vjp(e_batch)[0] 
        
        # Advanced indexing to slice out only the active columns for this group
        # sparse_batch shape: (MAX_BATCH_SIZE, MAX_DEPS)
        sparse_batch = dense_grads[:, c_idx]
        
        # Zero out padding on both axes
        sparse_batch = jnp.where(c_mask[None, :], sparse_batch, 0.0) # Mask columns
        sparse_batch = jnp.where(r_mask[:, None], sparse_batch, 0.0) # Mask rows
        
        return carry, sparse_batch

    # Scan over the groups
    _, batched_sparse_grads = jax.lax.scan(scan_body, None, jnp.arange(row_indices.shape[0]))
    
    # Returns shape (MAX_GROUPS, MAX_BATCH_SIZE, MAX_DEPS)
    return batched_sparse_grads

@partial(jax.jit, static_argnums=0)
def extract_vmapped_sparse_jacobian(f, x_full, row_indices, row_masks, col_indices, col_masks):
    y, vjp_fn = jax.vjp(f, x_full)
    M = y.shape[0]
    
    # Inner function that computes gradients for a single group
    def group_fn(r_idx, r_mask, c_idx, c_mask):
        # Create one-hot cotangents
        e_batch = jnp.zeros((row_indices.shape[1], M))
        e_batch = e_batch.at[jnp.arange(row_indices.shape[1]), r_idx].set(r_mask.astype(jnp.float32))
        
        # Pullback for this batch
        dense_grads = jax.vmap(vjp_fn)(e_batch)[0]
        
        # Slicing and masking
        sparse_batch = dense_grads[:, c_idx]
        sparse_batch = jnp.where(c_mask[None, :], sparse_batch, 0.0)
        sparse_batch = jnp.where(r_mask[:, None], sparse_batch, 0.0)
        return sparse_batch

    # Vectorize across the MAX_GROUPS dimension instead of scanning sequentially!
    batched_sparse_grads = jax.vmap(group_fn)(row_indices, row_masks, col_indices, col_masks)
    
    return batched_sparse_grads
