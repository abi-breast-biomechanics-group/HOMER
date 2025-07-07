from functools import partial
from typing import Callable, Optional
from tqdm import tqdm
import jax
import sparsejac
import numpy as np
import scipy
import time

from matplotlib import pyplot as plt


def jacobian(cost_function: Optional[Callable] = None, init_estimate: Optional[None] = None, param_range = None, param_n = None, sparsity=None, further_args = None, sparse = True):

    if init_estimate is None and sparsity is None:
        raise ValueError("Code needs an initial estimate for meaningfull sparsity estimation")

    if cost_function is None:
        return partial(jacobian, init_estimate=init_estimate, param_range=param_range, param_n=param_n)

    if further_args is None:
        further_args = {}
    
    # get the input function
    fwd_func = jax.jit(cost_function)
    # get a sparsity matrix from that input
    if sparsity is None:
        sparsity = estimate_sparsity(partial(fwd_func, **further_args), init_estimate, param_range, param_n)

    if sparse:
        @jax.jit
        def sparse_jacobian(params, **kwargs):
            with jax.ensure_compile_time_eval():
                jacfwd = sparsejac.jacfwd(cost_function, sparsity=sparsity, argnums=0)
            return jacfwd(params, **kwargs)

        def scipy_compat(params, **kwargs):
            jax_sparse = sparse_jacobian(params, **kwargs)
            return scipy.sparse.coo_array(
                (jax_sparse.data, (jax_sparse.indices[:, 0], jax_sparse.indices[:, 1])),
                shape = jax_sparse.shape,
            )
    else:

        def scipy_compat(params, **kwargs):
            with jax.ensure_compile_time_eval():
                jac_fwd = jax.jit(jax.jacfwd(cost_function, argnums=0))
            return np.asarray(jac_fwd(params, **kwargs))

    return fwd_func, scipy_compat
    # return the compiled function and the compiled 
    
def estimate_sparsity(callable, init_estimate, param_range = None, param_n=None) -> jax.experimental.sparse.BCOO:
    # take the function, and run it with a range of parameters
    init = callable(init_estimate)
    rows = []
    cols = []

    print(init_estimate.shape)

    for idp in tqdm(range(init_estimate.shape[0]), desc="Building jacobian"):
            init_estimate[idp] += 1
            new_dif = callable(init_estimate)
            changed = np.where(new_dif != init)[0]
            rows.append(changed)
            cols.append(np.ones_like(changed) * idp)
            init_estimate[idp] -= 1

    inds = np.column_stack((
            np.concatenate(rows, axis=0),
            np.concatenate(cols, axis=0),
    ))

    sparsity = jax.experimental.sparse.BCOO((np.ones(inds.shape[0]), inds), shape=(init.shape[0], init_estimate.shape[0]))
    # plt.imshow(sparsity.todense()
    # plt.show()
    return sparsity



