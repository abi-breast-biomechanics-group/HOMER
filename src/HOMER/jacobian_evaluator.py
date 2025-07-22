from functools import partial
from typing import Callable, Optional
from tqdm import tqdm
import jax
from jax.extend.core import Var
import sparsejac
import numpy as np
import scipy
import time

from matplotlib import pyplot as plt


def jacobian(cost_function: Optional[Callable] = None, init_estimate: Optional[None] = None, sparsity=None, further_args = None, sparse = True):
    """
    Given a jax compatible callable, returns both a compiled jax function, but also the autodifferentiated jacobian of the function.

    :param cost_function: the callable cost function.
    :param init_estimate: x0, the first estimate of the minimum of the cost function.

    :param sparsity: A pre known sparsity structure of the jacobian.
    :param further_args: Additional arguments to the cost function that are not compiled.
    :param sparse: Some jacobians may be dense, so will not benefit from sparse jac acceleration, and it can be skipped.

    :returns cost_function: The compiled cost function
    :returns jac_function: The jac function defining the derivative of outputs with respect to parameters of the cost function.
    """

    if init_estimate is None and sparsity is None:
        raise ValueError("Code needs an initial estimate for meaningfull sparsity estimation")

    if cost_function is None:
        return partial(jacobian, init_estimate=init_estimate)

    if further_args is None:
        further_args = {}
    
    # get the input function
    fwd_func = jax.jit(cost_function)
    # get a sparsity matrix from that input

    if sparse:
        if sparsity is None:
            sparsity = estimate_sparsity(partial(fwd_func, **further_args), init_estimate)

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
    
def estimate_sparsity(callable, init_estimate) -> jax.experimental.sparse.BCOO:
    """
    Estimates the sparsity of a callable with unit updates to parameters around the initial estimate.
    
    :param callable: The loss function
    :param init_estimate: the initial estimate.
    :return sparsity: A sparse boolean jax array inidcating the structure of the jacobian.
    """
    # take the function, and run it with a range of parameters
    init = callable(init_estimate)
    rows = []
    cols = []

    print(init_estimate.shape)

    start = time.time()
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
    # plt.imshow(sparsity.todense()[:1000, :1000])
    # plt.show()
    return sparsity

