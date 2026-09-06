"""
element_eval.py - building the per-element evaluation kernels.

Given the bases of an element and the ordering of their tensor-product weights,
the ``make_*`` factories here close over that structure once and return the
JAX functions a mesh calls for every subsequent evaluation.  :data:`GAUSS`
tabulates the quadrature rules, and :func:`volume_quadrature_order` picks the
rule a given basis needs.
"""

import logging

import numpy as np
import jax.numpy as jnp

from HOMER.basis_definitions import N2_weights, N3_weights, BasisGroup


def make_eval(basis_funcs: BasisGroup, bp_inds:list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a 
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])
            w1 = basis_funcs[1].fn(xis[:, 1]) 
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval


def make_deriv_eval(basis_funcs, bp_inds):
    """
    Returns a JAX-compliant evaluator function.
    - basis_funcs length must be 2 or 3
    - bp_inds should be static for best compilation behavior
    """
    bp_inds = jnp.asarray(bp_inds, dtype=jnp.int32)
    ndim = len(basis_funcs)

    if ndim == 2:
        def xi_eval(elem_params, xis, d_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])  
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1]) 
            weights = N2_weights(w0, w1, bp_inds)            
            params2 = elem_params.reshape(weights.shape[0], -1)  
            out = jnp.einsum("bo,bp->po", params2, weights)     
            return out.reshape(-1)

    elif ndim == 3:
        def xi_eval(elem_params, xis, d_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            w2 = basis_funcs[2].deriv[d_inds[2]](xis[:, 2])
            weights = N3_weights(w0, w1, w2, bp_inds)
            params2 = elem_params.reshape(weights.shape[0], -1)
            out = jnp.einsum("bo,bp->po", params2, weights)
            return out.reshape(-1)

    else:
        raise ValueError("Currently, meshes must be 2D or 3D")

    return xi_eval


def make_weight_eval(basis_funcs: BasisGroup, bp_inds):
    if len(basis_funcs) == 2:
        def xi_eval(xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            return weights
    elif len(basis_funcs) == 3:
        def xi_eval(xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            return weights
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval


def volume_quadrature_order(basis_functions: BasisGroup) -> list[int]:
    """Gauss points per direction needed to integrate det(J) exactly.

    A map of polynomial degree ``p_d`` in direction ``d`` has a Jacobian whose
    ``d``-th column has dropped to degree ``p_d - 1`` in that direction while
    the other two columns still carry degree ``p_d``.  Every term of the 3x3
    determinant takes one entry from each column, so det(J) reaches degree
    ``3 * p_d - 1`` in ``xi_d``.  An ``n``-point Gauss rule is exact to degree
    ``2n - 1``, so ``n_d = ceil(3 * p_d / 2)``.

    Using the basis order itself -- the obvious choice, and what this used to
    do -- under-integrates every element that is not affine.  A distorted
    trilinear hexahedron came out 1% wrong, and a tricubic Hermite element
    6e-4 wrong, with the error *not* shrinking as the quadrature was refined
    because the rule was never refined at all.

    The tabulated rules stop at :data:`GAUSS`'s highest order, which covers
    every basis HOMER ships (``L4Basis``, degree 4, needs 6 points).  A
    higher-degree basis is clamped to the table and warned about, since an
    approximate answer beats no answer.
    """
    max_order = max(GAUSS)
    orders = []
    for basis in basis_functions:
        needed = int(np.ceil(3 * basis.order / 2))
        if needed > max_order:
            logging.warning(
                f"Exact volume quadrature for {basis.name} (degree {basis.order}) "
                f"needs {needed} Gauss points per direction, but only {max_order} are "
                f"tabulated; the volume will be under-integrated."
            )
            needed = max_order
        orders.append(needed)
    return orders


GAUSS = { 
        1:[np.array([[0.5]]),
           np.array([1])],
        2:[np.array([[0.21132486540518708], [0.78867513459481287]]),
           np.array([0.5, 0.5])],
        3:[np.array([[0.1127016653792583], [0.5], [0.8872983346207417]]), 
           np.array([5./18., 4./9., 5./18])],
        4:[np.array([[0.33000947820757187, 0.6699905217924281, 0.06943184420297371, 0.9305681557970262]]).T,
           np.array([0.32607257743127305, 0.32607257743127305, 0.1739274225687269, 0.1739274225687269])],
        5:[np.array([[0.5, 0.230765344947, 0.769234655053, 0.0469100770307, 0.953089922969]]).T,
           np.array([0.284444444444, 0.23931433525, 0.23931433525, 0.118463442528, 0.118463442528])],
        6:[np.array([[0.8306046932331322, 0.1693953067668678, 0.3806904069584016, 0.6193095930415985, 0.0337652428984240, 0.9662347571015760]]).T,
           np.array([0.1803807865240693, 0.1803807865240693, 0.2339569672863455, 0.2339569672863455, 0.0856622461895852, 0.0856622461895852])],
}
