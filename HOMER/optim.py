import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial


import scipy

def jax_comp_kdtree_distance_query(fit_data, kdtree_args=None):
    kd_tree_args = {} if kdtree_args is None else kdtree_args
    local_fit_data = fit_data
    tree = scipy.spatial.KDTree(fit_data)

    def get_distances(pts):
        _, i = tree.query(pts, k=1, **kd_tree_args)
        dists = pts - local_fit_data[i]
        return dists.flatten().astype("float32")

    @jax.custom_jvp
    def distances(data):
        # data = jnp.asarray(data).squeeze()
        try:
            dists = jax.pure_callback(
                get_distances,
                jax.ShapeDtypeStruct((data.shape[0] * 3,), data.dtype),
                data,
            )
        except:
            breakpoint()
        return dists

    @distances.defjvp
    def distances_jax_deriv(primal, co_tangents):
        x, = primal
        co_tangent, = co_tangents
        # derivs = jnp.ones(data.shape[0]) * jnp.eye(3)
        # return derivs.reshape((-1, 3))
        primal_comp = distances(x)
        return primal_comp, co_tangent.flatten()
    
    return distances
        
        
def jax_comp_kdtree_normal_distance_query(fit_data, normals, kdtree_args=None):
    kd_tree_args = {} if kdtree_args is None else kdtree_args
    local_fit_data = fit_data
    tree = scipy.spatial.KDTree(fit_data)
    normal_data = normals

    def get_distances(pts):
        _, i = tree.query(pts, k=1, **kd_tree_args)
        dists = (pts - local_fit_data[i]) * normal_data[i]
        return dists.flatten().astype("float32"), i

    @jax.custom_jvp
    def distances(data):
        # data = jnp.asarray(data).squeeze()
        try:
            dists, i = jax.pure_callback(
                get_distances,
                jax.ShapeDtypeStruct((data.shape[0] * 3,), data.dtype),
                data,
            )
        except:
            breakpoint()
        return dists

    @distances.defjvp
    def distances_jax_deriv(primal, co_tangents):
        x, = primal
        primal_comp, i = distances(x)

        co_tangent_intermediate, = co_tangents
        co_tangent = co_tangent_intermediate * normal_data[i]
        # derivs = jnp.ones(data.shape[0]) * jnp.eye(3)
        # return derivs.reshape((-1, 3))
        return primal_comp, co_tangent.flatten()
    
    return distances




