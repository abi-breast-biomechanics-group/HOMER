"""
optim.py - how :func:`~HOMER.fitting.point_cloud_fit` measures distance to a
target point cloud.

Both functions here build a :class:`scipy.spatial.KDTree` over the reference
cloud and put it behind a :func:`jax.custom_jvp` callback, so a JAX residual
can query it and still differentiate.  The tree itself is SciPy's and runs on
the host: every query leaves the accelerator and comes back, which is why
these are used once per residual evaluation over a whole point cloud rather
than inside an inner loop.

* :func:`kdtree_distance_query` - nearest-neighbour distance to the cloud.
* :func:`kdtree_normal_distance_query` - the same, projected along the
  reference surface normals, for fitting an oriented cloud.

This module exists to serve ``point_cloud_fit``; it is not a general
nearest-neighbour interface.  The approximate nearest-neighbour search that
seeds :meth:`~HOMER.mesh.evaluation.embed_points` is a different thing
entirely, is written in JAX, and lives in :mod:`HOMER.utils`.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from HOMER.mesh import Mesh

import scipy

def kdtree_distance_query(fit_data, kdtree_args=None):
    """Differentiable nearest-neighbour distance to a fixed point cloud.

    Builds a :class:`scipy.spatial.KDTree` over *fit_data* once, then wraps
    queries against it in a :func:`jax.custom_jvp` callback so a JAX residual
    can use it and still differentiate.  The tree runs on the host, so each
    call costs a round-trip off the accelerator.

    :param fit_data:
        The reference cloud to measure against, shape ``(n_ref, 3)``.
    :param kdtree_args:
        Extra keyword arguments for ``tree.query``, e.g. ``{"workers": -1}``.
        ``None`` uses the SciPy defaults.

    :returns:
        ``distances(data)``, taking query points ``(n_pts, 3)`` and returning
        the flattened per-component offsets to their nearest reference
        points, shape ``(n_pts * 3,)``.
    """
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
        dists = jax.pure_callback(
            get_distances,
            jax.ShapeDtypeStruct((data.shape[0] * 3,), data.dtype),
            data,
        )
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
        
        
def kdtree_normal_distance_query(fit_data, normals, kdtree_args=None):
    """As :func:`kdtree_distance_query`, projected along the reference normals.

    Each offset is multiplied component-wise by the unit normal of the
    reference point it matched, so the residual measures distance along the
    surface and lets a point slide within the tangent plane.  That is what you
    want when fitting an oriented cloud, where the sample positions are less
    trustworthy than the surface they lie on.

    :param fit_data:
        The reference cloud, shape ``(n_ref, 3)``.
    :param normals:
        Unit normal at each reference point, same shape.
    :param kdtree_args:
        Extra keyword arguments for ``tree.query``.

    :returns:
        ``distances(data)``, with the same signature as
        :func:`kdtree_distance_query`'s.
    """
    kd_tree_args = {} if kdtree_args is None else kdtree_args
    local_fit_data = fit_data
    tree = scipy.spatial.KDTree(fit_data)
    normal_data = jnp.array(normals)

    def get_local_distances(pts):
        _, i = tree.query(pts, k=1, **kd_tree_args)
        dists = (pts - local_fit_data[i]) * normal_data[i]
        return dists.flatten().astype("float32"), i

    @jax.custom_jvp
    def distances(data):
        # data = jnp.asarray(data).squeeze()
        # try:
        dists, i = jax.pure_callback(
            get_local_distances,
            (jax.ShapeDtypeStruct((data.shape[0] * 3, ), data.dtype),
             jax.ShapeDtypeStruct((data.shape[0], ), jnp.zeros(3, dtype=int).dtype)),
            data,
        )
        # except:
        #     breakpoint()
        return dists
    def ind_distances(data):
        # data = jnp.asarray(data).squeeze()
        # try:
        dists, i = jax.pure_callback(
            get_local_distances,
            (jax.ShapeDtypeStruct((data.shape[0] * 3, ), data.dtype),
             jax.ShapeDtypeStruct((data.shape[0], ), jnp.zeros(3, dtype=int).dtype)),
            data,
        )
        return dists, i
    @distances.defjvp
    def distances_jax_deriv(primal, co_tangents):
        x, = primal
        primal_comp, i = ind_distances(x)

        co_tangent_intermediate, = co_tangents
        co_tangent = co_tangent_intermediate * normal_data[i]
        # derivs = jnp.ones(data.shape[0]) * jnp.eye(3)
        # return derivs.reshape((-1, 3))
        return primal_comp, co_tangent.flatten()
    
    return distances

