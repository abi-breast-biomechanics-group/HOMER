"""
fitting.py – High-level fitting utilities for HOMER meshes.

This module provides ready-to-use fitting functions that combine HOMER's
JAX-accelerated evaluation with scipy's ``least_squares`` optimiser.

The primary entry point is :func:`point_cloud_fit`, which constructs a
Sobolev-regularised point-cloud distance cost function and its sparse
Jacobian for use with ``scipy.optimize.least_squares``::

    from HOMER.fitting import point_cloud_fit
    from scipy.optimize import least_squares

    fitting_fn, jac_fn = point_cloud_fit(mesh, target_points, compile=True)
    result = least_squares(fitting_fn, mesh.optimisable_param_array,
                           jac=jac_fn, verbose=2)
    mesh.update_from_params(result.x)
"""

import numpy as np
import jax.numpy as jnp

from HOMER.mesher import Mesh
from HOMER.optim import jax_comp_kdtree_distance_query, jax_comp_kdtree_normal_distance_query
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt

def point_cloud_fit(mesh:Mesh, data, normals = None, res=20, compile=True, surface_only=False, sob_weight=0.01):
    """Build a point-cloud fitting cost function and its sparse Jacobian.

    Constructs a residual function suitable for ``scipy.optimize.least_squares``
    that measures the distance from the mesh surface to the target point cloud,
    with an optional Sobolev smoothness regularisation term.

    A KD-tree is built from *data* (and *normals* when provided) at
    construction time.  Every evaluation then queries this tree against the
    current mesh surface.

    Parameters
    ----------
    mesh:
        The :class:`~HOMER.mesher.Mesh` to fit.
    data:
        Target point cloud, shape ``(n_pts, 3)``.
    normals:
        Optional surface normals at each target point, shape ``(n_pts, 3)``.
        When provided, the distance metric is projected along the normal
        direction (useful for fitting to noisy oriented point clouds).
    res:
        Number of xi grid points per direction used to sample the mesh
        surface.
    compile:
        When ``True``, JIT-compiles the mesh evaluation functions
        (recommended for iterative optimisation).
    surface_only:
        When ``True``, only sample the mesh surface faces (for volume meshes).
    sob_weight:
        Scalar weight applied to the Sobolev smoothness term.  Increase to
        produce smoother fits at the cost of surface accuracy.

    Returns
    -------
    fitting_function : Callable
        Residual function ``(params) → residuals`` compatible with
        ``scipy.optimize.least_squares``.
    jacobian_fun : Callable
        Sparse Jacobian function ``(params) → scipy.sparse.coo_array``.

    Examples
    --------
    ::

        fitting_fn, jac_fn = point_cloud_fit(mesh, target_pts, compile=True)
        result = least_squares(fitting_fn, mesh.optimisable_param_array,
                               jac=jac_fn, verbose=2)
        mesh.update_from_params(result.x)
    """
    if normals is None:
        data_tree = jax_comp_kdtree_distance_query(data, kdtree_args={"workers":-1})
    else:
        data_tree = jax_comp_kdtree_normal_distance_query(data, normals, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res, surface=surface_only)
    # sob_points = mesh.gauss_grid([4, 4])

    mesh_elements = np.arange(len(mesh.elements))
    mesh.compile = compile #force the mesh to be compiled because we are running it a lot of times.
    mesh.generate_mesh()

    #get the initial params
    # create the handler object.

    def fitting_function(params: np.ndarray):
        outputs = []
        wpts = mesh.evaluate_embeddings(mesh_elements, eval_points, fit_params=params[:])
        dists = data_tree(wpts)
        outputs.append(dists.flatten())
        outputs.append(mesh.evaluate_sobolev().flatten() * sob_weight)
        outputs = jnp.concatenate(outputs)
        return outputs

    fitting_function, jacobian_fun = jacobian(fitting_function, init_estimate=mesh.optimisable_param_array)


    # plt.imshow(jacobian_fun(mesh.optimisable_param_array).todense())
    # plt.show()
    return fitting_function, jacobian_fun






