
import numpy as np
import jax.numpy as jnp

from HOMER.mesher import Mesh
from HOMER.optim import jax_comp_kdtree_normal_distance_query
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt


def pca_fit(mesh: Mesh, pca_weight_matrix, mean_shape, data, data_normals, res =20):

    #for simplicity, we imagine that the pca_weight matrix
    data_tree = jax_comp_kdtree_normal_distance_query(data, data_normals, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)

    init_params = np.zeros(pca_weight_matrix.shape[0])

    def fitting_func(params):
        fit_params = jnp.sum(params[:, None] * pca_weight_matrix, axis=0) + mean_shape
        pts = mesh.evaluate_embeddings_in_every_element(eval_points, fit_params=fit_params)
        dif = data_tree(pts)
        return dif 
    
    fun, jac = jacobian(fitting_func, init_estimate=init_params)
    return fun, jac

 
def nonlinear_geometric_sobolev_prior_with_mask(mesh:Mesh, data, data_normal, res=20, w=0.1, dist_thresh=0.01):
    """
        An example that creates a fitting problem for a mesh.
    """

    data_tree = jax_comp_kdtree_normal_distance_query(data, data_normal, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)

    #do a first evaluation, only using the subset of evals 
    wpts = mesh.evaluate_embeddings_in_every_element(eval_points)
    dists = data_tree(wpts)
    normed_dists = np.linalg.norm(dists.reshape(-1,3), axis=-1)
    mask = normed_dists < dist_thresh

    sob_init = mesh.evaluate_sobolev()
    mesh.generate_mesh()

    def fitting_function(params):
        outputs = []
        wpts = mesh.evaluate_embeddings_in_every_element(eval_points, fit_params=params)[mask]
        dists = data_tree(wpts)
        outputs.append(dists.flatten())
        prior_dif = (mesh.evaluate_sobolev(fit_params=params) - sob_init) * w
        outputs.append(prior_dif)

        outputs = jnp.concatenate(outputs)
        return outputs

    init_params = mesh.optimisable_param_array

    fun, jac = jacobian(fitting_function, init_estimate=init_params)
    return fun, jac

