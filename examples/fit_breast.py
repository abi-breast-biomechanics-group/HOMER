
import numpy as np
import jax.numpy as jnp
import pyvista as pv

<<<<<<< HEAD
from HOMER.mesher import Mesh, EVAL_PATTERN
from HOMER.optim import jax_comp_kdtree_normal_distance_query
=======
from HOMER.mesher import Mesh
from HOMER.optim import jax_comp_kdtree_distance_query, jax_comp_kdtree_normal_distance_query
>>>>>>> Anna
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt

def normal_fit(mesh: Mesh,
            rib_data, rib_data_normal,
            skin_data, skin_data_normal,
            rib_range, skin_range,
            res = 20, w=0.1):

    #for simplicity, we imagine that the pca_weight matrix
    rib_data_tree = jax_comp_kdtree_normal_distance_query(rib_data, rib_data_normal, kdtree_args={"workers":-1})
    skin_data_tree = jax_comp_kdtree_normal_distance_query(skin_data, skin_data_normal, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)

    init_params = mesh.optimisable_param_array.copy()

    def_sob = mesh.evaluate_sobolev()
    def fitting_func(params):
        fit_params = params
        rib_pts = mesh.evaluate_embeddings(rib_range, eval_points, fit_params=fit_params)
        skin_pts = mesh.evaluate_embeddings(skin_range, eval_points, fit_params=fit_params)
        rib_dif = rib_data_tree(rib_pts)
        skin_dif = skin_data_tree(skin_pts)
        sob_dif = (def_sob - mesh.evaluate_sobolev(fit_params=fit_params)) * w
        return jnp.concatenate((rib_dif, skin_dif, sob_dif))

    def update_fun(mesh, params):
        fit_params = params
        mesh.update_from_params(fit_params)

    fun, jac = jacobian(fitting_func, init_estimate=init_params)
    return fun, jac, init_params, update_fun


def dep_node_normal_fit(mesh: Mesh,
            rib_data, rib_data_normal,
            skin_data, skin_data_normal,
            dep_nodes, div_scalar,
            rib_range, skin_range,
            res = 20, w=0.1):

    rib_data_tree = jax_comp_kdtree_normal_distance_query(rib_data, rib_data_normal, kdtree_args={"workers":-1})
    skin_data_tree = jax_comp_kdtree_normal_distance_query(skin_data, skin_data_normal, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)


    #depnode stuff
    for node in dep_nodes: #fix all depnodes so not normally optimisable
        mesh.get_node(node[0]).fix_parameter(['loc', 'du', 'dv', 'dudv'])
    mesh.generate_mesh()
    depnode_params = np.array(mesh.associated_node_index(
        [
            'loc',
            'du',
            'dv',
            'dudv'
        ], [a[0] for a in dep_nodes], node_by_id=True)).astype(int)

    init_params = mesh.optimisable_param_array.copy()
    def_sob = mesh.evaluate_sobolev()
    full_params = jnp.asarray(mesh.true_param_array).copy()
    param_bool = mesh.optimisable_param_bool.copy()

    def fitting_func(params):
        fit_params = jnp.asarray(params) #these are the optimisable params from the param array
        #evaluate the depnode params
        depnode_evals = jnp.concatenate([
            mesh.evaluate_ele_xi_pair_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], fit_params=fit_params)[:, None],
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 0], fit_params=fit_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [0, 1], fit_params=fit_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 1], fit_params=fit_params)[:, None] * div_scalar,
        ], axis=1)

        full_params = jnp.asarray(mesh.true_param_array).copy()
        #update the array with these depnode params
        full_params = full_params.at[param_bool].set(fit_params)
        full_params = full_params.at[depnode_params].set(depnode_evals)

        rib_pts = mesh.evaluate_embeddings(rib_range, eval_points, fit_params=full_params)
        skin_pts = mesh.evaluate_embeddings(skin_range, eval_points, fit_params=full_params)
        rib_dif = rib_data_tree(rib_pts)
        skin_dif = skin_data_tree(skin_pts)
        sob_dif = (def_sob - mesh.evaluate_sobolev(fit_params=fit_params)) * w
        return jnp.concatenate((rib_dif, skin_dif, sob_dif))

    fitting_func(init_params)

    def update_fun(mesh, params):
        fit_params =  params
        depnode_evals = jnp.concatenate([
            mesh.evaluate_ele_xi_pair_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], fit_params=fit_params)[:, None],
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 0], fit_params=fit_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [0, 1], fit_params=fit_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 1], fit_params=fit_params)[:, None] * div_scalar,
        ], axis=1)

        full_params = jnp.asarray(mesh.true_param_array).copy()
        full_params = full_params.at[param_bool].set(params)
        full_params = full_params.at[depnode_params].set(depnode_evals)
        mesh.update_from_params(np.array(full_params))

    fun, jac = jacobian(fitting_func, init_estimate=init_params)
    return fun, jac, init_params, update_fun












def pca_fit(mesh: Mesh, pca_weight_matrix, mean_shape,
            rib_data, rib_data_normal,
            skin_data, skin_data_normal,
            rib_range, skin_range,
            res = 20):

    #for simplicity, we imagine that the pca_weight matrix
    rib_data_tree = jax_comp_kdtree_normal_distance_query(rib_data, rib_data_normal, kdtree_args={"workers":-1})
    skin_data_tree = jax_comp_kdtree_normal_distance_query(skin_data, skin_data_normal, kdtree_args={"workers":-1})
    # rib_data_tree = jax_comp_kdtree_distance_query(rib_data, kdtree_args={"workers":-1})
    # skin_data_tree = jax_comp_kdtree_distance_query(skin_data, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)

    init_params = np.zeros(pca_weight_matrix.shape[0])

    def fitting_func(params):
        fit_params = jnp.sum(params[:, None] * pca_weight_matrix, axis=0) + mean_shape
        rib_pts = mesh.evaluate_embeddings(rib_range, eval_points, fit_params=fit_params)
        skin_pts = mesh.evaluate_embeddings(skin_range, eval_points, fit_params=fit_params)
        rib_dif = rib_data_tree(rib_pts)
        skin_dif = skin_data_tree(skin_pts)
        return jnp.concatenate((rib_dif, skin_dif))

    def update_fun(mesh, params):
        fit_params = jnp.sum(params[:, None] * pca_weight_matrix, axis=0) + mean_shape
        mesh.update_from_params(fit_params)

    fun, jac = jacobian(fitting_func, init_estimate=init_params)
    return fun, jac, init_params, update_fun

 
def nonlinear_geometric_sobolev_prior_with_mask(mesh:Mesh,
                                                rib_data, rib_data_normal,
                                                skin_data, skin_data_normal,
                                                rib_range, skin_range,
                                                res=20, w=0.1, dist_thresh=0.01):
    """
        An example that creates a fitting problem for a mesh.
    """

    # data_tree = jax_comp_kdtree_normal_distance_query(data, data_normal, kdtree_args={"workers":-1})
    rib_data_tree = jax_comp_kdtree_distance_query(rib_data, kdtree_args={"workers": -1})
    skin_data_tree = jax_comp_kdtree_distance_query(skin_data, kdtree_args={"workers": -1})
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

