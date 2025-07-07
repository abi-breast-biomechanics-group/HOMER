
import numpy as np
import jax.numpy as jnp

from HOMER.mesher import Mesh, EVAL_PATTERN
from HOMER.optim import jax_comp_kdtree_normal_distance_query
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt



# first thing to do is fix the nodes.

def make_function_to_update_depnods(mesh:Mesh, dependent_nodes, elem_associated, elem_locs, deriv_scale, params):


    #get the derivative pattern to eval:
    d_to_eval = []
    e = mesh.elements[0] 
    deriv_bound = np.where([np.any([st[:2] == 'dx' for st in b.weights]) for b in e.basis_functions] )[0]
    for d_val in EVAL_PATTERN[len(e.used_node_fields)]:
        #calculate the additional derivatives in the directions that need them
        derivs = [0,0,0]
        for dl, di in zip(deriv_bound, d_val): 
            derivs[dl] = di
        d_to_eval.append(derivs)


    def 









    for dep_node_id, elem_num, eloc, dscale in zip(dependent_nodes, elem_associated, elem_locs, deriv_scale):
        template_node = 
        e = mesh.elements[elem_num] 
        deriv_bound = np.where([np.any([st[:2] == 'dx' for st in b.weights]) for b in e.basis_functions] )[0]
        for d_val in EVAL_PATTERN[len(e.used_node_fields)]:
            #calculate the additional derivatives in the directions that need them
            derivs = [0,0,0]
            for dl, di in zip(deriv_bound, d_val): 
                derivs[dl] = di
            additional_pts.append(self.evaluate_deriv_embeddings(np.array([ide]), eval_pts, derivs=derivs)/d_scale)





def pca_fit(mesh: Mesh, pca_weight_matrix, mean_shape, data, data_normals, res = 20):

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

