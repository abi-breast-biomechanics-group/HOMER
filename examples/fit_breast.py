import jax.experimental.mosaic.gpu.utils
import numpy as np
import jax.numpy as jnp
import pyvista as pv

from HOMER.mesher import Mesh
from HOMER.optim import jax_comp_kdtree_distance_query, jax_comp_kdtree_normal_distance_query
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt

def evaluate_embeddings_fixed(mesh, element_ids, xis, fit_params=None):
    element_ids = np.array(element_ids)
    xis = np.array(xis)

    unique_elements = np.unique(element_ids)
    outputs = []

    for e in unique_elements:
        mask = (element_ids == e)
        xis_subset = xis[mask]
        pts = mesh.evaluate_embeddings([e], xis_subset, fit_params=fit_params)
        outputs.append(pts)

    result = jnp.zeros((len(element_ids), 3))
    for e, pts in zip(unique_elements, outputs):
        mask = (element_ids == e)
        result = result.at[mask].set(pts)

    return result

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

def dif_sob_dep_node_normal_fit(mesh: Mesh,
                                rib_data, rib_data_normal,
                                skin_data, skin_data_normal,
                                dep_nodes, div_scalar,
                                rib_range, skin_range, strict_range, breast_range,
                                prior_mesh,
                                nodes_to_fix_x,
                                nodes_to_fix_group0,
                                nodes_to_fix_group1,
                                # nodes_to_fix_group2,
                                superior_inferior_nodes,
                                left_nipple = None,
                                right_nipple = None,
                                fixed_x = 0,
                                res = 20,
                                w_array = None,
                                ):
    """
    dep_nodes: side nodes that are constrained to be at a straight line.
    strict_range: ELEMENT ids that has strict (high) soblev weighting.
    fixed_y: used for fixing sternum nodes.
    nodes_to_fix_group0: inferior skin nodes + ribcage nodes, constrained to share the same z
    nodes_to_fix_group1: superior skin nodes
    nodes_to_fix_group2: superior rib nodes
    """

    for node in mesh.nodes:
        node.fixed_params = {}
    # rib_data_tree = jax_comp_kdtree_normal_distance_query(rib_data, rib_data_normal, kdtree_args={"workers":-1})
    # skin_data_tree = jax_comp_kdtree_normal_distance_query(skin_data, skin_data_normal, kdtree_args={"workers":-1})
    rib_data_tree = jax_comp_kdtree_distance_query(rib_data, kdtree_args={"workers":-1})
    skin_data_tree = jax_comp_kdtree_distance_query(skin_data, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)

    # depnode stuff
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


    # minus 1 number of elements
    # if use pca mean as sob prior
    def_sob = prior_mesh.evaluate_sobolev()
    # if not
    # def_sob = mesh.evaluate_sobolev()

    # shape need to apply different weight, weight can be an array
    # number of elements
    n_elem = mesh.ele_map.shape[0]
    per_ele_gauss = def_sob.reshape(-1, n_elem)
    # define weights
    if w_array is None:
        # ribcage
        w_array = np.full(n_elem, 0.1)
        strict_range.append(5)
        strict_range.append(59)
        w_array[strict_range] = 2
        w_array[breast_range] = 0.5
    w_array = jnp.asarray(w_array)

    ######## fixing the x of a few datapoints (sternum nodes here)
    # updated fix_params (mesher), new idx input = the index wanna be fixed (x,y,z) = (0,1,2)
    for node_num in nodes_to_fix_x:
        w_node = mesh.get_node(node_num)
        w_node.fix_parameter('loc', inds=[0], values=np.array(fixed_x))
        w_node.fix_parameter('dv', inds=[0], values=0)
        print(w_node.id, ':', w_node.loc)


    # Add constraint: fix dz/dxi1 for superior/inferior nodes
    for node_num in superior_inferior_nodes:
        w_node = mesh.get_node(node_num)
        w_node.fix_parameter('du', inds=[2], values=np.zeros(1))

    # Fix shoulder nodes (with good pca)
    # element id: 5, 59
    # shoulder_edge_nodes = [6, 13, 82, 88]
    # for node in shoulder_edge_nodes:
    #     shoulder_node = mesh.get_node(node)
    #     shoulder_node.fix_parameter('loc')

    ######## fixing another group's z to be shared: group0: inferior nodes (skin+rib)
    z_g0 = []
    for node_num in nodes_to_fix_group0:
        w_node = mesh.get_node(node_num)
        w_node.fix_parameter('loc', inds=[2])
        z_g0.append(w_node.loc[2])
    z_param_g0 = np.mean(z_g0)

    node_loc_inds_group0 = mesh.associated_node_index(['loc'], nodes_to_gather=nodes_to_fix_group0)
    node_loc_inds_group0 = np.array(node_loc_inds_group0)[:, 0, 2].flatten()
    print(node_loc_inds_group0)

    ######## fixing another group's z to be shared: group1: superior skin & rib
    z_g1 = []
    for node_num in nodes_to_fix_group1:
        w_node = mesh.get_node(node_num)
        w_node.fix_parameter('loc', inds=[2])
        # original z (pca)
        z_g1.append(w_node.loc[2])
    # z_param_g1 = np.max(z_g1)
    node_loc_inds_group1 = mesh.associated_node_index(['loc'], nodes_to_gather=nodes_to_fix_group1)
    node_loc_inds_group1 = np.array(node_loc_inds_group1)[:, 0, 2].flatten()

    ######## fixing another group's z to be shared: group2: superior rib
    # z_g2 = []
    # for node_num in nodes_to_fix_group2:
    #     w_node = mesh.get_node(node_num)
    #     w_node.fix_parameter('loc', inds=[2])
    #     z_g2.append(w_node.loc[2])
    # z_param_g2 = np.max(z_g2)
    # node_loc_inds_group2 = mesh.associated_node_index(['loc'], nodes_to_gather=nodes_to_fix_group2)
    # node_loc_inds_group2 = np.array(node_loc_inds_group2)[:, 0, 2].flatten()


    # fixing for the nipples (if landmarks available)
    if left_nipple is not None:
        left_nipple_node = mesh.get_node(163)
        left_nipple_node.fix_parameter('loc', np.array(left_nipple))
    if right_nipple is not None:
        right_nipple_node = mesh.get_node(210)
        right_nipple_node.fix_parameter('loc', np.array(right_nipple))
    mesh.generate_mesh()
    # fix shoulder edge nodes (refined): 6, 247, 13, 259, 82, 396, 88, 397
    # shoulder_edge_nodes=[6, 247, 13, 259, 82, 396, 88, 397]

    # unrefined: 6, 13, 82, 88


    # check fixed nodes

    # s = pv.Plotter()
    # mesh.plot(s, labels=True)
    # fixed = []
    # unfixed = []
    # for node in mesh.nodes:
    #     if len(node.fixed_params) != 0:
    #         fixed.append(node.loc)
    #     else:
    #         unfixed.append(node.loc)
    # s.add_mesh(np.array(fixed), color='r', render_points_as_spheres=True)
    # s.add_mesh(np.array(unfixed), color='g', render_points_as_spheres=True)
    # s.show()

    # raise ValueError()

    # groups
    init_fixed_group0_params = z_param_g0
    # init_fixed_group1_params = z_param_g1
    # init_fixed_group2_params = z_param_g2
    # init_fixed_group0_params (length1) + init_fixed_group1_params (length1) + init_fixed_group2_params (length1)
    n_fixed2 = 1 #3

    init_normal_params = mesh.optimisable_param_array.copy()
    init_params = np.concatenate((
        np.atleast_1d(init_fixed_group0_params),
        # np.atleast_1d(init_fixed_group1_params),
        # np.atleast_1d(init_fixed_group2_params),
        init_normal_params
    ))
    param_bool = mesh.optimisable_param_bool.copy()

    def fitting_func(params):
        unfixed_nodes = params[n_fixed2:]
        # z_g0, z_g1, z_g2 = params[:n_fixed2]
        z_g0 = params[:n_fixed2]

        ###### Turn group0 into 'uncompressed' parameters
        # explicit_xy_g0 = fixed_group0[1:].reshape(-1,2)
        #
        # explicit_xy_g1 = fixed_group1[1:].reshape(-1, 2)
        #
        # explicit_xy_g2 = fixed_group2[1:].reshape(-1, 2)

        full_params = jnp.asarray(mesh.true_param_array).copy()
        full_params = full_params.at[param_bool].set(unfixed_nodes)
        # For fixed groups

        full_params = full_params.at[node_loc_inds_group0].set(z_g0)
        full_params = full_params.at[node_loc_inds_group1].set(z_g1)
        # full_params = full_params.at[node_loc_inds_group2].set(z_g2)

        # jax.debug.print(str(z_g0))

        #these are the optimisable params from the param array
        #evaluate the depnode params
        depnode_evals = jnp.concatenate([
            mesh.evaluate_ele_xi_pair_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], fit_params=full_params)[:, None],
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 0], fit_params=full_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [0, 1], fit_params=full_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 1], fit_params=full_params)[:, None] * div_scalar,
        ], axis=1)

        #update the array with these depnode params
        full_params = full_params.at[depnode_params].set(depnode_evals)

        # Fit skin/rib separately
        for idx, elem in enumerate(mesh.elements):
            elem.id = idx
        mesh.element_id_to_ind = {elem.id: idx for idx, elem in enumerate(mesh.elements)}
        rib_point_element_ids = np.repeat(rib_range, len(eval_points))
        all_eval_points = np.tile(eval_points, (len(rib_range), 1))
        rib_pts = evaluate_embeddings_fixed(mesh, rib_point_element_ids, all_eval_points, fit_params=full_params)
        # rib_pts = mesh.evaluate_embeddings(rib_range, eval_points, fit_params=full_params)
        skin_pts = mesh.evaluate_embeddings(skin_range, eval_points, fit_params=full_params)
        rib_dif = rib_data_tree(rib_pts)
        skin_dif = skin_data_tree(skin_pts)

        # s = pv.Plotter()
        # mesh.plot(s)
        # s.add_mesh(np.array(rib_pts), color='r')
        # s.add_mesh(np.array(skin_pts), color='b')
        # print(len(rib_pts))
        # s.show()
        # w1 =
        # sob_dif = (def_sob - mesh.evaluate_sobolev(fit_params=fit_params)) * w
        cur_sob = mesh.evaluate_sobolev(fit_params=full_params).reshape(-1, n_elem)
        # Use prior
        # try to use pca mean as prior
        # sob_dif = (per_ele_gauss - cur_sob) * w_array
        # without prior, standard sob
        sob_dif = cur_sob * w_array
        sob_dif = sob_dif.reshape(-1)
        return jnp.concatenate((skin_dif, rib_dif, sob_dif))

    fitting_func(init_params)

    def update_fun(mesh, params):
        unfixed_nodes = params[n_fixed2:]
        # z_g0, z_g1, z_g2 = params[:n_fixed2]
        z_g0 = params[:n_fixed2]
        full_params = jnp.asarray(mesh.true_param_array).copy()
        full_params = full_params.at[param_bool].set(unfixed_nodes)

        full_params = full_params.at[node_loc_inds_group0].set(z_g0)
        full_params = full_params.at[node_loc_inds_group1].set(z_g1)
        # full_params = full_params.at[node_loc_inds_group2].set(z_g2)

        depnode_evals = jnp.concatenate([
            mesh.evaluate_ele_xi_pair_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], fit_params=full_params)[:, None],
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 0], fit_params=full_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [0, 1], fit_params=full_params)[:, None] * div_scalar,
            mesh.evaluate_ele_xi_pair_deriv_embeddings(dep_nodes[:, 1], dep_nodes[:, 2:], [1, 1], fit_params=full_params)[:, None] * div_scalar,
        ], axis=1)

        full_params = full_params.at[depnode_params].set(depnode_evals)
        mesh.update_from_params(np.array(full_params))

    fun, jac = jacobian(fitting_func, init_estimate=init_params)
    return fun, jac, init_params, update_fun












def pca_fit(mesh: Mesh, pca_weight_matrix, mean_shape,
            rib_data, rib_data_normal,
            skin_data, skin_data_normal,
            rib_range, skin_range,
            # left_nipple = None,
            # right_nipple = None,
            # nipple_weight = 5,
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

        # coords = fit_params.reshape((-1, 3))
        # extra_costs = []
        # if left_nipple is not None:
        #     extra_costs.append(nipple_weight * (coords[163] - left_nipple))
        # if right_nipple is not None:
        #     extra_costs.append(nipple_weight * (coords[210] - right_nipple))
        #
        # if extra_costs:
        #     extra_costs = jnp.concatenate(extra_costs)
        #     return jnp.concatenate((rib_dif, skin_dif, extra_costs))
        # else:
        #     return jnp.concatenate((rib_dif, skin_dif))

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

