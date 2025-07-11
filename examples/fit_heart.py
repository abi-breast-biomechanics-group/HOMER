from pathlib import Path
import pyvista as pv
import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares
from tqdm import tqdm

from functools import partial

from matplotlib import pyplot as plt


from HOMER import Mesh
from HOMER.basis_definitions import H3Basis
from HOMER.optim import jax_comp_kdtree_distance_query
# from HOMER.compat_functions.load_exelem import load_mesh
from HOMER.io import load_mesh, save_mesh
from HOMER.jacobian_evaluator import jacobian


# load in the heart mesh data
def load_heart_points(floc: Path):
    data = np.genfromtxt(floc, skip_header=1)[:, 1:4]
    return data


def geometric_fit_heart_to_data(mesh_obj:Mesh, inner_wall, outer_wall, plot=False):
    iw_tree = jax_comp_kdtree_distance_query(inner_wall)
    ow_tree = jax_comp_kdtree_distance_query(outer_wall)

    face_data = mesh_obj.xi_grid(dim=3, surface=True, res=20).reshape(6, -1, 3)
    ow_xis = face_data[-1]
    iw_xis = face_data[-2]


    gauss_points, weights = mesh_obj.gauss_grid([4,4,4])

    if plot:
        s = pv.Plotter()
        mesh_obj.plot(s)
        s.add_mesh(pv.PolyData(inner_wall), render_points_as_spheres=True, color='r', point_size=10)
        s.add_mesh(pv.PolyData(outer_wall), render_points_as_spheres=True, color='g', point_size=10)
        s.add_axes_at_origin()
        s.show()

    # fix all of the z derivatives of the mesh as zero
    base_param_array = jnp.array(mesh_obj.true_param_array.copy())
    # identify all of the node parameters.
    node_loc_inds = np.array(mesh_obj.associated_node_index(['loc'])).flatten().astype(int)
    node_deriv_inds = np.array(mesh_obj.associated_node_index(['du', 'dv', 'dudv']))
    node_non_x_deriv_inds = node_deriv_inds[..., 1:].flatten().astype(int) 
    n_delta_params = node_non_x_deriv_inds.shape[0]
    base_node_locs = base_param_array[node_loc_inds].reshape(-1, 3)
    init_yz_mag = np.linalg.norm(base_node_locs[:, 1:], axis=1)
    yz_norm = base_node_locs[:, 1:]/init_yz_mag[:, None]
    yz_norm_inds = node_loc_inds.reshape(-1,3)[:,1:].flatten().astype(int)
    n_locs = yz_norm.shape[0]

    init_params = np.concatenate((init_yz_mag, base_param_array[node_non_x_deriv_inds]))

    w = 0.001

    def fitting_function_params(params):
        lparam = base_param_array.copy()
        # first thing to do is create the param array 
        loc_params = params[:n_locs]
        delta_params = params[n_locs:] 

        lparam = lparam.at[yz_norm_inds].set(
            (loc_params[:, None] * yz_norm).ravel()
        )
        lparam = lparam.at[node_non_x_deriv_inds].set(delta_params)

        inner_points = mesh_obj.evaluate_embeddings([0,1,2,3,4,5,6,7], iw_xis, fit_params=lparam)
        outer_points = mesh_obj.evaluate_embeddings([0,1,2,3,4,5,6,7], ow_xis, fit_params=lparam)


        #woo smoothing

        gp_100 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [1,0,0], fit_params=lparam).flatten() * w
        gp_010 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [0,1,0], fit_params=lparam).flatten() * w
        gp_001 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [0,0,1], fit_params=lparam).flatten() * w
        gp_110 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [1,1,0], fit_params=lparam).flatten() * w
        gp_011 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [0,1,1], fit_params=lparam).flatten() * w
        gp_101 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [1,0,1], fit_params=lparam).flatten() * w
        gp_111 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [1,1,1], fit_params=lparam).flatten() * w
        gp_200 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [2,0,0], fit_params=lparam).flatten() * w
        gp_020 = mesh_obj.evaluate_deriv_embeddings([0,1,2,3,4,5,6,7], gauss_points, [0,2,0], fit_params=lparam).flatten() * w

        iw_dists = iw_tree(inner_points)
        ow_dists = ow_tree(outer_points)
        return jnp.concatenate((iw_dists, ow_dists, gp_100, gp_010, gp_001, gp_110, gp_011, gp_101, gp_111, gp_200, gp_020))

    function, jac = jacobian(fitting_function_params, init_estimate=init_params)


    def update_from_params(mesh_instance, params):
        loc_params = params[:n_locs]
        delta_params = params[n_locs:] 

        lparam = base_param_array.copy()
        lparam = lparam.at[yz_norm_inds].set(
            (loc_params[:, None] * yz_norm).ravel()
        )
        lparam = lparam.at[node_non_x_deriv_inds].set(delta_params)
        mesh_instance.update_from_params(lparam)

    return function, jac, init_params, update_from_params


def FFD_heart(mesh_obj:Mesh, start_points, end_points):

    elem, xis = mesh_obj.embed_points(start_points)
    unique_elem, inv = np.unique_inverse(elem)
    pre_sob = mesh_obj.evaluate_sobolev()

    def ffd(params, sob_w):
        out_data = []

        for ide, e in enumerate(unique_elem):
            mask = ide == inv
            dist = end_points[mask] - mesh_obj.evaluate_embeddings([e], xis[mask], fit_params=params)
            out_data.append(dist.flatten())
        out_data = jnp.concatenate(out_data)

        sob_dif = (mesh_obj.evaluate_sobolev(fit_params=params) - pre_sob) * sob_w
        return jnp.concatenate((out_data, sob_dif))

    init_params = mesh_obj.optimisable_param_array

    func, jac = jacobian(ffd, init_estimate=init_params, further_args={"sob_w":0.5})


    return func, jac, init_params, (elem, xis)

def fit_heart(func, jac, init_params, sob_w, verbose=2):
    optim = least_squares(partial(func, sob_w=sob_w), init_params, jac=partial(jac, sob_w=sob_w), verbose=verbose, max_nfev=50)
    mesh_obj.update_from_params(optim.x)
    return optim.fun



if __name__ == "__main__":
    inner_wall = load_heart_points('bin/endo.ipdata')
    outer_wall = load_heart_points('bin/epi.ipdata')


    if not (mloc := Path('bin/fitted_mesh.json')).exists():
        mesh_obj = load_mesh('bin/heart_default.json')

        s = pv.Plotter()
        mesh_obj.plot(s, node_colour='b', node_size=20, res=5)
        s.add_mesh(pv.PolyData(inner_wall), render_points_as_spheres=True, color='r', point_size=10)
        s.add_mesh(pv.PolyData(outer_wall), render_points_as_spheres=True, color='g', point_size=10)
        s.show()

        func, jac, init_params, update_fun = geometric_fit_heart_to_data(mesh_obj, inner_wall, outer_wall)


        optim = least_squares(func, init_params, jac=jac, verbose=2, max_nfev=50)
        update_fun(mesh_obj, optim.x)

        save_mesh(mesh_obj, 'bin/fitted_mesh.json')
    else: 
        mesh_obj = load_mesh(mloc)



    # mesh_obj.plot(s, node_colour='b', node_size=20)
    # s.add_mesh(pv.PolyData(inner_wall), render_points_as_spheres=True, color='r', point_size=10)
    # s.add_mesh(pv.PolyData(outer_wall), render_points_as_spheres=True, color='g', point_size=10)
    #
    start_points = load_heart_points('bin/landmark_points.ipdata')
    end_points = load_heart_points('bin/target_points.ipdata')

    s = pv.Plotter()
    mesh_obj.plot(s, mesh_color='blue', mesh_opacity=0.05)

    ws = 2**(-np.arange(-5, 10).astype(float))
    ws = [0.1]

    f, j, init_params, embed_ele_xis = FFD_heart(mesh_obj, start_points, end_points)

    unreg_sum = []
    for w in tqdm(ws, desc="L curve search"):
        res = fit_heart(f,j, init_params, w, verbose=0)
        unreg_sum.append(np.sum(np.abs(res[:(3 * start_points.shape[0])])))
    
    if len(unreg_sum) > 1:
        plt.loglog(ws, unreg_sum)
        plt.show()

    mesh_obj.plot(s, mesh_color='gray', mesh_opacity=0.02)
    
    s.add_mesh(pv.PolyData(start_points), render_points_as_spheres=True, color='orange', point_size=5)
    s.add_mesh(pv.PolyData(end_points), render_points_as_spheres=True, color='purple', point_size=5)
    
    s.show()


    wpts = mesh_obj.evaluate_ele_xi_pair_embeddings(eles=embed_ele_xis[0], xis=embed_ele_xis[1])
    deltas = end_points - wpts
    s = pv.Plotter()
    mesh_obj.plot(s, mesh_opacity=0.02)
    s.add_arrows(wpts, deltas, mag=10, cmap="turbo")
    s.show()

0

    # save_mesh(mesh_obj, 'bin/ffd_mesh.json')
