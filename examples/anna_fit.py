import numpy as np

import morphic
from HOMER.compat_functions.convert_morphic import *
from HOMER.io import *
from HOMER.basis_definitions import H3Basis
import os
import pyvista as pv
from fit_breast import pca_fit, normal_fit, dep_node_normal_fit

from scipy.optimize import least_squares


if __name__ == '__main__':

    ######################### SETUP
    mesh = load_mesh('bin/anna/test.json')
    skin_data_path = "bin/anna/Breast_MRI_500_skin.ply"
    rib_data_path = "bin/anna/Breast_MRI_500_rib.ply"
    pca_mean = np.load("bin/anna/pca_mean.npy")
    homer_args = pca_mean.reshape(-1, 3, 4).transpose(0,2,1).flatten()
    pca_weight_matrix = np.load('bin/anna/pca_matrix.npy')
    pca_weight_matrix = pca_weight_matrix.reshape(
        pca_weight_matrix.shape[0], -1, 3, 4
    ).transpose(0, 1, 3, 2).reshape(pca_weight_matrix.shape[0], -1)

    #STERNUM
    mesh.update_from_params(homer_args)
    sternum = np.array([0.84222253, 62.50341711, 92.86663256])

    #DEPENDENT NODES: format node id, elem, coord in elem
    depnodes = np.array([
            [137, 91, 1, 2/8 * 0],
            [398, 91, 1, 2/8 * 1],
            [138, 91, 1, 2/8 * 2],
            [399, 91, 1, 2/8 * 3],
            [139, 91, 1, 2/8 * 4],
            [400, 94, 1, 2/8 * 5 - 1],
            [140, 94, 1, 2/8 * 6 - 1],
            [401, 94, 1, 2/8 * 7 - 1],
            [141, 94, 1, 2/8 * 8 - 1],
            # other side
            [72,  37, 0, 2/8 * 0],
            [309, 37, 0, 2/8 * 1],
            [73,  37, 0, 2/8 * 2],
            [311, 37, 0, 2/8 * 3],
            [74,  37, 0, 2/8 * 4],
            [313, 40, 0, 2/8 * 5 - 1],
            [75,  40, 0, 2/8 * 6 - 1],
            [315, 40, 0, 2/8 * 7 - 1],
            [76,  40, 0, 2/8 * 8 - 1],
    ])

    seventh_node_loc = mesh.nodes[7].loc
    for node in mesh.nodes:
        node.loc = node.loc + sternum - seventh_node_loc

    # update mesh
    mesh.generate_mesh()

    # mesh.plot(labels=True, 
    #           # elem_labels=True,
    #           node_size=0, res=3)

    aligned_pca_mean = mesh.optimisable_param_array

    r_mesh_data = pv.read(rib_data_path)
    s_mesh_data = pv.read(skin_data_path)

    data = np.concatenate((r_mesh_data.points, s_mesh_data.points))
    data_normal = np.concatenate((r_mesh_data.point_normals, s_mesh_data.point_normals))


    ###################################### JOHN STYLE COOKING
    func, jac, init, ufun = pca_fit( #problem with just using 0ne node lol
        mesh, pca_weight_matrix[:3], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range = np.arange(348, 372),
        skin_range = np.arange(0, 348),
        res=6,
    )

    max_sd = 2
    bounds = (-max_sd * np.ones_like(init), max_sd * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2,
                          bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    func, jac, init, ufun = pca_fit(
        mesh, pca_weight_matrix[:6], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range = np.arange(348, 372),
        skin_range = np.arange(0, 348),
        res=6,
    )

    max_sd = 2
    bounds = (-max_sd * np.ones_like(init), max_sd * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2,
                          bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    func, jac, init, ufun = pca_fit(
        mesh, pca_weight_matrix[:9], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range = np.arange(348, 372),
        skin_range = np.arange(0, 348),
        res=6,
    )

    max_sd = 2
    bounds = (-max_sd * np.ones_like(init), max_sd * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2,
                          bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)
    # s = pv.Plotter()
    # mesh.plot(s)
    # s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    # s.add_mesh(s_mesh_data, opacity=0.5)
    # s.show()

    # raise ValueError()

    func, jac, init, ufun = dep_node_normal_fit(
        mesh,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        dep_nodes = depnodes, div_scalar=1/4,
        rib_range = np.arange(348, 372),
        skin_range = np.arange(0, 348),
        res=6,
        w=10,
    )

    optim = least_squares(func, init, jac=jac, verbose=2,
                          # bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    s = pv.Plotter()
    mesh.plot(s, res=3, labels=False)
    s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    s.add_mesh(s_mesh_data, opacity=0.5)
    s.show()

