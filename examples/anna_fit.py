import numpy as np

import morphic
from HOMER.compat_functions.convert_morphic import *
from HOMER.io import *
from HOMER.basis_definitions import H3Basis
import os
import pyvista as pv
from fit_breast import pca_fit, normal_fit, dep_node_normal_fit, dif_sob_dep_node_normal_fit

from scipy.optimize import least_squares
import open3d as o3d
def txt_to_ply(txt_path, output_path, visualisation = False):
    skin_data_path =txt_path
    xyz = np.genfromtxt(skin_data_path, delimiter=" ", skip_header=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    if visualisation:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    o3d.io.write_point_cloud(output_path, pcd, format="ply")

if __name__ == '__main__':
    with open("resources/refined_mesh_config.json", "r") as f:
        config = json.load(f)

    depnodes = np.array(config["depnodes"])
    posterior_element_ids = config["posterior_element_ids"]
    sternum_element_ids = config["sternum_element_ids"]
    edge_element_ids = config["edge_element_ids"]
    sternum_node_ids = config["sternum_node_ids"]
    inferior_group0 = config["inferior_group0"]
    superior_group1 = config["superior_group1"]
    superior_group2 = config["superior_group2"]
    rib_range = np.array(config["rib_range"])
    skin_range = np.array(config["skin_range"])

    ######################### SETUP
    mesh = load_mesh('resources/test.json')
    # skin_txt_path = r"Y:\sandbox\afu254\Duke\points\txt\skin\Breast_MRI_121_skin_pts.txt"
    # rib_txt_path = r"Y:\sandbox\afu254\Duke\points\txt\rib\Breast_MRI_121_rib_pts.txt"
    skin_data_path = r"Y:\sandbox\afu254\Duke\points\ply\skin\Breast_MRI_015_skin.ply"
    rib_data_path = r"Y:\sandbox\afu254\Duke\points\ply\rib\Breast_MRI_015_rib.ply"

    # run when there's no ply
    # txt_to_ply(skin_txt_path,skin_data_path)
    # txt_to_ply(rib_txt_path,rib_data_path)

    pca_mean = np.load("pca_mean.npy")
    homer_args = pca_mean.reshape(-1, 3, 4).transpose(0, 2, 1).flatten()
    pca_weight_matrix = np.load('pca_matrix.npy')
    pca_weight_matrix = pca_weight_matrix.reshape(
        pca_weight_matrix.shape[0], -1, 3, 4
    ).transpose(0, 1, 3, 2).reshape(pca_weight_matrix.shape[0], -1)

    # STERNUM
    mesh.update_from_params(homer_args)
    sternum = np.array([12.61973526, 55.31869016, 55.09848675])
    # Other landmarks (for Duke case 039)
    # left_nipple = [108.65693464, -124.25469544,  20.08874782]
    # right_nipple = [ -66.65557126, -134.8796958,  -0.4007652]

    high_weight_ids = posterior_element_ids + sternum_element_ids

    breast_element_ids = [i for i in range(len(skin_range)) if i not in high_weight_ids]
    # alignment
    seventh_node_loc = mesh.nodes[7].loc
    for node in mesh.nodes:
        node.loc = node.loc + sternum - seventh_node_loc

    # update mesh
    mesh.generate_mesh()
    pca_mean_mesh = mesh

    # mesh.plot(labels=True,
    #           # elem_labels=True,
    #           node_size=0, res=3)

    aligned_pca_mean = mesh.optimisable_param_array

    r_mesh_data = pv.read(rib_data_path)
    s_mesh_data = pv.read(skin_data_path)

    data = np.concatenate((r_mesh_data.points, s_mesh_data.points))
    data_normal = np.concatenate((r_mesh_data.point_normals, s_mesh_data.point_normals))

    ###################################### JOHN STYLE COOKING
    func, jac, init, ufun = pca_fit(  # problem with just using 0ne node lol
        mesh, pca_weight_matrix[:3], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range=rib_range,
        skin_range=skin_range,
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
    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_121_3mode.json")

    func, jac, init, ufun = pca_fit(
        mesh, pca_weight_matrix[:6], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range=rib_range,
        skin_range=skin_range,
        res=6,
    )

    max_sd = 2
    bounds = (-max_sd * np.ones_like(init), max_sd * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2,
                          bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_121_6mode.json")

    func, jac, init, ufun = pca_fit(
        mesh, pca_weight_matrix[:9], aligned_pca_mean,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range=rib_range,
        skin_range=skin_range,
        res=6,
    )

    max_sd = 2
    bounds = (-max_sd * np.ones_like(init), max_sd * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2,
                          bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_121_9mode.json")

    s = pv.Plotter()
    mesh.plot(s)
    s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    s.add_mesh(s_mesh_data, opacity=0.5)
    s.show()


    func, jac, init, ufun = dif_sob_dep_node_normal_fit(
        mesh,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        dep_nodes=depnodes, div_scalar=1 / 4,
        rib_range=rib_range,
        skin_range=skin_range,
        strict_range=high_weight_ids, breast_range=breast_element_ids,
        prior_mesh=pca_mean_mesh, nodes_to_fix_x=sternum_node_ids,
        nodes_to_fix_group0=inferior_group0,
        nodes_to_fix_group1=superior_group1,
        nodes_to_fix_group2=superior_group2,
        left_nipple=None,
        right_nipple=None,
        # left_nipple= left_nipple,
        # right_nipple = right_nipple,
        fixed_x=sternum[0],
        res=6,
        w_array=None
    )
    optim = least_squares(func, init, jac=jac, verbose=2,
                          # bounds=bounds,
                          max_nfev=50)
    ufun(mesh, optim.x)

    s = pv.Plotter()
    mesh.plot(s)
    s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    s.add_mesh(s_mesh_data, opacity=0.5)
    s.show()

    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_121.json")
