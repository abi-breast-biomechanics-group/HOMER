import numpy as np

import morphic
from HOMER.compat_functions.convert_morphic import *
from HOMER.io import *
from HOMER.basis_definitions import H3Basis
import os
import open3d as o3d
import pyvista as pv
from fit_breast import pca_fit, normal_fit

from scipy.optimize import least_squares


if __name__ == '__main__':
    # template_path = r"Y:\sandbox\fpan017\meshes\new_workflow\shape_model\combined\with_prior\gen3\VL00024_prone_combined.mesh"
    # template_mesh = morphic.Mesh(template_path)
    # homer_mesh = convert_morphic(template_mesh, basis_functions=(H3Basis,H3Basis))
    # save_mesh(homer_mesh, r"test.json")
    mesh = load_mesh('test.json')

    ############################## skin / rib parts
    # skin elements= [i for i in range(348)] rib elements= [i for i in range(348, 372)]

    skin_data_path = r"Y:\sandbox\fpan017\John_datasets\segmentation_xinyue\point_cloud\skin_upsample\VL00089\skin.ply"
    # rib_data_path = r"Y:\sandbox\fpan017\John_datasets\ribcage_pct_rai\rib_cage_VL00089.nii.txt"
    #
    # xyz = np.genfromtxt(rib_data_path, delimiter=" ", skip_header=False)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # path2 = r"Y:\sandbox\fpan017\John_datasets\ribcage_pct_rai"
    # o3d.io.write_point_cloud(os.path.join(path2, "rib_cage_VL00089.nii.ply"), pcd, format="ply")

    rib_data_path = r"Y:\sandbox\fpan017\John_datasets\ribcage_pct_rai\rib_cage_VL00089.nii.ply"

    pca_mean = np.load("pca_mean.npy")
    homer_args = pca_mean.reshape(-1, 3, 4).transpose(0,2,1).flatten()
    pca_weight_matrix = np.load('pca_matrix.npy')
    pca_weight_matrix = pca_weight_matrix.reshape(
        pca_weight_matrix.shape[0], -1, 3, 4
    ).transpose(0, 1, 3, 2).reshape(pca_weight_matrix.shape[0], -1)

    mesh.update_from_params(homer_args)

    # alignment using sternum landmark
    landmark_file_path= r"Y:\sandbox\fpan017\John_datasets\landmarks\landmarks_ray\VL00089_skeleton_data_prone_t2.json"
    with open(landmark_file_path) as f:
        d = json.load(f)
        x, y, z = d.get('bodies').get('Ray-Test').get('landmarks').get('sternal-superior').get(
            '3d_position').values()
        sternum = np.array([x, y, z])

    seventh_node_loc = mesh.nodes[7].loc
    for node in mesh.nodes:
        node.loc = node.loc + sternum - seventh_node_loc

    # update mesh
    mesh.generate_mesh()

    aligned_pca_mean = mesh.optimisable_param_array

    # plot template mesh (homer), data points together to check alignment
    # s = pv.Plotter()
    # mesh.plot(s)
    # s.add_mesh(pv.read(rib_data_path), color='orange', opacity=0.5)
    # s.add_mesh(pv.read(skin_data_path), opacity=0.5)
    # s.show()

    r_mesh_data = pv.read(rib_data_path)
    s_mesh_data = pv.read(skin_data_path)

    data = np.concatenate((r_mesh_data.points, s_mesh_data.points))
    data_normal = np.concatenate((r_mesh_data.point_normals, s_mesh_data.point_normals))

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

    func, jac, init, ufun = normal_fit(
        mesh,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        rib_range = np.arange(348, 372),
        skin_range = np.arange(0, 348),
        res=6,
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

