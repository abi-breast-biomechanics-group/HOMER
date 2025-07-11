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
import pca

def get_pca_mean_and_matrix(pca_path, template_path):
    # path of meshes to run PCA
    # path = r"Y:\sandbox\fpan017\meshes\new_workflow\shape_model\combined\with_prior\gen3"
    # a template mesh
    # template_path = r"Y:\sandbox\fpan017\meshes\new_workflow\shape_model\combined\with_prior\gen3\VL00024_prone_combined.mesh"
    template_mesh = morphic.Mesh(template_path)
    files = os.listdir(pca_path)
    # bad_cases = []  # in case if there are bad cases that need to be excluded
    # bad_cases_id = ['VL' + '{:0>5}'.format(i) + "_prone_combined.mesh" for i in bad_cases]
    data_matrix = []
    ids_done = []
    for file in files:
        skin_path = os.path.join(pca_path, file)
        # if file not in bad_cases_id:
        if os.path.exists(skin_path):
                mesh = morphic.Mesh(skin_path)
                pars = []
                # note that all meshes are aligned using the spatrial coordinates of node 7
                offset = mesh.nodes[7].values[:, 0] - template_mesh.nodes[7].values[:, 0]
                for node in mesh.nodes:
                    node.values[:, 0] = node.values[:, 0] - offset
                    pars.extend(node.values.flatten())
                data_matrix.append(pars)
                ids_done.append(file)
        # else:
        #     print(file + ": dropped")

    # the data matrix for PCA
    data_matrix = np.array(data_matrix)
    output_pca = pca.find_principal_components(data_matrix)

    return output_pca.components_, output_pca.mean_

def txt_to_ply():
    pass
if __name__ == '__main__':

    # template_skin_path = r"Y:\sandbox\afu254\volunteer_skin_mesh\VL00089_prone.mesh"
    # template_rib_path = r"Y:\sandbox\fpan017\John_datasets\manual_fitted_mesh_rai\meshes\rib_cage\gen3\VL00089_ribcage_prone.mesh"

    # run only for the first time (there's no template
    # template_skin_mesh = morphic.Mesh(template_skin_path)
    # homer_skin_mesh = convert_morphic(template_skin_mesh, basis_functions=(H3Basis,H3Basis))
    # save_mesh(homer_skin_mesh, r"skin_temp.json")
    # skin_mesh = load_mesh('skin_temp.json')
    # skin_mesh.plot()

    # template_rib_mesh = morphic.Mesh(template_rib_path)
    # homer_rib_mesh = convert_morphic(template_rib_mesh, basis_functions=(H3Basis, H3Basis))
    # save_mesh(homer_rib_mesh, r"rib_temp.json")
    # rib_mesh = load_mesh('rib_temp.json')
    # rib_mesh.plot()

    mesh = load_mesh('test.json')

    # skin_data_path = r"Y:\sandbox\afu254\Duke\points\txt\skin\Breast_MRI_500_skin_pts.txt"
    # xyz = np.genfromtxt(skin_data_path, delimiter=" ", skip_header=False)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # path2 = r"Y:\sandbox\afu254\Duke\points\ply\skin"
    # o3d.io.write_point_cloud(os.path.join(path2, "Breast_MRI_500_skin.ply"), pcd, format="ply")

    skin_data_path = r"Y:\sandbox\afu254\Duke\points\ply\skin\Breast_MRI_500_skin.ply"
    # run when only has txt no ply
    # rib_data_path = r"Y:\sandbox\afu254\Duke\points\txt\rib\Breast_MRI_754_rib_pts.txt"
    #
    # xyz = np.genfromtxt(rib_data_path, delimiter=" ", skip_header=False)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # path2 = r"Y:\sandbox\afu254\Duke\points\ply\rib"
    # o3d.io.write_point_cloud(os.path.join(path2, "Breast_MRI_754_rib.ply"), pcd, format="ply")

    rib_data_path = r"Y:\sandbox\afu254\Duke\points\ply\rib\Breast_MRI_500_rib.ply"

    # skin_pca_path = r"Y:\sandbox\fpan017\meshes\new_workflow\shape_model\combined\with_prior\gen3"
    # skin_pca_matrix, skin_pca_mean = get_pca_mean_and_matrix(skin_pca_path, template_skin_path)

    pca_mean = np.load("pca_mean.npy")
    homer_args = pca_mean.reshape(-1, 3, 4).transpose(0,2,1).flatten()
    pca_weight_matrix = np.load('pca_matrix.npy')
    pca_weight_matrix = pca_weight_matrix.reshape(
        pca_weight_matrix.shape[0], -1, 3, 4
    ).transpose(0, 1, 3, 2).reshape(pca_weight_matrix.shape[0], -1)

    mesh.update_from_params(homer_args)

    # alignment using sternum landmark

    # Duke
    sternum = np.array([0.84222253, 62.50341711, 92.86663256])

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
        rib_range=np.arange(348, 372),
        skin_range=np.arange(0, 348),
        res=20
    )

    bounds = (-3 * np.ones_like(init), 3 * np.ones_like(init))
    # limited to max 50
    optim = least_squares(func, init, jac=jac, verbose=2, bounds=bounds, max_nfev=50)
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
        rib_range=np.arange(348, 372),
        skin_range=np.arange(0, 348),
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

    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_500.json")



