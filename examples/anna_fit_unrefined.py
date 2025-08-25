import numpy as np
import json
import morphic
from HOMER.compat_functions.convert_morphic import *
from HOMER.io import *
from HOMER.basis_definitions import H3Basis
import os
import open3d as o3d
import pyvista as pv
from fit_breast import *
import pandas as pd
from scipy.optimize import least_squares
import pca
import ast

def get_pca_mean_and_matrix(pca_path, template_path):
    # path of meshes to run PCA
    # a template mesh
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

def get_pca_mean_and_matrix_Homer(pca_path):
    files = [f for f in os.listdir(pca_path) if f.endswith(".json")]
    data_matrix = []
    ids_done = []

    # Load template (first mesh in folder)
    template_path = os.path.join(pca_path, files[0])
    template_mesh = load_mesh(template_path)
    template_node7 = template_mesh.nodes[7].loc.copy()

    for file in files:
        mesh_path = os.path.join(pca_path, file)
        mesh = load_mesh(mesh_path)

        # Align to template's node 7
        node7 = mesh.nodes[7].loc.copy()
        translation = template_node7 - node7
        for node in mesh.nodes:
            node.loc += translation

        mesh.generate_mesh()

        # Extract parameter array for PCA
        param_array = mesh.true_param_array
        data_matrix.append(param_array)
        ids_done.append(file)

    data_matrix = np.array(data_matrix)
    output_pca = pca.find_principal_components(data_matrix)
    # mesh.plot()
    return output_pca.components_, output_pca.mean_
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
    landmark_excel = r"Y:\sandbox\afu254\clinical_workflow_cohort\clinical_landmark.xlsx"
    landmark_df = pd.read_excel(landmark_excel)
    # pca_path = r"Y:\sandbox\afu254\mesh_pca\vl"
    # unrefined_pca_matrix, unrefined_pca_mean = get_pca_mean_and_matrix(pca_path, template_path=r"Y:\sandbox\afu254\mesh_pca\vl\VL00020_prone_combined.mesh")
    # np.save('unrefined_pca_matrix.npy',unrefined_pca_matrix)
    # np.save('unrefined_pca_mean.npy',unrefined_pca_mean)

    # clinical_pca_path = r"Y:\sandbox\afu254\mesh_pca\clinical\unrefined\v2"
    # unrefined_pca_matrix, unrefined_pca_mean = get_pca_mean_and_matrix_Homer(clinical_pca_path)
    # np.save('resources/cl_unrefined_pca_matrix_v2.npy', unrefined_pca_matrix)
    # np.save('resources/cl_unrefined_pca_mean_v2.npy', unrefined_pca_mean)

    with open("resources/unrefined_mesh_config.json", "r") as f:
        config = json.load(f)

    depnodes = np.array(config["depnodes"])
    posterior_element_ids = config["posterior_element_ids"]
    sternum_element_ids = config["sternum_element_ids"]
    edge_element_ids = config["edge_element_ids"]
    sternum_node_ids = config["sternum_node_ids"]
    superior_inferior_nodes = config["superior_inferior_nodes"]
    inferior_group0 = config["inferior_group0"]
    superior_group1 = config["superior_group1"]
    superior_group2 = config["superior_group2"]
    rib_range = np.array(config["rib_range"])
    skin_range = np.array(config["skin_range"])

    ######################### SETUP
    mesh = load_mesh("resources/unrefined_mesh_template.json")
    ID = "754"
    match_id = f"Breast_MRI_{ID}"

    row = landmark_df[landmark_df['ID'] == match_id]
    landmark_df['sternum'] = landmark_df['sternum'].astype(str).str.strip()
    if not row.empty:
        sternum_str = row.iloc[0]['sternum']
        sternum_str = sternum_str.replace('\xa0', ' ')
        sternum_list = ast.literal_eval(sternum_str.replace('\n', '').replace('\r', '').strip())
        sternum = np.array(sternum_list)
    else:
        print(f"ID {match_id} not found")
    # skin_txt_path = r"Y:\sandbox\afu254\Duke\points\txt\skin\Breast_MRI_" +ID + "_skin_pts.txt"
    # rib_txt_path = r"Y:\sandbox\afu254\Duke\points\txt\rib\Breast_MRI_" + ID + "_rib_pts.txt"
    skin_data_path = r"Y:\sandbox\afu254\Duke\points\ply\skin\Breast_MRI_" + ID + "_skin.ply"
    rib_data_path = r"Y:\sandbox\afu254\Duke\points\ply\rib\Breast_MRI_" + ID + "_rib.ply"

    # skin_txt_path = r"Y:\sandbox\afu254\EA\points\txt\skin\Breast_MRI_" + ID + "_skin_pts.txt"
    # rib_txt_path = r"Y:\sandbox\afu254\EA\points\txt\rib\Breast_MRI_" + ID + "_rib_pts.txt"
    # skin_data_path = r"Y:\sandbox\afu254\EA\points\ply\Breast_MRI_" + ID + "_skin.ply"
    # rib_data_path = r"Y:\sandbox\afu254\EA\points\ply\Breast_MRI_" + ID + "_rib.ply"

    # run when there's no ply
    # txt_to_ply(skin_txt_path,skin_data_path)
    # txt_to_ply(rib_txt_path,rib_data_path)

    # pca_mean = np.load("resources/cl_unrefined_pca_mean_v2.npy")
    # homer_args = pca_mean.flatten()
    # pca_weight_matrix = np.load('resources/cl_unrefined_pca_matrix_v2.npy')

    # if use volunteer pca
    pca_mean = np.load("resources/unrefined_pca_mean.npy")
    homer_args = pca_mean.flatten().reshape(-1, 3, 4).transpose(0, 2, 1).flatten()
    pca_weight_matrix = np.load('resources/unrefined_pca_matrix.npy')
    pca_weight_matrix = pca_weight_matrix.reshape(
        pca_weight_matrix.shape[0], -1, 3, 4
    ).transpose(0, 1, 3, 2).reshape(pca_weight_matrix.shape[0], -1)

    # STERNUM
    mesh.update_from_params(homer_args)
    # mesh.plot()
    # sternum = np.array([345.85640305, 376.87679702,  8.02471103])
    # Other landmarks (for Duke case 039)
    # left_nipple = [108.65693464, -124.25469544,  20.08874782]
    # right_nipple = [ -66.65557126, -134.8796958,  -0.4007652]

    high_weight_ids = posterior_element_ids

    breast_element_ids = [i for i in range(len(skin_range)) if i not in high_weight_ids]
    # alignment
    seventh_node_loc = mesh.nodes[7].loc
    for node in mesh.nodes:
        node.loc = node.loc + sternum - seventh_node_loc

    # update mesh
    mesh.generate_mesh()
    # mesh.plot()
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
    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_" + ID "_3mode.json")

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

    # s = pv.Plotter()
    # mesh.plot(s)
    # s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    # s.add_mesh(s_mesh_data, opacity=0.5)
    # s.show()
    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_" + ID +"_6mode.json")

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

    # save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_" + ID + "_9mode.json")
    # mesh = load_mesh(r"Y:\sandbox\afu254\mesh_pca\clinical\unrefined\v2\Breast_MRI_754.json")
    s = pv.Plotter()
    mesh.plot(s)
    s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    s.add_mesh(s_mesh_data, opacity=0.5)
    s.show()

    # raise ValueError()

    func, jac, init, ufun = dif_sob_dep_node_normal_fit(
        mesh,
        r_mesh_data.points, r_mesh_data.point_normals,
        s_mesh_data.points, s_mesh_data.point_normals,
        dep_nodes=depnodes, div_scalar=1 / 2,
        rib_range=rib_range,
        skin_range=skin_range,
        strict_range=high_weight_ids, breast_range=breast_element_ids,
        prior_mesh=pca_mean_mesh, nodes_to_fix_x=sternum_node_ids,
        nodes_to_fix_group0=inferior_group0,
        nodes_to_fix_group1=superior_group1 + superior_group2,
        # nodes_to_fix_group2=superior_group2,
        superior_inferior_nodes=superior_inferior_nodes,
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
                          max_nfev=150)
    ufun(mesh, optim.x)

    s = pv.Plotter()
    mesh.plot(s)
    s.add_mesh(r_mesh_data, color='orange', opacity=0.5)
    s.add_mesh(s_mesh_data, opacity=0.5)
    s.show()

    save_mesh(mesh, r"Y:\sandbox\afu254\homer_fit_mesh\Breast_MRI_" + ID + ".json")



