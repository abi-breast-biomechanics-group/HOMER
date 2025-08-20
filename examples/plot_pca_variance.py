import os
import numpy as np
import matplotlib.pyplot as plt
from HOMER.io import *
import pca
import morphic

def plot_pca_explained_variance_Homer(pca_path):
    files = [f for f in os.listdir(pca_path) if f.endswith(".json")]
    data_matrix = []

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

    data_matrix = np.array(data_matrix)

    output_pca = pca.find_principal_components(data_matrix)

    if hasattr(output_pca, "explained_variance_ratio_"):
        explained_variance_ratio = output_pca.explained_variance_ratio_
    elif hasattr(output_pca, "explained_variance_"):
        var = output_pca.explained_variance_
        explained_variance_ratio = var / np.sum(var)
    else:
        raise AttributeError("PCA object does not have explained_variance_ratio_ or explained_variance_.")

    top_n = min(10, len(explained_variance_ratio))
    explained_variance_ratio = explained_variance_ratio[:top_n]

    modes = np.arange(1, top_n + 1)
    plt.figure(figsize=(8, 5))
    plt.bar(modes, explained_variance_ratio * 100)
    plt.xlabel('Mode')
    plt.ylabel('Explained Variance (%)')
    plt.title(f'Explained Variance per Mode (Top {top_n})')
    plt.xticks(modes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pca_explained_variance_mesh(pca_path, template_path):

    template_mesh = morphic.Mesh(template_path)
    files = os.listdir(pca_path)

    data_matrix = []
    ids_done = []

    for file in files:
        skin_path = os.path.join(pca_path, file)
        if os.path.exists(skin_path):
            mesh = morphic.Mesh(skin_path)
            pars = []


            offset = mesh.nodes[7].values[:, 0] - template_mesh.nodes[7].values[:, 0]
            for node in mesh.nodes:
                node.values[:, 0] = node.values[:, 0] - offset
                pars.extend(node.values.flatten())

            data_matrix.append(pars)
            ids_done.append(file)

    data_matrix = np.array(data_matrix)

    output_pca = pca.find_principal_components(data_matrix)

    if hasattr(output_pca, "explained_variance_ratio_"):
        explained_variance_ratio = output_pca.explained_variance_ratio_
    elif hasattr(output_pca, "explained_variance_"):
        var = output_pca.explained_variance_
        explained_variance_ratio = var / np.sum(var)
    else:
        raise AttributeError("PCA object does not have explained_variance_ratio_ or explained_variance_.")

    top_n = min(10, len(explained_variance_ratio))
    explained_variance_ratio = explained_variance_ratio[:top_n]

    modes = np.arange(1, top_n + 1)
    plt.figure(figsize=(8, 5))
    plt.bar(modes, explained_variance_ratio * 100)
    plt.xlabel('Mode')
    plt.ylabel('Explained Variance (%)')
    plt.title(f'Explained Variance per Mode (Top {top_n})')
    plt.xticks(modes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_pca_explained_variance_Homer(r"Y:\sandbox\afu254\mesh_pca\clinical\unrefined")
    # pca_path = r"Y:\sandbox\afu254\mesh_pca\vl"
    # plot_pca_explained_variance_mesh(pca_path, template_path=r"Y:\sandbox\afu254\mesh_pca\vl\VL00020_prone_combined.mesh")