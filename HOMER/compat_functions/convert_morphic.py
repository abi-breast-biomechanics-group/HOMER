import numpy as np

from morphic import Mesh as mMesh
from HOMER import Mesh, MeshElement, MeshNode
from HOMER.io import load_mesh, save_mesh


from HOMER.basis_definitions import H3Basis

import os
def convert_morphic(morphic_mesh:mMesh, basis_functions):

    nodes = []
    for node in morphic_mesh.nodes:
        #create based on the spec
        vals = np.array(node.values.T).copy()
        node_data_struct = {k:v for k, v in zip(['du', 'dv', 'dudv'], vals[1:, :])}
        nodes.append(
                MeshNode(
                    loc = vals[0, :],
                    **node_data_struct,
                    id=node.id
            )
        )
   
    elements = []
    for element in morphic_mesh.elements:
        elements.append(
                MeshElement(node_ids=element.node_ids, basis_functions=basis_functions)
        )
    
    objMesh = Mesh(nodes=nodes, elements=elements)
    # objMesh.plot()

    return objMesh

if __name__ == "__main__":
    # mesh = mMesh()
    # mesh.load(filepath=r"Y:\sandbox\fpan017\meshes\new_workflow\shape_model\combined\no_refine\VL00047_prone_combined.mesh")
    # # mesh.load(filepath=r"Y:\sandbox\fpan017\John_datasets\manual_fitted_mesh_rai\meshes\rib_cage\gen3\VL00047_ribcage_prone.mesh")
    # homer_mesh = convert_morphic(morphic_mesh=mesh, basis_functions=(H3Basis, H3Basis))
    # save_mesh(homer_mesh, "../../examples/resources/unrefined_mesh_template.json")
    root_folder = r"Y:\sandbox\afu254\mesh_pca\vl"
    output_folder = r"Y:\sandbox\afu254\volunteer_skin_mesh\unrefined"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(root_folder):
        if filename.endswith("_prone_combined.mesh") and filename.startswith("VL00"):
            mesh_path = os.path.join(root_folder, filename)

            mesh = mMesh()
            mesh.load(filepath=mesh_path)

            homer_mesh = convert_morphic(morphic_mesh=mesh, basis_functions=(H3Basis, H3Basis))

            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder, json_filename)
            save_mesh(homer_mesh, json_path)

    print("finished")




