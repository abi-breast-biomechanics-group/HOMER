import numpy as np
import pyvista as pv

from morphic import Mesh as mMesh
from HOMER import Mesh, MeshElement, MeshNode
from HOMER.io import load_mesh, save_mesh


from HOMER.basis_definitions import H3Basis


def convert_morphic(morphic_mesh:mMesh, basis_functions, plot=False):

    nodes = []
    used_ids = []
    for node in morphic_mesh.nodes:
        #create based on the spec
        if node.id in used_ids:
            continue
        if str(node.id) in used_ids:
            continue
        # if isinstance(node.id, str):
        #     continue
        used_ids.append(str(node.id))

        try:
            vals = np.array(node.values).copy().T
        except:
            breakpoint()
        try:

            node_data_struct = {k:v for k, v in zip(['du', 'dv', 'dudv'], vals[1:, :])}
        except:
            # print('node with 1d data')
            continue
        nodes.append(
                MeshNode(
                    loc = vals[0, :],
                    **node_data_struct,
                    id=str(node.id)
            )
        )
        print(node.id)
   
    elements = []
    for element in morphic_mesh.elements:
        elements.append(
                MeshElement(node_ids=[str(i) for i in element.node_ids], basis_functions=basis_functions)
        )
    
    objMesh = Mesh(nodes=nodes, elements=elements)

    if plot:
        objMesh.plot()

    breakpoint()


    return objMesh

if __name__ == "__main__":
    mesh = mMesh()
    mesh.load(filepath='bin/breast/breast.mesh')
    mesh.load(filepath='bin/breast/14_fitted_skin.mesh')
    mesh.load(filepath='bin/breast/prone_original.mesh')
    mesh.load(filepath='bin/breast/prone_original.mesh')
    mesh.load(filepath='bin/breast/breast_h3h3h3.mesh')
    homer_mesh = convert_morphic(morphic_mesh=mesh, basis_functions=(H3Basis, H3Basis, H3Basis))
    homer_mesh.plot()
    # save_mesh(homer_mesh, "bin/breast/DUKE_14.json")



