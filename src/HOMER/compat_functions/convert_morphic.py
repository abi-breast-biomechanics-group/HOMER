import numpy as np
import pyvista as pv

from morphic import Mesh as mMesh
from HOMER import Mesh, MeshElement, MeshNode
from HOMER.io import load_mesh, save_mesh


from HOMER.basis_definitions import H3Basis


def convert_morphic(morphic_mesh:mMesh, basis_functions, plot=False):

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

    if plot:
        objMesh.plot()

    breakpoint()


    return objMesh

if __name__ == "__main__":
    mesh = mMesh()
    mesh.load(filepath='bin/breast/breast.mesh')
    homer_mesh = convert_morphic(morphic_mesh=mesh, basis_functions=(H3Basis, H3Basis))
    save_mesh(homer_mesh, "bin/breast/breast.json")



