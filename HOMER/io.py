from HOMER.mesher import mesh, mesh_node, mesh_element
from HOMER.basis_definitions import L1Basis, L2Basis, L3Basis, L4Basis, H3Basis

from pathlib import Path
import json
import numpy as np

#how do we io these files
STR_LOOKUP = {str(k.__name__):k for k in [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis]}


def save_mesh(obj_mesh:mesh, file_location: Path):
    if not isinstance(file_location, Path):
        file_location = Path(file_location)

    dict_rep = {}
    nodes = {}
    for idn, node in enumerate(obj_mesh.nodes):
        node_def = {"loc": node.loc.tolist()}
        node_def.update({k:v.tolist() for k,v in node.items()})
        nodes[idn] = node_def
    dict_rep["nodes"] = nodes

    elements = {}
    for ide, element in enumerate(obj_mesh.elements):
        ele_def = {"nodes": element.nodes}
        ele_def['basis'] = [str(b.__name__) for b in element.basis_functions]
        elements[ide] = ele_def
    dict_rep["elements"] = elements

    with open(file_location, "w") as f:
        json.dump(dict_rep, fp=f, indent=4)
    
    return

def load_mesh(file_location: Path) -> mesh:
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    with open(file_location, 'r') as f:
        dict_rep = json.load(f)

    obj_mesh = mesh()

    node_dict = dict_rep['nodes']
    for node_def in node_dict.values():
        loc = node_def.pop('loc')
        obj_mesh.add_node(mesh_node(loc, **{k:np.array(v) for k, v in node_def.items()}))

    elem_dict = dict_rep['elements']
    for elem_def in elem_dict.values():
        obj_mesh.add_element(mesh_element(
            elem_def['nodes'], 
            [STR_LOOKUP[k] for k in elem_def['basis']],
        ))
    obj_mesh.generate_mesh()
    return obj_mesh
