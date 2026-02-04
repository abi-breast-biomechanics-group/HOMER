from os import PathLike
from HOMER.mesher import Mesh, MeshNode, MeshElement
from HOMER.basis_definitions import L1Basis, L2Basis, L3Basis, L4Basis, H3Basis

from pathlib import Path
import json
import numpy as np

#how do we io these files
STR_LOOKUP = {str(k.__name__):k for k in [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis]}

def dump_mesh_to_dict(obj_mesh:Mesh):
    """
    Takes an input mesh, and returns a dict structure representing the information about that mesh object.
    """
    dict_rep = {}
    nodes = {}
    for idn, node in enumerate(obj_mesh.nodes):
        node_def = {"loc": node.loc.tolist()}
        node_def.update({k:v.tolist() for k,v in node.items()})
        if node.id is not None:
            node_def['id'] = node.id
        nodes[idn] = node_def
    dict_rep["nodes"] = nodes

    elements = {}
    for ide, element in enumerate(obj_mesh.elements):
        nodes_sanitised = [n if not isinstance(n, np.int64) else int(n) for n in element.nodes]
        ele_def = {"nodes":nodes_sanitised}
        ele_def['basis'] = [str(b.__name__) for b in element.basis_functions]
        ele_def['used_index']= element.used_index
        elements[ide] = ele_def
    dict_rep["elements"] = elements
    return dict_rep

def parse_mesh_from_dict(dict_rep:dict) -> Mesh:
    """
    Parses a dict representing a mesh to a new mesh object
    """
    obj_mesh = Mesh()

    node_dict = dict_rep['nodes']
    for node_def in node_dict.values():
        loc = node_def.pop('loc')
        node_id = node_def.pop('id', None)
        obj_mesh.add_node(
            MeshNode(loc, **{k:np.array(v) for k, v in node_def.items()}, id=node_id))

    elem_dict = dict_rep['elements']
    for elem_def in elem_dict.values():
        if elem_def.get('used_index', True):
            obj_mesh.add_element(MeshElement(
                node_indexes=elem_def['nodes'], 
                basis_functions=[STR_LOOKUP[k] for k in elem_def['basis']],
            ))
        else:
            obj_mesh.add_element(MeshElement(
                node_ids=elem_def['nodes'], 
                basis_functions=[STR_LOOKUP[k] for k in elem_def['basis']],
            ))
    obj_mesh.generate_mesh()
    return obj_mesh

def save_mesh(obj_mesh:Mesh, file_location: PathLike):
    """
    Writes a given mesh to a .json file.
    """
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    dict_rep = dump_mesh_to_dict(obj_mesh)

    with open(file_location, "w") as f:
        json.dump(dict_rep, fp=f, indent=4)
    return

def load_mesh(file_location:PathLike) -> Mesh:
    """
    loads a mesh from a given .json structured file.
    """
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    with open(file_location, 'r') as f:
        dict_rep = json.load(f)
    return parse_mesh_from_dict(dict_rep)

