"""
io.py – JSON-based serialisation for HOMER :class:`~HOMER.mesher.Mesh` objects.

Meshes are stored as structured JSON files that capture all node locations,
derivative fields, element node-lists, and basis function types.  The JSON
format is human-readable and version-independent as long as the basis class
names remain stable.

Main public API:

* :func:`save_mesh` – write a mesh to a ``.json`` file.
* :func:`load_mesh` – read a mesh from a ``.json`` file.
* :func:`dump_mesh_to_dict` – serialise to a plain Python dict.
* :func:`parse_mesh_from_dict` – deserialise from a plain Python dict.

Example round-trip::

    from HOMER.io import save_mesh, load_mesh

    save_mesh(mesh, 'my_mesh.json')
    mesh2 = load_mesh('my_mesh.json')
"""

from os import PathLike
from HOMER.mesher import Mesh, MeshNode, MeshElement
from HOMER.basis_definitions import L1Basis, L2Basis, L3Basis, L4Basis, H3Basis

from pathlib import Path
import json
import numpy as np

#how do we io these files
STR_LOOKUP = {str(k.__name__):k for k in [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis]}

def dump_mesh_to_dict(obj_mesh:Mesh):
    """Serialise a :class:`~HOMER.mesher.Mesh` to a plain Python dictionary.

    The resulting dict has two top-level keys:

    * ``'nodes'`` – ordered dict of node definitions, each with ``'loc'``
      and any derivative arrays (``'du'``, ``'dv'``, …).
    * ``'elements'`` – ordered dict of element definitions, each with
      ``'nodes'`` (list of node indexes/ids), ``'basis'`` (list of basis
      class name strings), and ``'used_index'`` (bool).

    Parameters
    ----------
    obj_mesh:
        The mesh to serialise.

    Returns
    -------
    dict
        JSON-serialisable dictionary representation of the mesh.
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
        nodes_sanitised = [n if not isinstance(n, (np.int64, np.int32)) else int(n) for n in element.nodes]
        ele_def = {"nodes":nodes_sanitised}
        ele_def['basis'] = [str(b.__name__) for b in element.basis_functions]
        ele_def['used_index']= element.used_index
        elements[ide] = ele_def
    dict_rep["elements"] = elements
    return dict_rep

def parse_mesh_from_dict(dict_rep:dict) -> Mesh:
    """Deserialise a :class:`~HOMER.mesher.Mesh` from a plain Python dictionary.

    Reconstructs nodes (with all derivative arrays), elements (looking up
    basis classes by name), and calls :meth:`~HOMER.mesher.MeshField.generate_mesh`
    before returning.

    Parameters
    ----------
    dict_rep:
        Dictionary in the format produced by :func:`dump_mesh_to_dict`.

    Returns
    -------
    Mesh
        A fully initialised :class:`~HOMER.mesher.Mesh` object.
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
            ), generate_mesh=False)
        else:
            obj_mesh.add_element(MeshElement(
                node_ids=elem_def['nodes'], 
                basis_functions=[STR_LOOKUP[k] for k in elem_def['basis']],
            ), generate_mesh=False)
    obj_mesh.generate_mesh()
    return obj_mesh

def save_mesh(obj_mesh:Mesh, file_location: PathLike):
    """Serialise a mesh to a JSON file.

    Parameters
    ----------
    obj_mesh:
        The :class:`~HOMER.mesher.Mesh` to save.
    file_location:
        Destination path.  A ``.json`` extension is recommended.
    """
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    dict_rep = dump_mesh_to_dict(obj_mesh)

    with open(file_location, "w") as f:
        json.dump(dict_rep, fp=f, indent=4)
    return

def load_mesh(file_location:PathLike) -> Mesh:
    """Load a mesh from a JSON file produced by :func:`save_mesh`.

    Parameters
    ----------
    file_location:
        Path to the ``.json`` mesh file.

    Returns
    -------
    Mesh
        A fully initialised :class:`~HOMER.mesher.Mesh` object.
    """
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    with open(file_location, 'r') as f:
        dict_rep = json.load(f)
    return parse_mesh_from_dict(dict_rep)

