"""
io.py – JSON-based serialisation for HOMER :class:`~HOMER.mesher.Mesh` objects.

Meshes are stored as structured JSON files that capture all node locations,
derivative fields, element node-lists, and basis function types.  The JSON
format is human-readable and version-independent as long as the basis class
names remain stable.

Schema overview:

* **Field-only** (legacy / MeshField): ``{"nodes": ..., "elements": ...}``
* **Mesh with fields**: ``{"main": <field dict>, "fields": {name: <field dict>}}``

Main public API:

* :func:`save_mesh` – write a mesh to a ``.json`` file.
* :func:`load_mesh` – read a mesh from a ``.json`` file.
* :func:`dump_mesh_to_dict` – serialise a mesh (with fields) to a dict.
* :func:`parse_mesh_from_dict` – deserialise a mesh (with fields) from a dict.
* :func:`dump_meshfield_to_dict` – serialise a single field to a dict.
* :func:`parse_meshfield_from_dict` – deserialise a single field from a dict.

Example round-trip::

    mesh.save('my_mesh.json')
    mesh2 = Mesh.load('my_mesh.json')
"""

from os import PathLike
from HOMER.mesher import Mesh, MeshNode, MeshElement, MeshField
from HOMER.basis_definitions import BASIS_REGISTRY, BasisGroup, basis_by_name

from pathlib import Path
import json
import numpy as np

#how do we io these files.  A basis serialises as its name, and the basis
#registry resolves it back, so a user-defined basis round-trips too.
STR_LOOKUP = BASIS_REGISTRY  #kept as the old name for the live registry

def dump_meshfield_to_dict(obj_field: MeshField) -> dict:
    """Serialise a :class:`~HOMER.mesher.MeshField` to a plain Python dictionary.

    The resulting dict has two top-level keys:

    * ``'nodes'`` – ordered dict of node definitions, each with ``'loc'``,
      any derivative arrays (``'du'``, ``'dv'``, …), and ``'fixed_params'``
      for nodes carrying fixed (non-optimisable) parameters.
    * ``'elements'`` – ordered dict of element definitions, each with
      ``'nodes'`` (list of node indexes/ids), ``'basis'`` (list of basis
      class name strings), and ``'used_index'`` (bool).

    Parameters
    ----------
    obj_field:
        The field to serialise.

    Returns
    -------
    dict
        JSON-serialisable dictionary representation of the field.
    """
    dict_rep = {}
    nodes = {}
    for idn, node in enumerate(obj_field.nodes):
        node_def = {"loc": node.loc.tolist()}
        node_def.update({k: v.tolist() for k, v in node.items()})
        if node.id is not None:
            node_def['id'] = node.id
        if node.fixed_params:
            node_def['fixed_params'] = {
                k: np.asarray(v).astype(int).tolist()
                for k, v in node.fixed_params.items()
            }
        nodes[idn] = node_def
    dict_rep["nodes"] = nodes

    elements = {}
    for ide, element in enumerate(obj_field.elements):
        nodes_sanitised = [n if not isinstance(n, (np.int64, np.int32)) else int(n) for n in element.nodes]
        ele_def = {"nodes": nodes_sanitised}
        ele_def['basis'] = [b.name for b in element.basis_functions]
        ele_def['used_index'] = element.used_index
        if element.id is not None:
            ele_def['id'] = element.id
        elements[ide] = ele_def
    dict_rep["elements"] = elements
    return dict_rep


def dump_mesh_to_dict(obj_mesh: Mesh | MeshField) -> dict:
    """Serialise a :class:`~HOMER.mesher.Mesh` (or MeshField) to a dictionary.

    Mesh objects are stored with a ``'main'`` field and optional named
    ``'fields'``.  Passing a :class:`MeshField` returns the legacy
    ``{'nodes', 'elements'}`` structure for compatibility.
    """
    if isinstance(obj_mesh, Mesh):
        return {
            "main": dump_meshfield_to_dict(obj_mesh),
            "fields": {name: dump_meshfield_to_dict(field) for name, field in obj_mesh.fields.items()},
        }
    return dump_meshfield_to_dict(obj_mesh)

def _rehash_id(node_id):
    """JSON has no tuples, so a tuple id round-trips as a list.

    Ids are used as dict keys, so a list would be unhashable; anything that
    was a valid id before the dump is hashable again after this.
    """
    if isinstance(node_id, list):
        return tuple(_rehash_id(x) for x in node_id)
    return node_id


def _parse_field_from_dict(dict_rep: dict, field_cls: type[MeshField]) -> MeshField:
    obj_field = field_cls()
    node_dict = dict_rep.get('nodes', {})
    for node_def in node_dict.values():
        node_def = dict(node_def)
        loc = node_def.pop('loc')
        node_id = _rehash_id(node_def.pop('id', None))
        fixed_params = node_def.pop('fixed_params', None)
        node = MeshNode(
            loc, **{k: np.array(v) for k, v in node_def.items()}, id=node_id)
        if fixed_params:
            # restored before generate_mesh, so the optimisable mask accounts for it
            node.fixed_params = {
                k: np.asarray(v).astype(int) for k, v in fixed_params.items()
            }
        obj_field.add_node(node)

    elem_dict = dict_rep.get('elements', {})
    for elem_def in elem_dict.values():
        elem_def = dict(elem_def)
        basis_functions = BasisGroup(basis_by_name(k) for k in elem_def['basis'])
        elem_id = _rehash_id(elem_def.get('id', None))
        if elem_def.get('used_index', True):
            obj_field.add_element(MeshElement(
                node_indexes=elem_def['nodes'],
                basis_functions=basis_functions,
                id=elem_id,
            ), generate_mesh=False)
        else:
            obj_field.add_element(MeshElement(
                node_ids=[_rehash_id(n) for n in elem_def['nodes']],
                basis_functions=basis_functions,
                id=elem_id,
            ), generate_mesh=False)
    if obj_field.nodes and obj_field.elements:
        obj_field.generate_mesh()
    return obj_field


def parse_meshfield_from_dict(dict_rep: dict) -> MeshField:
    """Deserialise a :class:`~HOMER.mesher.MeshField` from a dictionary."""
    return _parse_field_from_dict(dict_rep, MeshField)


def parse_mesh_from_dict(dict_rep: dict) -> Mesh:
    """Deserialise a :class:`~HOMER.mesher.Mesh` from a plain Python dictionary.

    Reconstructs nodes (with all derivative arrays), elements (looking up
    bases by name), and calls :meth:`~HOMER.mesher.MeshField.generate_mesh`
    before returning.  Accepts both the legacy ``{'nodes','elements'}`` format
    and the newer ``{'main','fields'}`` schema.
    """
    if 'main' in dict_rep or 'fields' in dict_rep:
        main_dict = dict_rep.get('main')
        if main_dict is None:
            main_dict = {k: dict_rep[k] for k in ('nodes', 'elements') if k in dict_rep}
        obj_mesh = _parse_field_from_dict(main_dict, Mesh)
        for field_name, field_dict in dict_rep.get('fields', {}).items():
            obj_mesh[field_name] = _parse_field_from_dict(field_dict, MeshField)
        return obj_mesh
    return _parse_field_from_dict(dict_rep, Mesh)

def save_mesh(obj_mesh: Mesh | MeshField, file_location: PathLike):
    """Serialise a mesh (or field) to a JSON file.

    Parameters
    ----------
    obj_mesh:
        The :class:`~HOMER.mesher.Mesh` (or :class:`MeshField`) to save.
    file_location:
        Destination path.  A ``.json`` extension is recommended.
    """
    if not isinstance(file_location, Path):
        file_location = Path(file_location)
    dict_rep = dump_mesh_to_dict(obj_mesh)

    with open(file_location, "w") as f:
        json.dump(dict_rep, fp=f, indent=4)
    return

def load_mesh(file_location: PathLike) -> Mesh:
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
