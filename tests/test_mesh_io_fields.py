import numpy as np

from HOMER import Mesh, MeshNode, MeshElement, MeshField, L1Basis
from HOMER.io import dump_meshfield_to_dict


def test_mesh_field_round_trip():
    nodes = [
        MeshNode(loc=np.array([0.0, 0.0, 0.0])),
        MeshNode(loc=np.array([1.0, 0.0, 0.0])),
        MeshNode(loc=np.array([0.0, 1.0, 0.0])),
        MeshNode(loc=np.array([1.0, 1.0, 0.0])),
        MeshNode(loc=np.array([0.0, 0.0, 1.0])),
        MeshNode(loc=np.array([1.0, 0.0, 1.0])),
        MeshNode(loc=np.array([0.0, 1.0, 1.0])),
        MeshNode(loc=np.array([1.0, 1.0, 1.0])),
    ]
    element = MeshElement(
        node_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
        basis_functions=(L1Basis, L1Basis, L1Basis),
    )
    mesh = Mesh(nodes=nodes, elements=element)

    field_nodes = [MeshNode(loc=node.loc * 2.0) for node in nodes]
    field_element = MeshElement(
        node_indexes=element.nodes,
        basis_functions=element.basis_functions,
    )
    field = MeshField(nodes=field_nodes, elements=field_element)
    mesh["double"] = field

    payload = mesh.to_dict()
    loaded = Mesh.from_dict(payload)

    assert "double" in loaded.fields
    assert len(loaded.nodes) == len(mesh.nodes)
    assert len(loaded["double"].nodes) == len(field_nodes)
    assert np.allclose(loaded["double"].nodes[0].loc, field_nodes[0].loc)


def test_legacy_mesh_dict_load():
    nodes = [
        MeshNode(loc=np.array([0.0, 0.0, 0.0])),
        MeshNode(loc=np.array([1.0, 0.0, 0.0])),
        MeshNode(loc=np.array([0.0, 1.0, 0.0])),
        MeshNode(loc=np.array([1.0, 1.0, 0.0])),
        MeshNode(loc=np.array([0.0, 0.0, 1.0])),
        MeshNode(loc=np.array([1.0, 0.0, 1.0])),
        MeshNode(loc=np.array([0.0, 1.0, 1.0])),
        MeshNode(loc=np.array([1.0, 1.0, 1.0])),
    ]
    element = MeshElement(
        node_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
        basis_functions=(L1Basis, L1Basis, L1Basis),
    )
    mesh = Mesh(nodes=nodes, elements=element)

    legacy_payload = dump_meshfield_to_dict(mesh)
    loaded = Mesh.from_dict(legacy_payload)

    assert isinstance(loaded, Mesh)
    assert loaded.fields == {}
    assert len(loaded.nodes) == len(nodes)
