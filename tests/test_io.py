"""JSON serialisation round trips.

``mesh_io_test.py`` saved a mesh, loaded it and drew the result.  A round trip
either returns the same mesh or it does not, and "the same" means the
parameter vector, the constraint mask, the identifiers and the secondary
fields -- not just a picture that looks familiar.
"""

import json

import numpy as np
import pytest

from HOMER import Mesh, MeshElement, MeshField, MeshNode
from HOMER.basis_definitions import (B3Basis, H3Basis, L1Basis, L2Basis,
                                     L3Basis, L4Basis)
from HOMER.io import (STR_LOOKUP, dump_mesh_to_dict, dump_meshfield_to_dict,
                      load_mesh, parse_mesh_from_dict, save_mesh)

from _helpers import EXACT, arr, hermite_cube, node_locs, unit_hex


def roundtrip(mesh, tmp_path, name='mesh.json'):
    path = tmp_path / name
    mesh.save(path)
    return Mesh.load(path)


def unit_cube_field():
    """A MeshField over the unit cube corners, used as a secondary field."""
    locs = list(np.ndindex(2, 2, 2))
    nodes = [MeshNode(loc=np.array(l, dtype=float)) for l in locs]
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(L1Basis, L1Basis, L1Basis))
    return Mesh(nodes=nodes, elements=element)


############################################### the basic round trip

def test_geometry_survives_a_round_trip(tmp_path):
    mesh = hermite_cube()
    mesh.refine(2)

    loaded = roundtrip(mesh, tmp_path)

    assert len(loaded.nodes) == len(mesh.nodes)
    assert len(loaded.elements) == len(mesh.elements)
    np.testing.assert_array_equal(arr(loaded.true_param_array), arr(mesh.true_param_array))
    np.testing.assert_allclose(node_locs(loaded), node_locs(mesh), atol=EXACT)


@pytest.mark.parametrize("basis", [L1Basis, L2Basis, L3Basis, L4Basis, H3Basis, B3Basis],
                         ids=lambda b: b.__name__)
def test_every_basis_can_be_named_and_looked_up_again(basis, tmp_path):
    mesh = unit_hex(basis=[basis] * 3)

    loaded = roundtrip(mesh, tmp_path)

    assert [b.__name__ for b in loaded.elements[0].basis_functions] == [basis.__name__] * 3
    np.testing.assert_array_equal(arr(loaded.true_param_array), arr(mesh.true_param_array))


def test_the_written_file_is_plain_json(tmp_path):
    path = tmp_path / 'mesh.json'
    unit_hex().save(path)

    payload = json.loads(path.read_text())

    assert set(payload) == {'main', 'fields'}
    assert set(payload['main']) == {'nodes', 'elements'}


def test_load_accepts_a_string_path(tmp_path):
    path = tmp_path / 'mesh.json'
    mesh = unit_hex()
    mesh.save(str(path))

    loaded = load_mesh(str(path))

    np.testing.assert_array_equal(arr(loaded.true_param_array), arr(mesh.true_param_array))


def test_an_unknown_basis_name_is_rejected():
    payload = dump_mesh_to_dict(unit_hex())
    payload['main']['elements'][0]['basis'] = ['NotABasis'] * 3

    with pytest.raises(KeyError):
        parse_mesh_from_dict(payload)


############################################### constraints

def test_fixed_parameters_survive_a_round_trip(tmp_path):
    mesh = unit_hex(basis=[H3Basis] * 3)
    mesh.nodes[0].fix_parameter('loc', inds=[2])
    mesh.nodes[3].fix_parameter(['loc', 'du'])
    mesh.generate_mesh()

    loaded = roundtrip(mesh, tmp_path)

    assert list(loaded.nodes[0].fixed_params['loc']) == [2]
    assert set(loaded.nodes[3].fixed_params) == {'loc', 'du'}
    np.testing.assert_array_equal(loaded.optimisable_param_bool, mesh.optimisable_param_bool)
    assert len(loaded.optimisable_param_array) == len(mesh.optimisable_param_array)


############################################### identifiers

def test_node_and_element_ids_survive_a_round_trip(tmp_path):
    """Ids are dict keys, and JSON has no tuples -- a tuple id must come back
    hashable, not as a list."""
    ids = ['node_1', 2, '3', (1, 1)]
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0]]
    zero = np.zeros(3)
    nodes = [MeshNode(loc=np.array(l, dtype=float), du=zero, dv=zero, dudv=zero, id=i)
             for l, i in zip(locs, ids)]
    element = MeshElement(node_ids=ids, basis_functions=(H3Basis, H3Basis), id='patch')
    mesh = Mesh(nodes=nodes, elements=element)

    loaded = roundtrip(mesh, tmp_path)

    assert [n.id for n in loaded.nodes] == ids
    assert [e.id for e in loaded.elements] == ['patch']
    assert loaded.node_id_to_ind == mesh.node_id_to_ind
    assert loaded.element_id_to_ind == {'patch': 0}


def test_an_element_referencing_nodes_by_id_reloads_the_same_way(tmp_path):
    ids = ['a', 'b', 'c', 'd']
    locs = [[0, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 0]]
    nodes = [MeshNode(loc=np.array(l, dtype=float), id=i) for l, i in zip(locs, ids)]
    mesh = Mesh(nodes=nodes,
                elements=MeshElement(node_ids=ids, basis_functions=(L1Basis, L1Basis)))

    loaded = roundtrip(mesh, tmp_path)

    np.testing.assert_allclose(arr(loaded.evaluate_embeddings(0, np.array([[0.5, 0.5]]))),
                               arr(mesh.evaluate_embeddings(0, np.array([[0.5, 0.5]]))),
                               atol=EXACT)


############################################### secondary fields

def test_secondary_fields_survive_a_round_trip():
    mesh = unit_cube_field()
    field_nodes = [MeshNode(loc=node.loc * 2.0) for node in mesh.nodes]
    field = MeshField(nodes=field_nodes,
                      elements=MeshElement(node_indexes=mesh.elements[0].nodes,
                                           basis_functions=mesh.elements[0].basis_functions))
    mesh["double"] = field

    loaded = Mesh.from_dict(mesh.to_dict())

    assert "double" in loaded.fields
    assert len(loaded.nodes) == len(mesh.nodes)
    assert len(loaded["double"].nodes) == len(field_nodes)
    np.testing.assert_allclose(loaded["double"].nodes[0].loc, field_nodes[0].loc, atol=EXACT)


def test_field_constraints_survive_a_round_trip(tmp_path):
    mesh = unit_cube_field()
    field = MeshField(nodes=[MeshNode(loc=np.zeros(1)) for _ in mesh.nodes],
                      elements=MeshElement(node_indexes=mesh.elements[0].nodes,
                                           basis_functions=mesh.elements[0].basis_functions))
    field.nodes[2].fix_parameter('loc')
    field.generate_mesh()
    mesh['scalar'] = field

    loaded = roundtrip(mesh, tmp_path)

    assert list(loaded['scalar'].nodes[2].fixed_params['loc']) == [0]
    assert (~loaded['scalar'].optimisable_param_bool).sum() == 1


def test_a_legacy_field_only_payload_still_loads():
    """Files written before the ``{'main', 'fields'}`` schema must keep working."""
    mesh = unit_cube_field()

    loaded = Mesh.from_dict(dump_meshfield_to_dict(mesh))

    assert isinstance(loaded, Mesh)
    assert loaded.fields == {}
    assert len(loaded.nodes) == len(mesh.nodes)


def test_dump_of_a_bare_meshfield_uses_the_legacy_shape():
    field = MeshField(nodes=[MeshNode(loc=np.array(l, dtype=float))
                             for l in np.ndindex(2, 2, 2)],
                      elements=MeshElement(node_indexes=list(range(8)),
                                           basis_functions=(L1Basis,) * 3))

    payload = dump_mesh_to_dict(field)

    assert set(payload) == {'nodes', 'elements'}


def test_save_mesh_and_load_mesh_are_the_module_level_pair(tmp_path):
    mesh = unit_hex(basis=[L2Basis] * 3)
    path = tmp_path / 'field.json'

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    np.testing.assert_array_equal(arr(loaded.true_param_array), arr(mesh.true_param_array))


def test_str_lookup_covers_every_exported_basis():
    """A basis missing from the registry cannot be loaded back."""
    assert {'L1Basis', 'L2Basis', 'L3Basis', 'L4Basis',
            'H3Basis', 'B3Basis'} <= set(STR_LOOKUP)


def test_a_basis_group_round_trips_through_json(tmp_path):
    """The saved name is what rebuilds the group, direction by direction."""
    from HOMER.geometry import cube

    mesh = cube(basis=H3Basis * 2 + L1Basis)
    path = tmp_path / 'group.json'
    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert loaded.elements[0].basis_functions == H3Basis * 2 + L1Basis
