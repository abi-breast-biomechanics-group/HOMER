"""The import surface, and the class assembly that the split has to preserve.

`MeshField` is built by binding functions defined in sibling modules into the
class body.  That is load-bearing in a way a normal refactor is not: it is what
keeps `@expand_wide_evals` working (it reads `vars(cls)`, so a method reached
through a base class would be invisible to it), and it is what makes the
methods reachable at all.  These tests fail loudly if either goes.
"""

import pytest

import HOMER
from HOMER import mesh as mesh_pkg
from HOMER.mesh.field import MeshField
from HOMER.mesh.mesh import Mesh


#every method MeshField gets from a sibling module, by the module it comes from
BOUND = {
    'plotting': ['get_surface', 'get_hex_surface', 'get_triangle_surface',
                 'get_lines', 'plot', 'plot_strains'],
    'evaluation': ['evaluate_embeddings', 'evaluate_deriv_embeddings',
                   'evaluate_element_embeddings', 'evaluate_normals',
                   'eval_numeric_jac', 'evaluate_jacobians', 'xi_grid',
                   'gauss_grid', 'eval_surface', 'embed_points',
                   'evaluate_sobolev', 'get_volume', 'evaluate_strain'],
    'parameters': ['get_element_params', 'update_from_params', 'unfix_mesh',
                   'get_xi_weight_mat', 'linear_fit'],
    'topology': ['associated_node_index', '_explore_topology',
                 'get_xi_surface_nodes', 'get_faces', 'topo_chain_check',
                 '_update_id_mappings', '_clean_pts', 'get_colouring_dict'],
    'refinement': ['refine', 'rebase'],
}


@pytest.mark.parametrize("module,name",
                         [(m, n) for m, names in BOUND.items() for n in names],
                         ids=lambda x: x if isinstance(x, str) else str(x))
def test_every_sibling_method_is_bound_into_the_class(module, name):
    """`vars(MeshField)` must hold it -- not just `getattr` via a base class."""
    import importlib
    bound = vars(MeshField).get(name)
    assert bound is not None, f"MeshField.{name} is not bound"
    assert bound is getattr(importlib.import_module(f'HOMER.mesh.{module}'), name)


@pytest.mark.parametrize("name", ['evaluate_embeddings', 'evaluate_deriv_embeddings',
                                  'evaluate_normals', 'evaluate_jacobians',
                                  'eval_numeric_jac', 'evaluate_strain'])
def test_wide_eval_variants_are_still_generated(name):
    """`@expand_wide_evals` reads `vars(cls)`; binding is what keeps it fed."""
    assert callable(getattr(MeshField, f'{name}_in_every_element', None))
    assert callable(getattr(MeshField, f'{name}_ele_xi_pair', None))


def test_mesh_overrides_plot_with_its_own_implementation():
    """Mesh.plot draws the secondary fields; it must not inherit MeshField's."""
    from HOMER.mesh import plotting
    assert vars(Mesh)['plot'] is plotting.plot_mesh
    assert vars(MeshField)['plot'] is plotting.plot


@pytest.mark.parametrize("name", ['MeshNode', 'MeshElement', 'MeshField', 'Mesh',
                                  'GAUSS', 'volume_quadrature_order',
                                  'column_equilibrated_lstsq', 'MAX_XI_DENOMINATOR',
                                  'make_eval', 'make_deriv_eval', 'make_weight_eval'])
def test_the_mesher_shim_still_re_exports_everything(name):
    """`from HOMER.mesher import ...` predates the split and must keep working."""
    import HOMER.mesher as mesher
    assert getattr(mesher, name) is getattr(mesh_pkg, name)


@pytest.mark.parametrize("name", ['Mesh', 'MeshElement', 'MeshNode', 'MeshField'])
def test_the_top_level_package_exports_the_classes(name):
    assert getattr(HOMER, name) is getattr(mesh_pkg, name)


@pytest.mark.parametrize("name", ['jacobian', 'matrix_free_jacobian'])
def test_the_top_level_package_exports_both_jacobian_pathways(name):
    """The assembled one and the operator; a fit reaches for one or the other."""
    from HOMER import jacobian_evaluator
    assert getattr(HOMER, name) is getattr(jacobian_evaluator, name)
