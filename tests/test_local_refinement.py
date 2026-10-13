"""Refining some elements and not their neighbours.

``refine_elements`` halves the chosen elements, keeping the mesh 2:1 balanced,
and leaves the nodes they would put on a face shared with a coarser element
hanging: no node of their own, and a parameter map drawing them from the
coarse face.  The claims worth pinning are that the children reproduce their
parents at any depth, that the field stays continuous across a hanging face
however its parameters move, that the balance holds, and that the rest of the
library -- fitting, colouring, saving, topology -- treats the maps as it treats
any other.

Each element's box places it in the lattice the mesh was refined from, which
is what lets a test follow a point from a refined mesh back to the original.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from HOMER import Mesh
from HOMER.basis_definitions import B3, H3, L1, L2, L3
from HOMER.geometry import basic_surface, cube
from HOMER.examples import hermite_cube
from HOMER.jacobian_evaluator import make_jac_for_mesh_func

from _helpers import EXACT, arr


def block(basis):
    """A curved 2x2x2 block, so a hanging node that drifted off the face would show."""
    mesh = hermite_cube() if basis is H3 else hermite_cube().rebase([basis] * 3)
    mesh.refine(2)
    return mesh


def sheet(basis):
    mesh = basic_surface(basis=[basis] * 2)
    mesh.refine(2)
    return mesh


def inner(mesh):
    """The finest element furthest into the mesh: refining it makes the balance reach out."""
    finest = min(e.box[2] for e in mesh.elements)
    return [max((i for i, e in enumerate(mesh.elements) if e.box[2] == finest),
                key=lambda i: sum(mesh.elements[i].box[1]))]


CASES = {'3D-L1': (block, L1, [[0]]), '3D-L2': (block, L2, [[0]]), '3D-L3': (block, L3, [[0]]),
         '3D-H3': (block, H3, [[0]]), '2D-L1': (sheet, L1, [[0, 3]]), '2D-H3': (sheet, H3, [[0, 3]]),
         '3D-L2-layered': (block, L2, [[0], inner]), '3D-H3-layered': (block, H3, [[0], inner]),
         '2D-H3-layered': (sheet, H3, [[0], inner, inner]),
         '3D-B3': (block, B3, [[0]]), '2D-B3': (sheet, B3, [[0, 3]]),
         '2D-B3-layered': (sheet, B3, [[0], inner, inner])}


def refined(case):
    build, basis, steps = CASES[case]
    mesh = build(basis)
    for step in steps:
        mesh.refine_elements(step(mesh) if callable(step) else step)
    return mesh


def boxes(mesh):
    """``(origins, sizes)`` of every element, as floats -- exact, being dyadic."""
    return (np.array([[float(x) for x in e.box[1]] for e in mesh.elements]),
            np.array([float(e.box[2]) for e in mesh.elements]))


def locate(mesh, points):
    """The element holding each lattice point, and the point's xi there."""
    origins, sizes = boxes(mesh)
    xis = (points[:, None, :] - origins[None]) / sizes[None, :, None]
    eles = np.argmax(((xis >= 0) & (xis <= 1)).all(-1), axis=1)
    return eles, xis[np.arange(len(points)), eles]


@pytest.mark.parametrize("case", list(CASES))
def test_the_children_reproduce_their_parents(case):
    build, basis, _ = CASES[case]
    original = build(basis)
    original.refine_elements([])        #gives every element its box, and nothing else
    mesh = refined(case)
    rng = np.random.default_rng(0)
    eles, xis = rng.integers(len(mesh.elements), size=300), rng.random((300, mesh.ndim))

    origins, sizes = boxes(mesh)
    points = origins[eles] + xis * sizes[eles, None]

    np.testing.assert_allclose(arr(mesh.evaluate_embeddings_ele_xi_pair(eles, xis)),
                               arr(original.evaluate_embeddings_ele_xi_pair(*locate(original, points))),
                               atol=EXACT)


@pytest.mark.parametrize("case", list(CASES))
def test_the_field_stays_continuous_across_a_hanging_face(case):
    """With every DOF moved at random, the two sides of each hanging face still meet."""
    mesh = refined(case)
    assert mesh.hanging

    rng = np.random.default_rng(1)
    mesh.update_from_params(arr(mesh.true_param_array) + rng.normal(size=mesh.true_param_array.shape))

    origins, sizes = boxes(mesh)
    t = rng.random((20, mesh.ndim))
    for (small, dim, side), (big, big_dim, big_side) in mesh.hanging.items():
        assert (big_dim, big_side) == (dim, 1 - side)
        xi_small = t.copy()
        xi_small[:, dim] = side
        xi_big = (origins[small] + xi_small * sizes[small] - origins[big]) / sizes[big]

        np.testing.assert_allclose(arr(mesh.evaluate_embeddings_ele_xi_pair([small] * 20, xi_small)),
                                   arr(mesh.evaluate_embeddings_ele_xi_pair([big] * 20, xi_big)),
                                   atol=1e-4)


@pytest.mark.parametrize("case", ['3D-L2-layered', '2D-H3-layered'])
def test_touching_elements_are_never_more_than_one_level_apart(case):
    mesh = refined(case)
    origins, sizes = boxes(mesh)

    meets = ((origins[:, None] <= origins[None] + sizes[None, :, None]).all(-1)
             & (origins[None] <= origins[:, None] + sizes[:, None, None]).all(-1))
    ratio = sizes[None, :] / sizes[:, None]

    assert ratio[meets].max() == 2


@pytest.mark.parametrize("basis", [L1, H3, B3], ids=lambda b: b.name)
def test_the_exposed_faces_are_the_ones_on_the_outside(basis):
    """Scattered refinement of a cube: a face is exposed exactly when its box face is on the cube's."""
    mesh = cube(basis=[basis] * 3)
    mesh.refine(4)
    mesh.refine_elements(list(range(0, len(mesh.elements), 7)))
    origins, sizes = boxes(mesh)
    low, high = origins.min(0), (origins + sizes[:, None]).max(0)

    outside = {(e, d, side) for e in range(len(mesh.elements)) for d in range(3) for side in (0, 1)
               if (origins[e, d] + side * sizes[e]) == (high[d] if side else low[d])}

    assert set(mesh.faces) == outside


def test_the_balance_refines_what_the_children_would_touch():
    """The inner child of a refined corner touches all seven coarse elements."""
    mesh = block(L1)
    mesh.refine_elements([0])
    coarse = [i for i, e in enumerate(mesh.elements) if e.box[2] == 1]

    refined_too = mesh.refine_elements(inner(mesh))

    assert set(coarse) <= set(refined_too.tolist())


def test_refining_a_neighbour_joins_the_faces_that_hung_against_it():
    mesh = block(L1)
    mesh.refine_elements([0])
    neighbour = next(i for i, e in enumerate(mesh.elements) if e.box[2] == 1 and sum(e.box[1]) == 1)

    mesh.refine_elements([neighbour])

    #the face between the two refined elements is ordinary again; each refined
    #corner leaves two faces against coarse neighbours, four children each
    assert len(mesh.hanging) == 16
    #the block's 24, less 3 + 3 for the two corners, plus 12 + 12 for their children
    assert len(mesh.faces) == 42
    drawn = np.unique(np.asarray(mesh.ele_map)[np.asarray(mesh.ele_weights) != 0])
    assert len(drawn) == len(mesh.dof_node)       #no DOF is left without an element


def test_hanging_faces_are_neither_exposed_nor_boundaries():
    mesh = refined('3D-L1')

    #three faces of the refined corner meet unrefined elements, four children each
    assert len(mesh.hanging) == 12
    #the block's 24 exposed faces, less the corner's 3, plus its children's 12
    assert len(mesh.faces) == 33
    assert not set(mesh.hanging) & set(mesh.faces)


@pytest.mark.parametrize("case", ['3D-L2', '3D-H3', '2D-H3', '2D-H3-layered'])
def test_a_linear_fit_recovers_a_field_the_mesh_can_hold(case):
    mesh = refined(case)
    rng = np.random.default_rng(2)
    truth = arr(mesh.true_param_array) + 0.1 * rng.normal(size=mesh.true_param_array.shape)
    mesh.update_from_params(truth)

    grid = mesh.gauss_grid([5] * mesh.ndim)[0] if mesh.ndim == 3 else mesh.xi_grid(6, dim=2)
    eles = np.repeat(np.arange(len(mesh.elements)), len(grid))
    xis = np.tile(grid, (len(mesh.elements), 1))
    targets = arr(mesh.evaluate_embeddings_ele_xi_pair(eles, xis))

    mesh.update_from_params(np.zeros_like(truth))
    weights, columns = mesh.get_xi_weight_blocks(eles, xis)
    mesh.linear_fit(targets, weights, sparse_columns=columns)

    np.testing.assert_allclose(arr(mesh.evaluate_embeddings_ele_xi_pair(eles, xis)), targets,
                               atol=1e-4)


def test_the_coloured_jacobian_matches_the_dense_one():
    mesh = refined('3D-H3')
    eles, grid = np.arange(len(mesh.elements)), mesh.xi_grid(3)

    def residual(params):
        return mesh.evaluate_embeddings(eles, grid, fit_params=params).ravel()

    params = jnp.asarray(mesh.optimisable_param_array)
    coloured = make_jac_for_mesh_func(mesh, residual, False)(params).toarray()

    np.testing.assert_allclose(coloured, np.asarray(jax.jacfwd(residual)(params)), atol=EXACT)


def test_a_locally_refined_mesh_round_trips(tmp_path):
    mesh = refined('3D-H3')
    xis = np.random.default_rng(3).random((10, 3))

    mesh.save(tmp_path / 'local.json')
    loaded = Mesh.load(tmp_path / 'local.json')

    assert loaded.hanging == mesh.hanging
    np.testing.assert_allclose(arr(loaded.evaluate_embeddings_in_every_element(xis)),
                               arr(mesh.evaluate_embeddings_in_every_element(xis)), atol=EXACT)


def test_a_same_basis_field_follows_the_hanging_mesh():
    mesh = refined('3D-L1')

    mesh.new_field('pressure', 1, field_params=np.ones(len(mesh.nodes)))

    np.testing.assert_allclose(arr(mesh['pressure'].evaluate_embeddings_in_every_element([[0.3] * 3])),
                               1.0, atol=EXACT)


@pytest.mark.parametrize("case", list(CASES))
def test_the_functions_still_sum_to_one(case):
    """Every node at one point, and the field is that point everywhere.

    For a control net that is the truncation at work: the coarse functions give
    way exactly where the fine ones take over, so the control points keep their
    geometric meaning.
    """
    mesh = refined(case)
    point = np.array([0.3, -1.2, 2.0])
    for node in mesh.nodes:
        node.loc = point.copy()
        for field in node:
            node[field] = np.zeros(3)
    mesh.generate_mesh()

    xis = np.random.default_rng(6).random((50, mesh.ndim))
    np.testing.assert_allclose(arr(mesh.evaluate_embeddings_in_every_element(xis)),
                               np.broadcast_to(point, (50 * len(mesh.elements), 3)), atol=EXACT)


def test_a_control_net_refines_further_than_a_nodal_basis():
    """B3 functions reach three cells, so the balance reaches that far too."""
    refined_with = []
    for basis in (L1, B3):
        mesh = basic_surface(basis=[basis] * 2)
        mesh.refine(6)
        mesh.refine_elements([0])
        refined_with.append(len(mesh.refine_elements(inner(mesh))))

    assert refined_with[0] < refined_with[1]


def test_a_control_point_goes_once_refinement_covers_its_support():
    """Refining everything around a coarse B3 function retires it, and its node."""
    mesh = sheet(B3)
    before = len(mesh.nodes)

    mesh.refine_elements(list(range(len(mesh.elements))))

    assert not mesh.hanging
    #a 2x2 sheet's 5x5 control net becomes the 4x4 sheet's 7x7, nothing left over
    assert before == 25 and len(mesh.nodes) == 49


@pytest.mark.parametrize("operation", [lambda m: m.refine(2), lambda m: m.rebase([L2] * 3)],
                         ids=['refine', 'rebase'])
def test_lattice_operations_refuse_a_hanging_mesh(operation):
    mesh = refined('3D-L1')

    with pytest.raises(NotImplementedError, match="hanging faces"):
        operation(mesh)


############################################### crossing a hanging face

@pytest.mark.parametrize("case", ['3D-L1', '3D-L2', '3D-H3', '2D-H3', '3D-H3-layered', '2D-H3-layered'])
def test_topomap_crosses_a_hanging_face_from_either_side(case):
    """Just outside a face, the mapped point is where the clipped one was."""
    mesh = refined(case)
    rng = np.random.default_rng(4)

    for (small, dim, side), (big, _, big_side) in mesh.hanging.items():
        for element, s in ((small, side), (big, big_side)):
            xi = 0.01 + 0.98 * rng.random(mesh.ndim)
            xi[dim] = 1 + 1e-6 if s == 1 else -1e-6
            across, mapped, valid = mesh.topomap(element, jnp.asarray(xi))

            assert bool(valid) and int(across) != element
            np.testing.assert_allclose(
                arr(mesh.evaluate_embeddings_ele_xi_pair([int(across)], arr(mapped)[None])),
                arr(mesh.evaluate_embeddings_ele_xi_pair([element], np.clip(xi, 0, 1)[None])),
                atol=EXACT)


def test_a_hanging_face_is_not_reported_as_a_boundary():
    mesh = refined('3D-L1')
    (small, dim, side) = next(iter(mesh.hanging))
    xi = jnp.full(3, 0.5).at[dim].set(float(side))
    on_face = jnp.arange(3) == dim
    at_lo, at_hi = on_face & (side == 0), on_face & (side == 1)

    assert not bool(mesh.topo_chain_check(small, xi, at_lo, at_hi)[dim])


@pytest.mark.parametrize("case", ['3D-L1', '3D-L2', '3D-L2-layered'])
def test_points_embed_across_a_locally_refined_mesh(case):
    mesh = refined(case)
    rng = np.random.default_rng(5)
    eles, xis = rng.integers(len(mesh.elements), size=2000), rng.random((2000, 3))
    points = arr(mesh.evaluate_embeddings_ele_xi_pair(eles, xis))

    _, residual = mesh.embed_points(points, return_residual=True, iterations=25)

    assert np.linalg.norm(arr(residual), axis=-1).max() < 1e-5


def test_surface_nodes_stop_at_the_outer_surface():
    """A hanging face is internal, so none of its nodes are on the boundary."""
    mesh = cube(basis=[L1] * 3)
    mesh.refine(2)
    mesh.refine_elements([0])
    locs = np.array([n.loc for n in mesh.nodes])
    low, high = locs.min(0), locs.max(0)

    for dim, side in itertools.product(range(3), (0, 1)):
        _, nodes = mesh.get_xi_surface_nodes(dim, side)
        on_surface = np.isclose(locs[nodes], low, atol=EXACT) | np.isclose(locs[nodes], high, atol=EXACT)
        assert on_surface.any(axis=1).all()
        assert len(nodes) >= 9
