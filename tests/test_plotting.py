"""Drawing a mesh, headless.

Every script this suite replaced ended in ``plotter.show()``.  Rendering
cannot be "checked" the way a number can, but three things about it can:
the geometry handed to the renderer must match the mesh, the plotter must
end up with actors in it, and an off-screen render must produce a non-blank
image.  That is enough to catch a plot that draws nothing, draws the wrong
shape, or crashes.
"""

import numpy as np
import pyvista as pv
import pytest

from HOMER.basis_definitions import B3, H3, L1, L2
from HOMER.geometry import basic_surface, cube

from _helpers import arr, node_locs
from HOMER.examples import hermite_cube, unit_hex


def drawn_bounds(scene):
    return np.asarray(scene.bounds).reshape(3, 2)


@pytest.fixture(scope="module")
def block():
    mesh = cube(scale=2.0, centre=np.array([1.0, 2.0, 3.0]), basis=[L1] * 3)
    mesh.refine(2)
    return mesh


############################################### what gets drawn

def test_the_drawn_geometry_spans_the_mesh(block, plotter):
    """The strongest cheap statement about a picture: it is in the right place."""
    block.plot(plotter)

    expected = np.column_stack((node_locs(block).min(0), node_locs(block).max(0)))
    np.testing.assert_allclose(drawn_bounds(plotter), expected, atol=1e-4)


def test_plot_adds_nodes_surface_and_wireframe(block, plotter):
    block.plot(plotter)

    assert len(plotter.actors) == 3


def test_labels_replace_the_node_spheres_with_text(block, plotter):
    block.plot(plotter, labels=True, elem_labels=True)

    #two labelled point sets, each contributing a points actor and a labels actor
    assert sum(name.endswith('-labels') for name in plotter.actors) == 2


def test_an_array_of_node_colours_is_accepted(block, plotter):
    block.plot(plotter, node_colour=np.arange(len(block.nodes), dtype=float))

    assert len(plotter.actors) >= 3


def test_each_colour_array_uses_its_own_colour_map(block, plotter):
    lines = block.get_lines()
    hex_pts, _ = block.get_hex_surface(list(range(len(block.elements))))
    block.plot(plotter, render_name='m',
               node_colour=np.arange(len(block.nodes), dtype=float), node_cmap='plasma',
               mesh_colour=hex_pts[:, 2], mesh_cmap='coolwarm',
               line_colour=lines.points[:, 0], line_cmap='viridis')

    lut = lambda name: plotter.actors[name].mapper.lookup_table.cmap.name
    assert (lut('m_nodes'), lut('m_hexes'), lut('m_lines')) == ('plasma', 'coolwarm', 'viridis')


def test_hanging_positions_are_drawn_blue_at_the_node_size(plotter):
    mesh = cube(basis=[L1] * 3)
    mesh.refine(2)
    mesh.refine_elements([0])

    mesh.plot(plotter, render_name='m', node_size=7)

    hanging = plotter.actors['m_hanging']
    assert hanging.mapper.dataset.n_points == len(mesh.hanging_points()) > 0
    assert hanging.prop.point_size == plotter.actors['m_nodes'].prop.point_size == 7
    assert hanging.prop.color == pv.Color('b')


def test_refined_nodes_darken_by_level(plotter):
    mesh = hermite_cube().rebase([B3] * 3)
    mesh.refine(2)
    mesh.refine_elements([0])
    levels = mesh.node_levels()

    mesh.plot(plotter, render_name='m', node_colour='r', level_shade=0.5)

    rgb = plotter.actors['m_nodes'].mapper.dataset['level colour']
    assert set(levels) == {0, 1}
    np.testing.assert_allclose(rgb[:, 0], 0.5 ** levels)


def test_a_mesh_with_nothing_hanging_draws_no_hanging_positions(block, plotter):
    block.plot(plotter, render_name='m')

    assert 'm_hanging' not in plotter.actors


def test_a_surface_mesh_draws(plotter):
    mesh = basic_surface(basis=[L2] * 2)
    mesh.refine(2)

    mesh.plot(plotter)

    np.testing.assert_allclose(drawn_bounds(plotter)[0], [0.0, 0.0], atol=1e-4)


def test_plot_honours_a_parameter_override(plotter):
    mesh = unit_hex()
    shifted = arr(mesh.optimisable_param_array).reshape(-1, 3) + [10.0, 0.0, 0.0]

    mesh.plot(plotter, fit_params=shifted.ravel())

    np.testing.assert_allclose(drawn_bounds(plotter)[0], [10.0, 11.0], atol=1e-4)


def test_a_node_can_draw_itself(plotter):
    mesh = unit_hex()

    mesh.nodes[0].plot(plotter)

    assert len(plotter.actors) > 0


def test_a_field_overlay_draws(plotter):
    mesh = cube(basis=[L1] * 3)
    mesh.new_field('index', field_dimension=1, field_params=np.arange(8, dtype=float))

    mesh.plot(plotter, field_to_draw='index', default_xi_res=3)

    assert len(plotter.actors) > 3


############################################### it really renders

def test_an_off_screen_render_is_not_blank(block, plotter):
    """Catches a plot that adds actors the renderer then draws as nothing."""
    block.plot(plotter)

    image = plotter.screenshot(return_img=True)

    assert image.ndim == 3
    assert image.std() > 1.0        #a blank frame has zero variance


def test_plotting_without_a_scene_returns_without_blocking():
    """``plot(scene=None)`` calls ``show()``; off screen that must return."""
    mesh = unit_hex()

    mesh.plot()                      #no assertion beyond "this returns"


############################################### lattice tilings

def test_a_lattice_tiling_evaluates_onto_the_surface():
    """``lattice_surface.py`` drew a hex tiling over a patch and eyeballed it.

    What matters is that the tiling coordinates are valid xi and that the line
    connectivity indexes points that exist.
    """
    mesh = basic_surface(basis=[L2] * 2)

    points, lines = mesh.xi_grid(res=4, lattice=(1, 1))

    points = np.asarray(points)
    lines = np.asarray(lines).reshape(-1, 3)
    assert points.shape[1] == 2
    assert points.min() >= 0 and points.max() <= 1
    assert lines[:, 1:].max() < len(points)

    drawn = arr(mesh.evaluate_embeddings(0, points))
    np.testing.assert_allclose(drawn[:, 0], 0.0, atol=1e-5)      #the patch plane


def test_hex_surface_tiling_covers_a_volume_element(block):
    points, connectivity = block.get_hex_surface([0], tiling=(4, 3))

    points = arr(points)
    assert connectivity.shape[1] == 3
    assert connectivity[:, 1:].max() < len(points)
    inside = (points >= node_locs(block).min(0) - 1e-4).all(-1) & \
             (points <= node_locs(block).max(0) + 1e-4).all(-1)
    assert inside.all()
