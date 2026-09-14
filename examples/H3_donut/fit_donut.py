"""Fit a closed bicubic Hermite (H3 x H3) surface mesh to the Blender donut.

Pipeline
--------
1. Export the donut body from ``DONA.blend`` as a PLY surface (headless
   Blender, only when ``donut_points.ply`` is missing).
2. Estimate a torus (centre, major radius, tube half-axes) from the cloud
   and build a periodic bilinear (L1 x L1) torus grid with those parameters.
3. Rebase the grid to cubic Hermite, which fills in the du/dv/dudv fields.
4. Sobolev-regularised least-squares fit: every point is embedded on the
   mesh surface and its residual vector minimised, following the MeshVOS
   skin-fitting pattern (``meshvos.fitting.fit_skin_surface``).
5. Report residuals on the full cloud, save ``h3_donut_initial.json`` and
   ``h3_donut.json``, and render ``h3_donut_fit.png``: the coloured Blender
   donut (body, icing, sprinkles) with the initial and fitted tori overlaid,
   plus the point-to-surface residual map.

Usage::

    python fit_donut.py            # fit + save PNG
    python fit_donut.py --show     # also open the interactive scene
"""
import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

from HOMER import H3Basis, L1Basis, Mesh, MeshElement, MeshNode
from HOMER.jacobian_evaluator import jacobian

HERE = Path(__file__).resolve().parent
BLEND_FILE = HERE / "DONA.blend"
EXPORT_SCRIPT = HERE / "export_donut_points.py"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

N_MAJOR = 8          # elements around the ring
N_MINOR = 4          # elements around the tube
N_FIT_POINTS = 5000  # subsample used inside the optimiser
SOBOLEV_WEIGHT = 0.1
MAX_NFEV = 50
SEED = 0


def export_points(blender: str) -> None:
    """Run Blender headlessly to write the body and scene PLY files."""
    subprocess.run(
        [blender, "-b", str(BLEND_FILE), "--python", str(EXPORT_SCRIPT),
         "--", str(HERE)],
        check=True)


def initial_torus(points: np.ndarray) -> Mesh:
    """Build a periodic L1 torus from cloud statistics, rebased to H3 x H3.

    The tube cross-section is an ellipse: half-width ``a`` in the radial
    direction and half-height ``b`` along z, because the Blender torus is
    scaled anisotropically.
    """
    centre = points.mean(axis=0)
    rho = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
    major = rho.mean()
    a = (rho.max() - rho.min()) / 2
    b = (points[:, 2].max() - points[:, 2].min()) / 2
    print(f"Initial torus: centre={np.round(centre, 3)}, R={major:.3f}, "
          f"a={a:.3f}, b={b:.3f}")

    nodes = []
    for i in range(N_MAJOR):
        theta = 2 * np.pi * i / N_MAJOR
        for j in range(N_MINOR):
            phi = 2 * np.pi * j / N_MINOR
            radial = major + a * np.cos(phi)
            nodes.append(MeshNode(loc=centre + np.array([
                radial * np.cos(theta), radial * np.sin(theta),
                b * np.sin(phi)])))

    def node(i, j):
        return (i % N_MAJOR) * N_MINOR + (j % N_MINOR)

    elements = [
        MeshElement(node_indexes=[node(i, j), node(i + 1, j),
                                  node(i, j + 1), node(i + 1, j + 1)],
                    basis_functions=(L1Basis, L1Basis))
        for i in range(N_MAJOR) for j in range(N_MINOR)]

    linear = Mesh(nodes=nodes, elements=elements)
    return linear.rebase([H3Basis, H3Basis])


def fit_surface(mesh: Mesh, data: np.ndarray) -> Mesh:
    """Sobolev-regularised point-to-surface fit (MeshVOS pattern)."""
    init_sob = mesh.evaluate_sobolev()

    def optfun(params):
        sob_err = mesh.evaluate_sobolev(fit_params=params) - init_sob
        _, dist_err = mesh.embed_points(
            data, surface_embed=True, grid_res=5,
            return_residual=True, fit_params=params)
        return jnp.concatenate((
            sob_err.ravel() * SOBOLEV_WEIGHT, dist_err.ravel()))

    print(f"Fitting {len(mesh.elements)} elements, {len(mesh.nodes)} nodes, "
          f"{len(mesh.optimisable_param_array)} free parameters "
          f"to {len(data)} points")
    ff, jj = jacobian(optfun, mesh.optimisable_param_array, sparse=True)
    result = least_squares(ff, mesh.optimisable_param_array, jac=jj,
                           verbose=1, max_nfev=MAX_NFEV)
    mesh.update_from_params(result.x)
    return mesh


def residual_norms(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    _, residual = mesh.embed_points(
        points, surface_embed=True, grid_res=5, return_residual=True)
    return np.asarray(jnp.linalg.norm(residual, axis=-1))


def render(initial: Mesh, fitted: Mesh, scene: pv.PolyData,
           body: pv.PolyData, norms: np.ndarray,
           png_path: Path, show: bool) -> None:
    plotter = pv.Plotter(shape=(1, 2), off_screen=not show,
                         window_size=(1600, 700), border=False)

    plotter.subplot(0, 0)
    plotter.add_text("Blender donut with initial (blue) and fitted (black) "
                     "H3 x H3 tori", font_size=12)
    plotter.add_mesh(scene, scalars="RGB", rgb=True, opacity=0.6,
                     smooth_shading=True)
    initial.plot(plotter, mesh_opacity=0.0, node_size=10, line_width=3,
                 node_colour="blue", line_colour="blue")
    fitted.plot(plotter, mesh_opacity=0.0, node_size=12, line_width=3,
                node_colour="red", line_colour="black")

    plotter.subplot(0, 1)
    plotter.add_text("Point-to-surface residual", font_size=12)
    cloud = pv.PolyData(np.asarray(body.points))
    cloud["residual"] = norms
    plotter.add_mesh(cloud, scalars="residual", cmap="viridis",
                     point_size=4, render_points_as_spheres=True,
                     scalar_bar_args={"title": "residual"})
    fitted.plot(plotter, mesh_opacity=0.0, node_size=0, line_width=2)

    plotter.link_views()
    plotter.view_isometric()
    if show:
        plotter.show(screenshot=str(png_path))
    else:
        plotter.screenshot(str(png_path))
    print(f"Saved {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--blender", default=DEFAULT_BLENDER,
                        help="Blender executable used to export the PLY files")
    parser.add_argument("--show", action="store_true",
                        help="Open the interactive scene after saving")
    args = parser.parse_args()

    body_path = HERE / "donut_points.ply"
    scene_path = HERE / "donut_scene.ply"
    if not (body_path.exists() and scene_path.exists()):
        export_points(args.blender)
    body = pv.read(body_path)
    scene = pv.read(scene_path)
    points = np.asarray(body.points, dtype=float)
    print(f"Loaded {len(points)} body points and "
          f"{scene.n_points} coloured scene points")

    initial_mesh = initial_torus(points)
    initial_mesh.save(HERE / "h3_donut_initial.json")
    initial = residual_norms(initial_mesh, points)
    mesh = deepcopy(initial_mesh)

    rng = np.random.default_rng(SEED)
    subset = points[rng.choice(len(points), N_FIT_POINTS, replace=False)]
    mesh = fit_surface(mesh, subset)

    final = residual_norms(mesh, points)
    print("Residual on all points (model units):")
    print(f"  initial  mean={initial.mean():.4f}  "
          f"p95={np.percentile(initial, 95):.4f}  max={initial.max():.4f}")
    print(f"  fitted   mean={final.mean():.4f}  "
          f"p95={np.percentile(final, 95):.4f}  max={final.max():.4f}")

    mesh.save(HERE / "h3_donut.json")
    print(f"Saved {HERE / 'h3_donut.json'}")
    render(initial_mesh, mesh, scene, body, final,
           HERE / "h3_donut_fit.png", args.show)


if __name__ == "__main__":
    main()
