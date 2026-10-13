"""The typical workload: hundreds of thousands of points into a mesh of at most ~1k elements.

At that ratio the per-point work decides the time -- the Newton refinement and
the evaluations inside it -- and the seed grid is a fixed cost.  Points are
drawn on the mesh, and again moved off it by 1% of its extent, because data is
rarely on the surface it is fitted to, and an off-mesh point never reaches a
zero residual.
"""

import time

import jax
import numpy as np

from HOMER import B3, H3, L1
from HOMER.examples import hermite_cube
from HOMER.geometry import cube

N_POINTS = 400_000
N_ITERS = 3


def meshes():
    mesh = cube(basis=[L1] * 3)
    mesh.refine(10)
    yield 'L1, 1000 elements', mesh

    mesh = cube(basis=[H3] * 3)
    mesh.refine(8)
    yield 'H3, 512 elements', mesh

    mesh = hermite_cube()
    mesh.refine(6)
    mesh.refine_elements(list(range(0, 216, 5)))
    yield f'H3 curved, locally refined, {len(mesh.elements)} elements', mesh

    mesh = cube(basis=[B3] * 3)
    mesh.refine(6)
    mesh.refine_elements(list(range(0, 216, 7)))
    yield f'B3 THB, {len(mesh.elements)} elements', mesh


def timed(label, fn, n=N_ITERS):
    jax.block_until_ready(fn())                         #warm up the compile
    start = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    print(f"  {label}: {(time.perf_counter() - start) / n:.3f} s/iter")


def main():
    rng = np.random.default_rng(0)
    for label, mesh in meshes():
        print(label)
        params = mesh.optimisable_param_array
        eles = rng.integers(len(mesh.elements), size=N_POINTS)
        xis = rng.random((N_POINTS, 3))
        on = np.asarray(mesh.evaluate_embeddings_ele_xi_pair(eles, xis))
        off = on + rng.normal(scale=0.01 * np.ptp(on, axis=0).max(), size=on.shape)

        timed("evaluate", lambda: mesh.evaluate_embeddings_ele_xi_pair(eles, xis, fit_params=params))
        timed("jacobians", lambda: mesh.evaluate_jacobians_ele_xi_pair(eles, xis, fit_params=params))
        timed("embed, seeding only", lambda: mesh.embed_points(on, iterations=0))
        timed("embed on the mesh", lambda: mesh.embed_points(on))
        timed("embed 1% off the mesh", lambda: mesh.embed_points(off))

        _, residual = mesh.embed_points(on, return_residual=True)
        stuck = np.mean(np.linalg.norm(np.asarray(residual), axis=-1) > 1e-4)
        print(f"  on-mesh points left unconverged: {stuck:.2%}")


if __name__ == "__main__":
    main()
