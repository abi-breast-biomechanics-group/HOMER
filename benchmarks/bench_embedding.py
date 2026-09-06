"""Throughput of ``embed_points`` on a large point cloud.

Was ``tests/accel_test_embed.py``.  Times a cold embed (coarse seed plus
Newton-Raphson) against a warm one started from a previous answer.
"""

import time

import jax
import numpy as np

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import L2Basis

N_POINTS = 1_000_000
N_ITERS = 5


def build_mesh(refinement=5):
    locs = [[0, 0, 1], [0, 0, 0.5], [0, 0, 0],
            [0, 0.5, 1], [0.5, 0.5, 0.5], [0, 0.5, 0],
            [0, 1, 1], [0, 1, 0.5], [0, 1, 0]]
    element = MeshElement(node_indexes=list(range(9)), basis_functions=(L2Basis, L2Basis))
    mesh = Mesh(nodes=[MeshNode(loc=l) for l in locs], elements=element, jax_compile=True)
    mesh.refine(refinement)
    mesh.generate_mesh()
    return mesh


def timed(label, fn):
    jax.block_until_ready(fn())                      #warm up the compile
    start = time.time()
    for _ in range(N_ITERS):
        jax.block_until_ready(fn())
    print(f"{label}: {(time.time() - start) / N_ITERS:.3f} s/iter")


def main():
    rng = np.random.default_rng(42)
    mesh = build_mesh()
    points = lambda: rng.random((N_POINTS, 3))

    seed = mesh.embed_points(points(), verbose=1)

    timed(f"cold embed ({N_POINTS:,} pts)",
          lambda: mesh.embed_points(points(), return_residual=True)[1])
    timed(f"warm embed ({N_POINTS:,} pts)",
          lambda: mesh.embed_points(points(), init_elexi=seed, return_residual=True)[1])


if __name__ == '__main__':
    main()
