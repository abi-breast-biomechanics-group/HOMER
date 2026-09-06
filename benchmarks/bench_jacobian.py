"""Cost of a sparse Jacobian of the embedding residual.

Was the timing half of ``tests/point_to_plane_fit_test.py``.  Compares the
estimated sparsity pattern against one built by hand from the element map.

Note that ``jacobian(..., sparsity=<callable>)`` -- the "dynamic sparsity"
half of the original script -- raises ``NotImplementedError``-style
``ValueError`` in the current library, so only the static path is timed.
"""

import time

import jax
import numpy as np
from jax.experimental import sparse

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import L2Basis
from HOMER.jacobian_evaluator import jacobian

N_POINTS = 10_000
N_ITERS = 5


def build_mesh():
    locs = [[0, 0, 1], [0, 0, 0.5], [0, 0, 0],
            [0, 0.5, 1], [0.5, 0.5, 0.5], [0, 0.5, 0],
            [0, 1, 1], [0, 1, 0.5], [0, 1, 0]]
    element = MeshElement(node_indexes=list(range(9)), basis_functions=(L2Basis, L2Basis))
    mesh = Mesh(nodes=[MeshNode(loc=l) for l in locs], elements=element, jax_compile=True)
    for corner in (0, 2, 6, 8):
        mesh.nodes[corner].fix_parameter('loc')
    mesh.generate_mesh()
    return mesh


def hand_built_sparsity(mesh, elements):
    """Each residual triple depends only on its own element's parameters."""
    pattern = np.zeros((3 * len(elements), mesh.true_param_array.shape[0]))
    for row, element in enumerate(elements):
        pattern[3 * row:3 * row + 3, mesh.ele_map[element].astype(int)] = 1
    return sparse.BCOO.fromdense(pattern[:, mesh.optimisable_param_bool])


def timed(label, fn):
    fn()                                             #warm up the compile
    start = time.time()
    for _ in range(N_ITERS):
        fn()
    print(f"{label}: {(time.time() - start) / N_ITERS:.3f} s/iter")


def main():
    rng = np.random.default_rng(42)
    mesh = build_mesh()

    points = rng.random((N_POINTS, 3))
    points[:, 0] = 0.3 + (points[:, 1] ** 2 + points[:, 2] ** 2) / 20

    def residual(params):
        _, res = mesh.embed_points(points, fit_params=params, return_residual=True)
        return res.flatten()

    start_params = np.asarray(mesh.optimisable_param_array)
    elements, _ = mesh.embed_points(points, verbose=1, grid_res=3, iterations=5)

    _, estimated_jac = jacobian(residual, init_estimate=start_params)
    _, hand_jac = jacobian(residual, init_estimate=start_params,
                           sparsity=hand_built_sparsity(mesh, np.asarray(elements)))

    timed("estimated sparsity", lambda: estimated_jac(start_params))
    timed("hand-built sparsity", lambda: hand_jac(start_params))


if __name__ == '__main__':
    main()
