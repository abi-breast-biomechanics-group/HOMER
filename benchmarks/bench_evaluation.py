"""Throughput of evaluating embeddings at many (element, xi) pairs.

Was ``tests/test_eval_time.py``.  Compares the chunked in-every-element
evaluator against a hand-written scan over the element map.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from HOMER import cube

CHUNK_SIZE = 100_000
N_ITERS = 5
N_POINTS = 5_000_000        #raise until this machine runs out of memory


def build_hand_written_eval(mesh):
    """The scan the library's chunking replaced, kept as a reference point."""

    @jax.jit
    def fast_eval(ele, xi, params):
        full = jnp.asarray(mesh.true_param_array).at[mesh.optimisable_param_bool].set(params)
        per_element = jnp.asarray(
            full[mesh.ele_map.astype(int)].reshape((mesh.ele_map.shape[0], -1, 3)))

        def one(e, xi_single):
            weights = mesh.generate_weight_matrix(xi_single[None])
            return (weights[None, :, 0] @ per_element[e.astype(int)]).squeeze()

        chunked = jax.vmap(one)
        n_chunks = ele.shape[0] // CHUNK_SIZE
        main = n_chunks * CHUNK_SIZE

        def body(carry, chunk):
            return carry, chunked(*chunk)

        _, scanned = jax.lax.scan(
            body, None,
            (ele[:main].reshape((n_chunks, CHUNK_SIZE)),
             xi[:main].reshape((n_chunks, CHUNK_SIZE, xi.shape[1]))))
        tail = chunked(ele[main:], xi[main:])
        return jnp.concatenate([scanned.reshape((main, -1)), tail], axis=0)

    return fast_eval


def timed(label, fn, *args):
    fn(*args)                                        #warm up the compile
    start = time.time()
    for _ in range(N_ITERS):
        jax.block_until_ready(fn(*args))
    print(f"{label}: {(time.time() - start) / N_ITERS:.3f} s/iter")


def main():
    rng = np.random.default_rng(42)
    mesh = cube()
    mesh.refine(3)

    xi = rng.random((N_POINTS, 3))
    eles = rng.integers(0, len(mesh.elements), N_POINTS)
    params = mesh.optimisable_param_array

    hand_written = build_hand_written_eval(mesh)
    timed(f"hand-written scan  ({N_POINTS:,} pts)",
          lambda: hand_written(jnp.asarray(eles, dtype=float), jnp.asarray(xi), params))

    library = jax.jit(mesh.evaluate_embeddings_ele_xi_pair, static_argnames='self')
    timed(f"evaluate_embeddings_ele_xi_pair ({N_POINTS:,} pts)",
          lambda: library(eles, xi))


if __name__ == '__main__':
    main()
