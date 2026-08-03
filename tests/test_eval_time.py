import jax
from HOMER import cube
import numpy as np
import time


mesh = cube()
mesh.refine(3)

n_iters = 10

np.random.seed(42)

pts = np.random.random((2_00_000, 3)) #1_000_000 causes OOM errors, TODO: implement chunking
mesh.evaluate_embeddings_in_every_element(pts)

start = time.time()
for _ in range(n_iters):
    mesh.evaluate_embeddings_in_every_element(pts)
end = time.time()
print(f"took {(end - start)/n_iters} seconds")


emb = jax.jit(mesh.evaluate_embeddings_in_every_element, static_argnames='self')
emb(pts)

start = time.time()
for _ in range(n_iters):
    emb(pts)
end = time.time()
print(f"compiled eval took {(end - start)/n_iters} seconds")

mesh.embed_points(pts, verbose=3)
