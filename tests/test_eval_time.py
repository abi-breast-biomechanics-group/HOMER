import jax
from HOMER import cube
import numpy as np
import time


mesh = cube()
mesh.refine(3)

pts = np.random.random((1000, 3))
mesh.evaluate_embeddings_in_every_element(pts)

start = time.time()
for _ in range(100):
    mesh.evaluate_embeddings_in_every_element(pts)
end = time.time()
print(f"took {(end - start)/100} seconds")


mesh.evaluate_embeddings_in_every_element = jax.jit(mesh.evaluate_embeddings_in_every_element, static_argnames='self')
mesh.evaluate_embeddings_in_every_element(pts)

start = time.time()
for _ in range(100):
    mesh.evaluate_embeddings_in_every_element(pts)
end = time.time()
print(f"compiled eval took {(end - start)/100} seconds")
