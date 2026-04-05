from HOMER import cube
import numpy as np
import pyvista as pv

mesh = cube(scale=1, centre=np.zeros(3))
mesh.refine(2)

# s = pv.Plotter()
# mesh.plot(s, elem_labels=True)
e_pt = mesh.evaluate_embeddings(0, [0.5,0.5, 1.1])
e, xi, valid = mesh.topomap(0, [0.5, 0.5, 1.1])
assert e == 1 and np.all(np.isclose(xi, [0.5, 0.5, 0.1])) and valid, "Topology mapping failed"

