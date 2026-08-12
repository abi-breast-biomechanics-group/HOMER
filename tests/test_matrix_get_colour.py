from HOMER import cube, L1Basis, L4Basis, H3Basis, L3Basis
import jax
from matplotlib import pyplot as plt
import numpy as np 
from scipy.sparse import csr_array
import networkx as nx

mesh = cube(basis=[L1Basis]*3)
mesh.refine([3,3,3])
# mesh = mesh.rebase([H3Basis]*3)
# mesh = mesh.rebase([L3Basis]*3)
coloring_dict, seed_matrix = mesh.get_colouring_dict(fields_seperable=True, seed_matrix = True)

# 5. Interpret the Results
n_colors = max(coloring_dict.values()) + 1
print(f"Jacobian compressed into {n_colors} evaluations for {mesh.optimisable_param_array.shape[0]} parameters")
# Map colors back to column indices

params = np.zeros(len(mesh.optimisable_param_array)//3)
for col_idx, color in coloring_dict.items():
    params[col_idx] = color
node_colours = params
mesh.plot(node_colour=node_colours, node_size=50)

# so now we have the base colouring block for a mesh - you can do an inverse lookup into the block to generate the value.

# in element x

# lookup elemap into the colour vector

# assin

# so assume the individual elements are tiled nicely







