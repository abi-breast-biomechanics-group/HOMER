from HOMER import cube, L1Basis, L4Basis
import jax
from matplotlib import pyplot as plt
import numpy as np 
from scipy.sparse import csr_array
import networkx as nx

mesh = cube(basis=[L1Basis]*3)
mesh.refine(8)
print(len(mesh.nodes))
# mesh.plot()

graph_struct = np.zeros((len(mesh.elements), len(mesh.optimisable_param_array)))
print(graph_struct.shape)

for ide, emap in enumerate(mesh.ele_map):
    graph_struct[ide, emap.astype(int)] = 1

graph_struct = graph_struct[:, ::3] #note that we can basically repeat this structure 3 times.
jacobian = csr_array(graph_struct)

# 2. Build the Column Intersection Graph
# Columns i and j are connected if (J^T * J)[i, j] != 0
# We only care about the structure, so we use boolean math
adj_matrix = (jacobian.T @ jacobian).tocsr()
adj_matrix.setdiag(0) # Remove self-loops for coloring
adj_matrix.eliminate_zeros()

G = nx.from_scipy_sparse_array(adj_matrix)
G.remove_edges_from(nx.selfloop_edges(G))

# 4. Perform the Greedy Coloring
# 'DSATUR' or 'largest_first' are excellent for Jacobian compression
coloring_dict = nx.coloring.greedy_color(G, strategy="largest_first")

# 5. Interpret the Results
n_colors = max(coloring_dict.values()) + 1
print(f"Jacobian compressed into {n_colors} evaluations.")
# Map colors back to column indices
params = np.zeros(len(mesh.optimisable_param_array)//3)
for col_idx, color in coloring_dict.items():
    params[col_idx] = color


# plt.plot(params); plt.show()

# now we want to color the nodes by the values!
node_colours = params
mesh.plot(node_colour=node_colours)

# so now we have the base colouring block for a mesh - you can do an inverse lookup into the block to generate the value.

# in element x

# lookup elemap into the colour vector

# assin

# so assume the individual elements are tiled nicely







