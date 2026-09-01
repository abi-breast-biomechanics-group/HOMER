import pyvista as pv
import time
from scipy.spatial import KDTree
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
# import asdex

from scipy.optimize import approx_fprime, least_squares
from matplotlib import pyplot as plt

from HOMER import MeshNode, MeshElement, Mesh, L1Basis, L2Basis, L4Basis, H3Basis, jacobian_evaluator
from HOMER.fitting import point_cloud_fit, jacobian


seed = np.random.seed(42)
#CONSTRUCT THE DATA TO embed the point in
point0 = MeshNode(loc=[0,0,1])
point0_1 = MeshNode(loc=[0,0,0.5])
point1 = MeshNode(loc=[0,0,0])
point2 = MeshNode(loc=[0,1,1])
point2_3 = MeshNode(loc=[0,1,0.5])
point3 = MeshNode(loc=[0,1,0])
point0_2 = MeshNode(loc=[0,0.5,1])
point1_3 = MeshNode(loc=[0,0.5,0])
point_middle = MeshNode(loc=[0.5, 0.5, 0.5])
element0 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7,8], basis_functions=(L2Basis, L2Basis))
mesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0, jax_compile=True)

point0.fix_parameter('loc')
point1.fix_parameter('loc')
point2.fix_parameter('loc')
point3.fix_parameter('loc')
mesh.refine(5)
# mesh.plot(labels=True)

# for n in [0, 17, 38, 48]:
#     mesh.nodes[n].fix_parameter('loc')
mesh.generate_mesh()



s = pv.Plotter()
mesh.plot(s, node_colour='g')

pts = lambda :np.random.rand(1_000_000, 3)
# breakpoint()
# pnorm = pts[:, 1] **2 + pts[:, 2] **2
# pts[:, 0] = 0.3 + pnorm/20

#look at an initial embedding:
# eles, xis = mesh.embed_points(pts, verbose=3, grid_res=1, iterations=1)
# @jax.jit
def loc_dist(ps):
    _, res = mesh.embed_points(ps, return_residual=True)
    return res.flatten()                 

# loc_dist(pts)
ele_xi = mesh.embed_points(pts(), verbose=3)

@jax.jit
def loc_dist_init(ps):
    _, res = mesh.embed_points(ps, return_residual=True, init_elexi=ele_xi)
    return res.flatten()                 

loc_dist(pts())
# loc_dist_init(pts())

time_res = 10
start = time.time()
for _ in range(time_res):
    loc_dist(pts()).block_until_ready()
end = time.time()
print(f"full took {(end - start)/time_res:.2f} seconds")
#
# start = time.time()
# for _ in range(time_res):
#     loc_dist_init(pts()).block_until_ready()
# end = time.time()
# print(f"init took {(end - start)/time_res:.2f} seconds")


# with jax.profiler.trace("/tmp/jax-trace"):
#     # Run the workload you want to profile
#     for i in range(10):
#         result = loc_dist(pts()).block_until_ready()
# import jax
# import jax.numpy as jnp

