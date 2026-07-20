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
# mesh.refine(1)
# mesh.plot(labels=True)

# for n in [0, 17, 38, 48]:
#     mesh.nodes[n].fix_parameter('loc')
mesh.generate_mesh()



s = pv.Plotter()
mesh.plot(s, node_colour='g')

pts = np.random.rand(10000, 3)
# breakpoint()
pnorm = pts[:, 1] **2 + pts[:, 2] **2
pts[:, 0] = 0.3 + pnorm/20

#look at an initial embedding:
eles, xis = mesh.embed_points(pts, verbose=3, grid_res=3, iterations=5)

def loc_dist(params):
    _, res = mesh.embed_points(pts, fit_params=params, return_residual=True)
    return res.flatten()                 

# there are two ways to implement a jacobian of this type.
############################### STATIC
#by assuming that the element embedding is constant, we can then create a fast sparse jacobian
jac_base = np.zeros((np.prod(pts.shape), mesh.true_param_array.shape[0]))
out_ind = 0
for e in eles:
    local_map = mesh.ele_map[e].astype(int)
    jac_base[out_ind:out_ind+3, local_map] = 1; 
    out_ind+=3
jac_base = jac_base[:, mesh.optimisable_param_bool]
jac_sparse = jax.experimental.sparse.BCOO.fromdense(jac_base)
fitting_function, jacobian_static = jacobian(loc_dist, init_estimate=mesh.optimisable_param_array, sparsity=jac_sparse)

############################### DYNAMIC
# If it's dynamic, we need to pass a fast constructor for the jacobian.
@partial(jax.jit, static_argnums = (3))
def build_jacobian(ele_indices, ele_map, param_locs, total_params):
    local_maps = ele_map[ele_indices] 
    E, K = local_maps.shape
    rows = jnp.arange(E * 3).reshape(E, 3, 1)
    cols = local_maps[:, None, :]
    rows_flat = jnp.broadcast_to(rows, (E, 3, K)).flatten()
    cols_flat = jnp.broadcast_to(cols, (E, 3, K)).flatten()
    jac_base = jnp.zeros((E * 3, total_params))
    jac_base = jac_base.at[rows_flat.astype(int), cols_flat.astype(int)].set(1.0)
    masked_jac = jac_base[:, param_locs]
    return masked_jac

inds = np.where(mesh.optimisable_param_bool)[0]
approx_ele_embed = jax.jit(lambda p: mesh.embed_points(pts, fit_params=p, iterations=3, grid_res=5)[0]) # needs recalcing - jit compile for 2x speedup.
# 0 iterations is okay here because the estimate is probably good enough! Because you only need to get the ele correct, you can also reduce the grid res.
def get_sparsity(params):
    ele = approx_ele_embed(params)
    stest =  scipy.sparse.csr_array(build_jacobian(ele, mesh.ele_map, inds, len(mesh.optimisable_param_array)))
    return stest

_, jacobian_dynamic = jacobian(loc_dist, init_estimate=mesh.optimisable_param_array, sparsity=get_sparsity, sparse=True)

_ = jacobian_dynamic(mesh.optimisable_param_array) #is this really slow only on the first call?
_ = jacobian_static(mesh.optimisable_param_array)
##############################
#now time to drag race:
ts0 = time.time()
for _ in range(1):
    _ = jacobian_dynamic(mesh.optimisable_param_array)
ts1 = time.time()
for _ in range(1):
    _ = jacobian_static(mesh.optimisable_param_array)
ts2 = time.time()

print(f"Dynamic took {(ts1-ts0)/1} seconds, Static took {(ts2 - ts1)/1} seconds")


res = least_squares(fitting_function, x0=mesh.optimisable_param_array, jac=jacobian_static, verbose=2, max_nfev=30)

mesh.update_from_params(res.x)

mesh.plot(s)

s.add_mesh(pts)
s.show()
                            
