import pyvista as pv
import time
from scipy.spatial import KDTree
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial


from scipy.optimize import approx_fprime, least_squares
from matplotlib import pyplot as plt

from HOMER import MeshNode, MeshElement, Mesh, L1Basis, L2Basis, L4Basis, H3Basis
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
mesh.refine(2)

point0.fix_parameter('loc')
point1.fix_parameter('loc')
point2.fix_parameter('loc')
point3.fix_parameter('loc')
mesh.generate_mesh()


s = pv.Plotter()
mesh.plot(s, node_colour='g')

pts = np.random.rand(1000, 3)
# breakpoint()
pts[:, 0] = 0.3

#look at an initial embedding:
eles, xis = mesh.embed_points(pts, verbose=0)



def loc_dist(params):
    _, res = mesh.embed_points(pts, fit_params=params, return_residual=True)
    return res.flatten()                 


# jac_true = approx_fprime(mesh.optimisable_param_array, loc_dist, epsilon=1e-4)
#
# plt.imshow(jac_true); plt.show()


# there are two ways to implement a jacobian of this type.
############################### STATIC
#by assuming that the element embedding is constant, we can then create a fast sparse jacobian
jac_base = np.zeros((np.prod(pts.shape), mesh.true_param_array.shape[0]))
out_ind = 0
for e in eles:
    local_map = mesh.ele_map[e]
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
    jac_base = jac_base.at[rows_flat, cols_flat].set(1.0)
    masked_jac = jac_base[:, param_locs]
    return masked_jac

inds = np.where(mesh.optimisable_param_bool)[0]
def get_sparsity(params):
    ele, _ = mesh.embed_points(pts, fit_params=params)
    # t0 = time.time()
    stest =  scipy.sparse.csr_array(build_jacobian(ele, mesh.ele_map, inds, len(mesh.optimisable_param_array)))
    # t1 = time.time()
    # print(t1 - t0)
    return stest

_, jacobian_dynamic = jacobian(loc_dist, init_estimate=mesh.optimisable_param_array, sparsity=get_sparsity)

##############################
#now time to drag race:
ts0 = time.time()
for _ in range(10):
    _ = jacobian_dynamic(mesh.optimisable_param_array)
ts1 = time.time()
for _ in range(10):
    _ = jacobian_static(mesh.optimisable_param_array)
ts2 = time.time()
print(f"Dynamic took {(ts1-ts0)/10} seconds, Static took {(ts2 - ts1)/10} seconds")


res = least_squares(fitting_function, x0=mesh.optimisable_param_array, jac=jacobian_static, verbose=2)

mesh.update_from_params(res.x)

mesh.plot(s)

s.add_mesh(pts)
s.show()
                            
