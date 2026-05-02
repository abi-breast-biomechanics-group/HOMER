import pyvista as pv
import time
from scipy.spatial import KDTree
import numpy as np
import jax
import scipy

from jaxopt import LevenbergMarquardt

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


#by assuming that the element embedding is constant, we can then create a fast sparse jacobian
#for ease, we create this as dense and pass to the jacobian function, but large problems will need a sparse sparsity as well.
jac_base = np.zeros((np.prod(pts.shape), mesh.true_param_array.shape[0]))
out_ind = 0
for e in eles:
    local_map = mesh.ele_map[e]
    jac_base[out_ind:out_ind+3, local_map] = 1; 
    out_ind+=3
jac_base = jac_base[:, mesh.optimisable_param_bool]
jac_sparse = jax.experimental.sparse.BCOO.fromdense(jac_base)
# jac_sparse = None


fitting_function, jacobian_fun = jacobian(loc_dist, init_estimate=mesh.optimisable_param_array, sparsity=jac_sparse)

res = least_squares(fitting_function, x0=mesh.optimisable_param_array, jac=jacobian_fun, verbose=2)

mesh.update_from_params(res.x)

mesh.plot(s)

s.add_mesh(pts)
s.show()
                            
