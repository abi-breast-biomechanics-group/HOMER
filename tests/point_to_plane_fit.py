import pyvista as pv
from scipy.spatial import KDTree
import numpy as np
import jax

from scipy.optimize import least_squares


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

point0.fix_parameter('loc')
point1.fix_parameter('loc')
point2.fix_parameter('loc')
point3.fix_parameter('loc')
mesh.generate_mesh()


s = pv.Plotter()
mesh.plot(s, node_colour='g')

pts = np.random.rand(1000, 3)
pts[:, 0] = 0.3

#look at an initial embedding:
# mesh.embed_points(pts, verbose=3)

def loc_dist(params):
    _, res = mesh.embed_points(pts, fit_params=params, return_residual=True)
    return res.flatten()                 

test = jax.jacfwd(loc_dist)
test(mesh.optimisable_param_array)
                            
fitting_function, jacobian_fun = jacobian(loc_dist, init_estimate=mesh.optimisable_param_array)
res = least_squares(fitting_function, x0=mesh.optimisable_param_array, jac=jacobian_fun, verbose=2)
mesh.update_from_params(res.x)

mesh.plot(s)
s.add_mesh(pts)
s.show()
                            
