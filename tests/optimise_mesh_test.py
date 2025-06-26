import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

from scipy.optimize import least_squares


from HOMER import MeshNode, MeshElement, Mesh, L2Basis, H3Basis
from HOMER.fitting import point_cloud_fit


#CONSTRUCT THE DATA TO FIT TO
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
objMesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0, jax_compile=True)


surface_xi = objMesh.xi_grid(100)
surface_to_fit = objMesh.evaluate_embeddings([0], surface_xi)



################## CONSTRUCT THE FITTING MESH
point0 = MeshNode(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point1 = MeshNode(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point2 = MeshNode(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point3 = MeshNode(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

point0.fix_parameter(['loc', 'du', 'dv', 'dudv'])
# point0.fix_parameter('loc')
point1.fix_parameter(['loc', 'du', 'dv'])
point2.fix_parameter(['loc', 'du', 'dv', 'dudv'])
point3.fix_parameter(['loc', 'du', 'dv'])
# point1.fix_parameter('loc')
# point2.fix_parameter('loc')
# point3.fix_parameter('loc')

element0 = MeshElement(node_indexes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))
fitting_mesh = Mesh(nodes = [point0, point1, point2, point3], elements = element0)


################## Show the initial fit
s = pv.Plotter()
surf_pts = pv.PolyData(np.array(surface_to_fit))
s.add_mesh(surf_pts)
fitting_mesh.plot(scene=s, mesh_opacity=0.5, res=40)
s.show()

################## Actual fitting code
fitting_fn, jacobian = point_cloud_fit(fitting_mesh, surface_to_fit, compile=True)
init_params = fitting_mesh.optimisable_param_array.copy()

optim = least_squares(fitting_fn, init_params, jac=jacobian, verbose=2)

################## visualise the fit
fitting_mesh.update_from_params(optim.x)
fitting_mesh.plot(mesh_opacity=0.5, res=40)

