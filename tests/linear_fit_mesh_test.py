import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

from scipy.optimize import least_squares


from HOMER import MeshNode, MeshElement, Mesh, L2Basis, H3Basis
from HOMER.fitting import point_cloud_fit
from matplotlib import pyplot as plt


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
target_mesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0, jax_compile=True)
target_mesh.refine(2)


################## CONSTRUCT THE FITTING MESH
point0 = MeshNode(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point1 = MeshNode(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point2 = MeshNode(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point3 = MeshNode(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
element0 = MeshElement(node_indexes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))
fitting_mesh = Mesh(nodes = [point0, point1, point2, point3], elements = element0)
fitting_mesh.refine(2)

#node locs 
grid = fitting_mesh.xi_grid(20)
grid = np.tile(grid, (4,1))
elems = np.repeat([0,1,2,3], 20**2)

target_points = target_mesh.evaluate_ele_xi_pair_embeddings(elems, grid)

################## Show the initial fit
s = pv.Plotter()
target_mesh.plot(scene=s)
fitting_mesh.plot(scene=s, mesh_opacity=0.5)
s.add_mesh(np.asarray(target_points), render_points_as_spheres=True, color='g')
s.show()




wmat = fitting_mesh.get_xi_weight_mat(elems, grid)
fitting_mesh.linear_fit(target_points, weight_mat=wmat)
fitting_mesh.plot()

breakpoint()
