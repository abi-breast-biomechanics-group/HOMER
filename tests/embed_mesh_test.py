import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

from scipy.optimize import least_squares


from HOMER import MeshNode, MeshElement, Mesh, L2Basis, H3Basis
from HOMER.fitting import point_cloud_fit


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


ele, xi = mesh.embed_points(np.array([0.5, 0.5, 0.5]), init_elexi=([0], [(0.25,0.25)]))
e_loc = np.asarray(mesh.evaluate_ele_xi_pair_embeddings(ele, xi))

s = pv.Plotter()
mesh.plot(s)
s.add_mesh(e_loc, color='b', render_points_as_spheres=True, point_size=10)
s.add_mesh(.5 * np.ones(3), color='g', render_points_as_spheres=True, point_size=10)
s.show()
