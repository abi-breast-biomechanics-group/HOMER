import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

from scipy.optimize import least_squares


from HOMER import MeshNode, MeshElement, Mesh, L1Basis, L2Basis, L4Basis, H3Basis
from HOMER.fitting import point_cloud_fit


seed = np.random.seed(42)
point0 = MeshNode(loc=np.array([0,0,1]))
point1 = MeshNode(loc=np.array([0,0,0]))
point2 = MeshNode(loc=np.array([0,1,1]))
point3 = MeshNode(loc=np.array([0,1,0]))
element0 = MeshElement(node_indexes=[0,1,2,3], basis_functions=[L1Basis]*2)
mesh = Mesh(nodes = [point0, point1, point2, point3], elements = element0).rebase([H3Basis]*2)

# mesh.embed_points([0.1, -0.1, -0.1], init_elexi=([0], [(1, 0.5)]), verbose=3)
mesh.refine(2)

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

pts = np.random.rand(1000, 3)
pts[:, 0] = 0.6
ele, xi = mesh.embed_points(pts, verbose=3)


point0 = MeshNode(loc=([0,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([2,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = MeshNode(loc=([0,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=([0,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=([0,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([2,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=([1,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=([1,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,-0.5]),  dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = MeshNode(loc=([1,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5, 0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = MeshNode(loc=([1,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])


element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))

mesh1 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase([H3Basis]*3)

pts = np.random.rand(1000, 3) * 1.5 - 0.25


# pts = np.array( [
#                [-0.25, 0.5, 0.5],
#                [+1.25, 0.5, 0.5],
#                [0.5, -0.25, 0.5],
#                [0.5, 1.25, 0.5],
#                [0.5, 0.5, -0.25],
#                [0.5, 0.5, 1.5],
#                [0.5, 0.5, 0.5],
# ])


(ele, xi), res = mesh1.embed_points(
                            # pts[819],
                            pts,
                            # init_elexi=([0], [(0.25,0.25, 0.1)]), 
                            verbose=3, return_residual=True,
                            iterations=20,
                            )

# mesh2 = mesh1.rebase([L4Basis]*3)
#  
# pts = np.random.rand(1000, 3) * 1.5 - 0.25
#
#
#
#
# ind = np.argmax(np.linalg.norm(res, axis=-1))
# print(ind)



# (ele, xi), res = mesh2.embed_points(
#                             # pts[819],
#                             pts,
#                             # init_elexi=([0], [(0.25,0.25, 0.1)]), 
#                             verbose=3, return_residual=True,
#                             )
