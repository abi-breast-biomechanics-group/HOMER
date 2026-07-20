import pyvista as pv
from scipy.spatial import KDTree
import numpy as np

from scipy.optimize import least_squares


from HOMER import MeshNode, MeshElement, Mesh, L1Basis, L2Basis, L4Basis, H3Basis, B3Basis, L3Basis
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
mesh.refine(4)

pts = np.random.rand(1000, 3)
pts[:, 0] = 0.6
(ele, xi), res = mesh.embed_points(pts, verbose=3, return_residual=True, iterations=20)
#on this mesh, we can check for convergence by looking for the angle between the normals and the residualts.
normal = mesh.evaluate_normals_ele_xi_pair(ele, xi)
n_normal = normal/np.linalg.norm(normal, axis=-1, keepdims=True)
r_normal = res/np.linalg.norm(res, axis=-1, keepdims=True)
product = np.abs(np.sum(n_normal*r_normal, axis=-1))
print("mean normal similarity: ", np.mean(product), "This should essentially be 1")


# the mesh is close to degenerate, so some node values will fail to converge.
tl = 0
point0 = MeshNode(loc=([0,0,1]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([2,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = MeshNode(loc=([0,0,0]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([0.1,0.1,0.1]),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=([0,1,1]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = (np.zeros(3)*0.1),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=([0,1,0]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([2,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=([1,0,1]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([1,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=([1,0,0]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([1,-0.5,-0.5]),  dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = MeshNode(loc=([1,1,1]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([1,0.5, 0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = MeshNode(loc=([1,1,0]), du=np.zeros(3)*0.1, dv=np.zeros(3)*0.1, dw = ([1,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])


element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
mesh1 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1) #.rebase([L2Basis]*3).rebase([H3Basis]*3)
mesh1.refine(2)
# mesh1.plot()

pts = np.random.rand(1_00_000, 3) * 1.5 - 0.25
# pts = pts[4156]
# pts = pts[1469]

sl = pv.Plotter()
(ele, xi), res = mesh1.embed_points(
    pts,
    # init_elexi=([0], [(0.25,0.25, 0.1)]), 
    verbose=3, 
    return_residual=True,
    iterations=10,
    # iterations=15,
    # vis_max_norm=0.1,
    scene=sl,
    robust_init_est=True, #use a robuse initial estimate because the initial mesh is degenerate
)


#a debug script to catch points sharing! This occurs often if boundaries are improperly set
val, rinds, inv, count = np.unique(np.concatenate((ele[:, None], xi), axis=-1), axis=0, return_index=True, return_inverse=True, return_counts=True)
pts = np.where(count>2)[0]
locs = val[pts]
concern_pts = mesh1.evaluate_embeddings_ele_xi_pair(locs[:, 0].astype(int), locs[:, 1:])
sl.add_mesh(np.array(concern_pts), render_points_as_spheres=True, point_size=25, color='r')
sl.show()
# for idc, c in enumerate(count):
#     if c < 2:
#         continue
#     print(np.where(inv==idc)[0], val[idc], c)


# (ele, xi), res = mesh2.embed_points(
#                             # pts[819],
#                             pts,
#                             # init_elexi=([0], [(0.25,0.25, 0.1)]), 
#                             verbose=3, return_residual=True,
#                             )
