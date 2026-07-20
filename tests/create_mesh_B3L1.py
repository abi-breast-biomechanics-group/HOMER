from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis, B3Basis
import numpy as np
import pyvista as pv
import jax
import jax.numpy as jnp


# point0 = MeshNode(loc=[0,0,0.1])
# point1 = MeshNode(loc=[0,0,0.2])
# point2 = MeshNode(loc=[0,0,0.3])
# point3 = MeshNode(loc=[0,0,0.4])
# point4 = MeshNode(loc=[0,0,0.5])
#
# point5 = MeshNode(loc=[0,1,0.1])
# point6 = MeshNode(loc=[0,1,0.2])
# point7 = MeshNode(loc=[0,1,0.3])
# point8 = MeshNode(loc=[0,1,0.4])
# point9 = MeshNode(loc=[0,1,0.5])
#
# ele0 = MeshElement(
#     node_indexes=[0,1,2,3,5,6,7,8],
#     # node_indexes=[0,5,1,6,2,7,3,8],
#     basis_functions=(B3Basis, L1Basis)
# )
# ele1 = MeshElement(
#     node_indexes=[1,2,3,4,6,7,8,9], 
#     # node_indexes=[1,6,2,7,3,8,4,9], 
#     basis_functions=(B3Basis, L1Basis)
# )
# mesh = Mesh(nodes=[point0, point1, point2, point3, point4, point5, point6, point7, point8, point9],
#             elements = [
#             ele0, 
#             ele1,
#             ])
# mesh.plot()

point0 = MeshNode(loc=[0,0,1])
point0_1 = MeshNode(loc=[0,0,0.5])
point1 = MeshNode(loc=[0,0,0])
point2 = MeshNode(loc=[0,1,1])
point2_3 = MeshNode(loc=[0,1,0.5])
point3 = MeshNode(loc=[0,1,0])
point0_2 = MeshNode(loc=[0,0.5,1])
point1_3 = MeshNode(loc=[0,0.5,0])
point_middle = MeshNode(loc=[0.5, 0.5, 0.5])

# element0 = MeshElement(nodes=[0,1,2,3], basis_functions=(L1Basis, L1Basis))
# objMesh = mesh(nodes=[point0, point1, point2, point3], elements = element0)
# objMesh.plot()

ele2 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7,8], basis_functions=(L2Basis, L2Basis))
target_mesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = ele2)
target_mesh = target_mesh.rebase([L3Basis, L3Basis])
prams = target_mesh.optimisable_param_array.copy().reshape(-1,3)
# prams[0] += 0.5
# prams[-1] += 0.5
target_mesh.update_from_params(prams.ravel())
target_mesh.refine(3)
target_mesh.plot()

new_mesh = target_mesh.rebase([B3Basis, B3Basis], res=20)
new_mesh.refine(3)
new_mesh.plot()


point0 = MeshNode(loc=([0,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([2,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = MeshNode(loc=([0,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=([0,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]),        dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=([0,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([2,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=([1,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=([1,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,-0.5]),  dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = MeshNode(loc=([1,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5, 0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = MeshNode(loc=([1,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5,-0.5]),   dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])


element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
mesh1 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase([B3Basis]*3)
mesh1.refine(4)
mesh1.plot()

