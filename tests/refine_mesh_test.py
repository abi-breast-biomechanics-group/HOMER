from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax

point0 = MeshNode(loc=[0,0,1], du=[-0.5, -0.5, -0.5], dv=[-0.5, -0.5, 1.5], dw = [0,0,0.5], dudv=[-0.5, 1.5, -0.5], dudw=[1.5, -0.5, -0.5], dvdw=[1.5, 1.5, 1.5], dudvdw=[-0.1, -0.1, -0.1])
point1 = MeshNode(loc=[0,0,0], du=[0,0,0], dv=[0,0,0], dw = [0,0,0], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=[0,1,1], du=[0,0,0], dv=[0,0,0], dw = [0,0,0], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=[0,1,0], du=[0,0,0], dv=[0,0,0], dw = [2,0.5,-0.5], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=[1,0,1], du=[0,0,0], dv=[0,0,0], dw = [1,-0.5,0.5], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=[1,0,0], du=[0,0,0], dv=[0,0,0], dw = [1,-0.5,-0.5], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = MeshNode(loc=[1,1,1], du=[0,0,0], dv=[0,0,0], dw = [1,0.5, 0.5], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = MeshNode(loc=[1,1,0], du=[0,0,0], dv=[0,0,0], dw = [1,0.5,-0.5], dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])

element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
objMesh_refine = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)

s = pv.Plotter()
objMesh.plot(scene=s, node_colour='b', node_size=20)
# objMesh_refine.refine(refinement_factor=2)
# objMesh_refine.refine(refinement_factor=2)
# objMesh_refine.plot(scene=s, node_colour='g', node_size=15)
objMesh_refine.refine(by_xi_refinement=([0, 1/4, 2/4, 3/4, 1], [0, 1/3, 2/3, 1], [0, 1/2, 1]))
# objMesh_refine.refine(by_xi_refinement=([0, 2/4, 1], [0, 1/2, 1], [0, 1/2, 1]))
objMesh_refine.plot(scene=s, node_colour='r', node_size=10)
s.show()

