from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = MeshNode(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point1 = MeshNode(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point2 = MeshNode(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point3 = MeshNode(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point4 = MeshNode(loc=np.array([1,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point5 = MeshNode(loc=np.array([1,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point6 = MeshNode(loc=np.array([1,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point7 = MeshNode(loc=np.array([1,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, L1Basis, H3Basis))

objMesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)

print("volume", objMesh.get_volume())

objMesh.refine(2)
objMesh.plot(node_colour='r')

print("volume", objMesh.get_volume())
