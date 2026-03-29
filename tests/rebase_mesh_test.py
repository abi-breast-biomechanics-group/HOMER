from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv


point0 = MeshNode(loc=([0,0,1]))
point1 = MeshNode(loc=([0,0,0]))
point2 = MeshNode(loc=([0,1,1]))
point3 = MeshNode(loc=([0,1,0]))
point4 = MeshNode(loc=([1,0,1]))
point5 = MeshNode(loc=([1,0,0]))
point6 = MeshNode(loc=([1,1,1]))
point7 = MeshNode(loc=([1,1,0]))


element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(L1Basis, L1Basis, L1Basis))
mesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
mesh.refine(2)
h3h3h3_mesh = mesh.rebase((H3Basis, H3Basis, H3Basis))
h3h3h3_mesh.plot()

