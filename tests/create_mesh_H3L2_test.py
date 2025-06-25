from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = MeshNode(loc=[0,0,1], du=[0,0,0])
point0_1 = MeshNode(loc=[0,0,0.5], du=[0,0,0])
point1 = MeshNode(loc=[0,0,0], du=[0,0,0])
point2 = MeshNode(loc=[0,1,1], du=[0,0,0])
point2_3 = MeshNode(loc=[0.5,1,0.5], du=[0,0,0])
point3 = MeshNode(loc=[0,1,0], du=[0,0,0])

# element0 = MeshElement(nodes=[0,1,2,3], basis_functions=(L1Basis, L1Basis))
# objMesh = mesh(nodes=[point0, point1, point2, point3], elements = element0)
# objMesh.plot()

element0 = MeshElement(node_indexes=[0,1,2,3,4,5], basis_functions=(L2Basis, H3Basis))
objMesh = Mesh(nodes=[point0, point0_1, point1, point2, point2_3, point3], elements = element0)
objMesh.refine(refinement_factor=2)

s = pv.Plotter()
for n in objMesh.nodes:
    n.plot(s)

objMesh.plot(s, node_colour='r')
s.show() 


