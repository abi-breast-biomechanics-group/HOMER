from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = MeshNode(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3), id='node_1')
point1 = MeshNode(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3), id=2)
point2 = MeshNode(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3), id='3')
point3 = MeshNode(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3), id=(1,1))

element0 = MeshElement(node_ids=['node_1', 2, '3', (1,1)], basis_functions=(H3Basis, H3Basis), id='test_elem')

objMesh = Mesh(nodes = [point0, point1, point2, point3], elements = element0)
objMesh.refine(2)

objMesh.plot(elem_labels=True)

print(objMesh.element_id_to_ind)
