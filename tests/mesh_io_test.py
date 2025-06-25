from HOMER import MeshNode, MeshElement, Mesh, H3Basis
from HOMER.io import save_mesh, load_mesh
import numpy as np

point0 = MeshNode(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([2,-0.5,0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point1 = MeshNode(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point2 = MeshNode(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point3 = MeshNode(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([2,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point4 = MeshNode(loc=np.array([1,0,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point5 = MeshNode(loc=np.array([1,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point6 = MeshNode(loc=np.array([1,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5, 0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point7 = MeshNode(loc=np.array([1,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))

element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)

save_mesh(objMesh, "bin/test_mesh.json")
new_mesh = load_mesh("bin/test_mesh.json")


