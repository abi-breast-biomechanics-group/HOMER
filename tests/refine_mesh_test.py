from HOMER import mesh, mesh_node, mesh_element, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax

point0 = mesh_node(loc=np.array([0,0,1]), du=np.array([-0.5, -0.5, -0.5]), dv=np.array([-0.5, -0.5, 1.5]), dw = np.array([0,0,0.5]), dudv=np.array([-0.5, 1.5, -0.5]), dudw=np.array([1.5, -0.5, -0.5]), dvdw=np.array([1.5, 1.5, 1.5]), dudvdw=np.array([-0.1, -0.1, -0.1]))
point1 = mesh_node(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point2 = mesh_node(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point3 = mesh_node(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([2,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point4 = mesh_node(loc=np.array([1,0,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point5 = mesh_node(loc=np.array([1,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point6 = mesh_node(loc=np.array([1,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5, 0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point7 = mesh_node(loc=np.array([1,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))

element1 = mesh_element(nodes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
objMesh_refine = mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)

s = pv.Plotter()
objMesh.plot(scene=s, node_colour='b', node_size=20)
objMesh_refine.refine(refinement_factor=2)
objMesh_refine.refine(refinement_factor=2)
# objMesh_refine.plot(scene=s, node_colour='g', node_size=15)
# objMesh_refine.refine(by_xi_refinement=([0, 1/4, 2/4, 3/4, 1], [0, 1/3, 2/3, 1], [0, 1/2, 1]))
# objMesh_refine.refine(by_xi_refinement=([0, 2/4, 1], [0, 1/2, 1], [0, 1/2, 1]))
objMesh_refine.plot(scene=s, node_colour='r', node_size=10)
s.show()

