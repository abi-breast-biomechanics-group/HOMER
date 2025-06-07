import numpy as np
from HOMER import mesh, mesh_node, mesh_element, H3Basis
import pyvista as pv

point0 = mesh_node(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([2,-0.5,0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))



point1 = mesh_node(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point2 = mesh_node(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([0,0,0]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point3 = mesh_node(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([2,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point4 = mesh_node(loc=np.array([1,0,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point5 = mesh_node(loc=np.array([1,0,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,-0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point6 = mesh_node(loc=np.array([1,1,1]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5, 0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
point7 = mesh_node(loc=np.array([1,1,0]), du=np.zeros(3), dv=np.zeros(3), dw = np.array([1,0.5,-0.5]), dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))


element1 = mesh_element(nodes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
gauss, w = objMesh.gauss_grid([4,4,4])
pt = objMesh.evaluate_embeddings([0], gauss)

s = pv.Plotter()
pts = pv.PolyData(pt)
pts['weights'] = w
s.add_mesh(pts, render_points_as_spheres=True, point_size=30)
objMesh.plot(scene=s)
s.show()
