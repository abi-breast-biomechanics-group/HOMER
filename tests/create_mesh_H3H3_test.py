from HOMER import mesh, mesh_node, mesh_element, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = mesh_node(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point1 = mesh_node(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point2 = mesh_node(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point3 = mesh_node(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

element0 = mesh_element(nodes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))

objMesh = mesh(nodes = [point0, point1, point2, point3], elements = element0)
objMesh.refine(2)
objMesh.plot()
