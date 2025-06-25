from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = MeshNode(loc=([0,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([2,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = MeshNode(loc=([0,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=([0,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=([0,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([2,0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=([1,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=([1,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])


element1 = MeshElement(node_indexes=[0,1,2,3,5,5,5,5], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5], elements = element1)
# objMesh.refine(2)

gauss, weights = objMesh.gauss_grid([4,4,4])

pt = objMesh.evaluate_embeddings(np.array([0]), gauss)

s = pv.Plotter()
pts = pv.PolyData(np.array(pt))
pts['weights'] = weights

s = pv.Plotter()
objMesh.plot(s, node_colour='r')
s.add_mesh(pts) 
s.show()
