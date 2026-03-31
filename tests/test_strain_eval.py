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
mesh0 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
mesh0.refine(2)

point0 = MeshNode(loc=([0,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([2,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = MeshNode(loc=([0,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = MeshNode(loc=([0,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = MeshNode(loc=([0,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([2,0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = MeshNode(loc=([1,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = MeshNode(loc=([1,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = MeshNode(loc=([1,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5, 0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = MeshNode(loc=([1,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
mesh1 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
mesh1.refine(2)

res = 5
grid = mesh0.xi_grid(res, boundary_points=False)
grid = np.tile(grid, (8,1))
elems = np.repeat([0,1,2,3,4,5,6,7], res**3)
print(elems.shape, grid.shape)
print('evaluating embeddings')
locs = mesh0.evaluate_embeddings_ele_xi_pair(elems, grid)
strains = mesh0.evaluate_strain_ele_xi_pair(elems, grid, mesh1)

s = pv.Plotter()
mesh0.plot(s)
s.add_arrows(locs, strains[:, 0], color='r')
s.add_arrows(locs, strains[:, 1], color='b')
s.add_arrows(locs, strains[:, 2], color='g')
s.show()
