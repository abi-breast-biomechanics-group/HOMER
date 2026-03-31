from copy import deepcopy
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
mesh0 = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase([H3Basis]*3)

mesh_def = deepcopy(mesh0)
grid = mesh0.xi_grid(res=10)
elems = np.zeros(grid.shape[0], dtype=int)

wlocs = mesh0.evaluate_embeddings_in_every_element(grid)
#apply the strain mapping
wlocs_def = np.stack((wlocs[:, 0], wlocs[:, 1] + wlocs[:, 0] * 0.1, wlocs[:, 2]**2), axis=1)

#fit the deformed mesh to this strain field (it can represent it perfectly)
wmat = mesh_def.get_xi_weight_mat(elems, grid)
mesh_def.linear_fit(wlocs_def, weight_mat=wmat)

# evaluate the strain field over this deformation
strains = mesh0.evaluate_strain_in_every_element(grid, mesh_def)

s = pv.Plotter()
mesh0.plot(s)
mesh_def.plot(s, node_colour='g')
s.add_mesh(np.array(wlocs_def), render_points_as_spheres=True)
s.add_arrows(wlocs, strains[:, 0], color='r')
s.add_arrows(wlocs, strains[:, 1], color='g')
s.add_arrows(wlocs, strains[:, 2], color='b')
s.show()
