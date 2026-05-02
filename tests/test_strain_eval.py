from copy import deepcopy
from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


point0 = MeshNode(loc=([0,0,0]))
point1 = MeshNode(loc=([1,0,0]))
point2 = MeshNode(loc=([0,1,0]))
point3 = MeshNode(loc=([1,1,0]))
point4 = MeshNode(loc=([0,0,1]))
point5 = MeshNode(loc=([1,0,1]))
point6 = MeshNode(loc=([0,1,1]))
point7 = MeshNode(loc=([1,1,1]))
element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(L1Basis, L1Basis, L1Basis))
mesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1).rebase([H3Basis]*3)

mesh_def = deepcopy(mesh)
grid = mesh.xi_grid(res=10)
elems = np.zeros(grid.shape[0], dtype=int)

wlocs = mesh.evaluate_embeddings_in_every_element(grid)
#apply the strain mapping
wlocs_def = np.stack((wlocs[:, 0], wlocs[:, 1] + wlocs[:, 0] * 0.1, 0 * wlocs[:,2] + 1 * wlocs[:, 2]**2), axis=1)

#fit the deformed mesh to this strain field (it can represent it perfectly)
wmat = mesh_def.get_xi_weight_mat(elems, grid)
mesh_def.linear_fit(wlocs_def, weight_mat=wmat)

# evaluate the strain field over this deformation
grid = mesh.xi_grid(res=5)
strains = mesh.evaluate_strain_in_every_element(grid, mesh_def)


mesh.plot_strains(eles = np.zeros(grid.shape[0]), xis = grid, strains = strains)


slocs = mesh.evaluate_embeddings(0, grid)
svecs = mesh_def.evaluate_jacobians(0, grid)

print(strains[0])
print(svecs[0])

s = pv.Plotter()
mesh.plot(s, mesh_opacity=0.1)
mesh_def.plot(s, node_colour='g', mesh_opacity=0.1)
s.add_mesh(np.array(wlocs_def), render_points_as_spheres=True)
s.add_arrows(slocs, strains[:, 0, 0][:, None] * svecs[:, 0], color='r')
s.add_arrows(slocs, strains[:, 1, 1][:, None] * svecs[:, 1], color='g')
s.add_arrows(slocs, strains[:, 2, 2][:, None] * svecs[:, 2], color='b')
# s.add_arrows(slocs, svecs[:,:,0], color='r')
# s.add_arrows(slocs, svecs[:,:,1], color='g')
# s.add_arrows(slocs, svecs[:,:,2], color='b')
s.show()

plt.plot(np.linspace(0,1,10), 2 * np.linspace(0,1,10)**2 - 1/2)
plt.plot(slocs[:, 2], strains[:, 2, 2], '.')
plt.show()
