from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax

from matplotlib import pyplot as plt


point0 = MeshNode(loc=[0,0,1])
point0_1 = MeshNode(loc=[0,0,0.5])
point1 = MeshNode(loc=[0,0,0])
point2 = MeshNode(loc=[0,1,1])
point2_3 = MeshNode(loc=[0,1,0.5])
point3 = MeshNode(loc=[0,1,0])
point0_2 = MeshNode(loc=[0,0.5,1])
point1_3 = MeshNode(loc=[0,0.5,0])
point_middle = MeshNode(loc=[0, 0.5, 0.5])

element0 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7,8], basis_functions=(L2Basis, L2Basis))
mesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0)

n_grid = 1
grid = np.linspace(0,1, n_grid*3+1)
grid_p = grid + 0.5/n_grid*3
zoff = 0.25

unit_0 = np.array([
    [0, 0],
    [0, 1/3],
    [1/2, 1/3 + 1/6], 
    [1/2, 2/3 + 1/6],
    [0, 1],
])
unit_1 = np.array([
    [1, 0],
    [1, 1/3],
    [1/2, 1/3 + 1/6], 
    [1/2, 2/3 + 1/6],
    [1, 1],
])

base_line = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
])

combined_ls = np.concatenate((unit_0[None], unit_1[None]))

def make_tiling(xn, yn):
    base_unit = (combined_ls / [[[xn, yn]]])
    up_grid = np.column_stack([a.flatten() for a in np.mgrid[:xn, :yn]]) / [[xn, yn]]
    long_grid = base_unit[None] + up_grid[:, None, None, :]
    
    shape_mat = np.arange(np.prod(long_grid.shape[:-1])).reshape(long_grid.shape[:-1]) 
    ind_mat = shape_mat[:, :, 0][..., None, None] + base_line[None, None]
    connectivity = np.ones(ind_mat.shape[:-1] + (1,)) * 2
    lmat = np.concatenate((connectivity, ind_mat), axis=-1).ravel()

    return long_grid.reshape(-1, 2), lmat


evals, lines = make_tiling(2, 1)


l0 = np.array(mesh.evaluate_embeddings([0], evals) )

l = pv.PolyData(l0, lines=lines.astype(int))
sml = pv.PolyData(l0[:5], lines=lines[:4*3].astype(int))
pp = pv.PolyData(l0[:5])
# l.lines = lines.astype(int)

cmap = plt.get_cmap('viridis')
cols = [cmap(i) for i in range(5)]

sml['col'] = cols
pp['col'] = cols

s = pv.Plotter()
mesh.plot(s, mesh_opacity=0)
s.add_mesh(l, line_width=1, color='k', style='wireframe', point_size=0)
s.add_mesh(sml, render_lines_as_tubes=True, line_width=40,)
s.add_mesh(pp, render_points_as_spheres=True, point_size=40)
s.show()





