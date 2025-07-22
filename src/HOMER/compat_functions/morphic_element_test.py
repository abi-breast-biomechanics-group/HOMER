from HOMER.basis_definitions import H3Basis
import morphic
import pyvista as pv
import numpy as np
import scipy

from HOMER import mesh_node, mesh_element, mesh

Mmesh = morphic.Mesh()
Mmesh.auto_add_faces = True



Xn = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
])


deriv_arrays = [np.zeros((3,3)) for _ in range(8)]
for node_idx, node_coordinates in enumerate(Xn):
    if node_idx > 0:
        deriv_arrays[node_idx][:, node_idx - 1] = 1
    Mmesh.add_stdnode(node_idx + 1, np.concatenate((node_coordinates[:, None], deriv_arrays[node_idx].T), axis=1))

node_data_struct = [{k:v for k, v in zip(['du', 'dv', 'dudv'], d)} for d in deriv_arrays]
Mmesh.add_element(1, ['H3', 'H3'], [1, 2, 3, 4])
Mmesh.generate()


homer_nodes = [mesh_node(loc = Xn[idn], **node_data_struct[idn]) for idn in range(4)] 
element = mesh_element(node_indexes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))
homer_mesh = mesh(nodes=homer_nodes, elements=element)

# X, T = mesh.get_faces()

XYZ = np.mgrid[:20,:20]/20

xis = np.column_stack([x.flatten() for x in XYZ])

#   Get node coordinates
node_points = Mmesh.evaluate(element_ids=[1], xi=xis)

#   Visualise mesh
plotter = pv.Plotter()
plotter.add_points(
    node_points, style='points', color='orange',
    point_size=10, label='nodes', render_points_as_spheres=True)

homer_mesh.plot(scene=plotter, node_size=0, labels=True)
for n in homer_mesh.nodes:
    n.plot(plotter)
plotter.show()
