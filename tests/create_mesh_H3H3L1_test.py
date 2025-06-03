from HOMER import mesh, mesh_node, mesh_element, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = mesh_node(loc=np.array([0,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point1 = mesh_node(loc=np.array([0,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point2 = mesh_node(loc=np.array([0,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point3 = mesh_node(loc=np.array([0,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

# element0 = mesh_element(nodes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))
# objMesh = mesh(nodes = [point0, point1, point2, point3], elements = element0)
# # objMesh = mesh()
# # objMesh.add_node(point0)
# # objMesh.add_node(point1)
# # objMesh.add_node(point2)
# # objMesh.add_node(point3)
# # objMesh.add_element(element0)
# # objMesh.generate_mesh()
# xi_grid = objMesh.xi_grid(res=100)
# objMesh.elem_evals[0] = jax.jit(objMesh.elem_evals[0])
# test_result = objMesh.evaluate_embeddings(element_ids=[0], xis=xi_grid)
# pv.PolyData(test_result).plot(render_points_as_spheres=True)

point4 = mesh_node(loc=np.array([1,0,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point5 = mesh_node(loc=np.array([1,0,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point6 = mesh_node(loc=np.array([1,1,1]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
point7 = mesh_node(loc=np.array([1,1,0]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

element1 = mesh_element(nodes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, L1Basis, H3Basis))
objMesh = mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
xi_grid = objMesh.xi_grid(res=10, dim=3)
test = objMesh.evaluate_embeddings(element_ids=[0], xis=xi_grid)
test_mesh =pv.PolyData(test)
s = pv.Plotter()
s.add_mesh(pv.PolyData(np.array([node.loc for node in objMesh.nodes])), render_points_as_spheres=True, color='r', point_size=10)
s.add_mesh(test_mesh)
s.show()
