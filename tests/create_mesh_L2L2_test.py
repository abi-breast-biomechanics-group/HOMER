from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax
import jax.numpy as jnp


point0 = MeshNode(loc=[0,0,1])
point0_1 = MeshNode(loc=[0,0,0.5])
point1 = MeshNode(loc=[0,0,0])
point2 = MeshNode(loc=[0,1,1])
point2_3 = MeshNode(loc=[0,1,0.5])
point3 = MeshNode(loc=[0,1,0])
point0_2 = MeshNode(loc=[0,0.5,1])
point1_3 = MeshNode(loc=[0,0.5,0])
point_middle = MeshNode(loc=[0.25, 0.5, 0.5])

# element0 = MeshElement(nodes=[0,1,2,3], basis_functions=(L1Basis, L1Basis))
# objMesh = mesh(nodes=[point0, point1, point2, point3], elements = element0)
# objMesh.plot()

element0 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7,8], basis_functions=(L2Basis, L2Basis))
objMesh = Mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0)
# objMesh = mesh(nodes=[point0, point0_2, point0_1, point2, point_middle, point2_3, point1, point1_3, point3], elements = element0)
objMesh.refine(4)
objMesh.plot()

test =  objMesh.evaluate_embeddings.deriv(1, [0], [[0.5,0.5]])
print(test.shape)
print(test)

test =  objMesh.evaluate_embeddings.deriv(2, [1, 3], [[0.25, 0.25], [0.5,0.5]])
print(test.shape)
print(test)
