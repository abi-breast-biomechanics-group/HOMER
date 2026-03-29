from HOMER import Mesh, MeshNode, MeshElement, H3Basis, L1Basis, L3Basis, L2Basis, MeshField
import numpy as np
import pyvista as pv
import math

point0 = MeshNode(loc=([0,0,1]))
point1 = MeshNode(loc=([0,0,0]))
point2 = MeshNode(loc=([0,1,1]))
point3 = MeshNode(loc=([0,1,0]))
point4 = MeshNode(loc=([1,0,1]))
point5 = MeshNode(loc=([1,0,0]))
point6 = MeshNode(loc=([1,1,1]))
point7 = MeshNode(loc=([1,1,0]))
element1 = MeshElement(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(L1Basis, L1Basis, L1Basis))
mesh = Mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
mesh.rebase([H3Basis]*3)

# then we create a data field.
def fibonacci_sphere(n: int, radius: float = 0.5, centre = (0.0, 0, 0)):
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 137.508°
    i = np.arange(n)
    z = radius * (1.0 - 2.0 * i / (n - 1))
    r_xy = radius * np.sqrt(1.0 - (z / radius) ** 2)
    theta = golden_angle * i          # longitude angle
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    return np.vstack((x, y, z)).T + centre

data0 = fibonacci_sphere(300, radius = 0.49, centre=(.5,.5,.5))
data1 = fibonacci_sphere(300, radius = 0.39, centre=(.5,.5,.5))
data2 = fibonacci_sphere(300, radius = 0.29, centre=(.5,.5,.5))
data = np.concat((data0, data1, data2), axis=0)
normal_field = data - (.5,.5,.5)
normal_field = normal_field/np.linalg.norm(normal_field, axis=-1, keepdims=True)
z2field = data[:,-1]
# z2field = np.ones_like(z2field)

# mesh.refine(2)
# get point locations by embedding in the mesh.

mesh.new_field('vec_dir', field_dimension=3, field_locs=data, field_values=normal_field, new_basis=[H3Basis]*3)
mesh.new_field('vec_mag', field_dimension=1, field_locs=data, field_values=z2field, new_basis=[L1Basis]*3)

# mesh['vec_dir'].plot()

s = pv.Plotter()


datum = pv.PolyData(data)
datum['mags'] = z2field

# mesh.plot(s, field_to_draw='vec_mag')
mesh.plot(s, field_to_draw='vec_dir', default_xi_res=6)
s.add_arrows(data, normal_field, mag=0.1)
# s.add_arrows(data, z2field)
# s.add_mesh(datum, render_points_as_spheres=True, point_size=50)
s.show()
