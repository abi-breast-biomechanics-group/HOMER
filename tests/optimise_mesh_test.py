from HOMER import mesh_node, mesh_element, mesh, L2Basis
import pyvista as pv
from scipy.spatial import KDTree


point0 = mesh_node(loc=[0,0,1])
point0_1 = mesh_node(loc=[0,0,0.5])
point1 = mesh_node(loc=[0,0,0])
point2 = mesh_node(loc=[0,1,1])
point2_3 = mesh_node(loc=[0,1,0.5])
point3 = mesh_node(loc=[0,1,0])
point0_2 = mesh_node(loc=[0,0.5,1])
point1_3 = mesh_node(loc=[0,0.5,0])
point_middle = mesh_node(loc=[0.5, 0.5, 0.5])

# element0 = mesh_element(nodes=[0,1,2,3], basis_functions=(L1Basis, L1Basis))
# objMesh = mesh(nodes=[point0, point1, point2, point3], elements = element0)
# objMesh.plot()

element0 = mesh_element(nodes=[0,1,2,3,4,5,6,7,8], basis_functions=(L2Basis, L2Basis))
objMesh = mesh(nodes=[point0, point0_1, point1, point0_2, point_middle, point1_3, point2, point2_3, point3], elements = element0)


surface_xi = objMesh.xi_grid(30)
surface_to_fit = objMesh.evaluate_embeddings([0], surface_xi)
pv.PolyData(surface_to_fit).plot() 



def create_fitting_function(mesh:mesh, data, res=20):
    """
        An example that creates a fitting problem for a mesh.
        This is a fit that connects a point to a surface.
    """

    data_tree = KDTree(data)
    eval_points = mesh.xi_grid(res)
    # sob_points = mesh.gauss_grid([4, 4])
    #
    def fitting_function(params):
        locs = mesh.evaluate_embeddings([0], xis=eval_points, params=params)
        dists, _ = data_tree.query(locs, k=1)
        return dists
    
    return fitting_function
