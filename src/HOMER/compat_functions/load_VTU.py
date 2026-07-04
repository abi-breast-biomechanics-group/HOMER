import pyvista as pv
import numpy as np
from HOMER import Mesh, MeshElement, MeshNode, L3Basis

def get_vtk_l3_hex_ijk_sequence():
    """
    Returns the same list of 64 (i, j, k) grid coordinates from the original code,
    representing the strict topological node ordering of VTK_LAGRANGE_HEXAHEDRON.
    """
    pts = []
    
    # 1. Eight Corners
    pts.extend([(0,0,0), (3,0,0), (3,3,0), (0,3,0), (0,0,3), (3,0,3), (3,3,3), (0,3,3)])

    # 2. Twenty-Four Edge Interiors
    pts.extend([(1,0,0), (2,0,0)])  # Edge 0
    pts.extend([(3,1,0), (3,2,0)])  # Edge 1
    pts.extend([(1,3,0), (2,3,0)])  # Edge 2
    pts.extend([(0,1,0), (0,2,0)])  # Edge 3
    pts.extend([(1,0,3), (2,0,3)])  # Edge 4
    pts.extend([(3,1,3), (3,2,3)])  # Edge 5
    pts.extend([(1,3,3), (2,3,3)])  # Edge 6
    pts.extend([(0,1,3), (0,2,3)])  # Edge 7
    pts.extend([(0,0,1), (0,0,2)])  # Edge 8
    pts.extend([(3,0,1), (3,0,2)])  # Edge 9
    pts.extend([(3,3,1), (3,3,2)])  # Edge 10
    pts.extend([(0,3,1), (0,3,2)])  # Edge 11

    # 3. Twenty-Four Face Interiors
    pts.extend([(0,1,1), (0,2,1), (0,1,2), (0,2,2)])  # Face 0 (-x)
    pts.extend([(3,1,1), (3,2,1), (3,1,2), (3,2,2)])  # Face 1 (+x)
    pts.extend([(1,0,1), (2,0,1), (1,0,2), (2,0,2)])  # Face 2 (-y)
    pts.extend([(1,3,1), (2,3,1), (1,3,2), (2,3,2)])  # Face 3 (+y)
    pts.extend([(1,1,0), (2,1,0), (1,2,0), (2,2,0)])  # Face 4 (-z)
    pts.extend([(1,1,3), (2,1,3), (1,2,3), (2,2,3)])  # Face 5 (+z)

    # 4. Eight Volume Interiors
    for k in [1, 2]:
        for j in [1, 2]:
            for i in [1, 2]:
                pts.append((i, j, k))

    return pts

def get_lex_to_vtk_mapping():
    """
    Creates an inverse lookup table mapping the lexicographical index (0-63) 
    back to VTK's topological node index (0-63).
    """
    pts = get_vtk_l3_hex_ijk_sequence()
    mapping = [0] * 64
    
    for vtk_idx, (i, j, k) in enumerate(pts):
        lex_idx = i + 4 * j + 16 * k
        mapping[lex_idx] = vtk_idx
        
    return mapping

def read_vtu_to_lexmaps(filename):
    """
    Reads a VTU file containing L3 Hexahedrons and reconstructs 
    the point_pool and lexmaps lists.
    """
    mesh = pv.read(filename)
    
    # 1. Recover the global points array
    point_pool = mesh.points
    
    # 2. Get the inverse mapping
    lex_to_vtk_map = get_lex_to_vtk_mapping()
    # 3. Parse the VTK connectivity array
    # PyVista stores cells in a 1D array: [n_points, p0, p1...pn, n_points, p0...]
    cells = mesh.cells
    lexmaps = []
    
    idx = 0
    while idx < len(cells):
        n_nodes = cells[idx]
        
        if n_nodes != 64:
            raise ValueError(f"Expected an L3 Hex element with 64 points. Found {n_nodes}.")
            
        # Extract the node IDs in VTK order for this element
        vtk_connectivity = cells[idx + 1 : idx + 1 + n_nodes]
        
        # Rearrange to Lexicographical order
        lex_connectivity = [0] * 64
        for lex_idx in range(64):
            vtk_idx = lex_to_vtk_map[lex_idx]
            lex_connectivity[lex_idx] = vtk_connectivity[vtk_idx]
            
        lexmaps.append(lex_connectivity)
        
        # Advance the index to the next cell block
        idx += n_nodes + 1
        
    return point_pool, np.array(lexmaps)

def load_L3_vtu_as_HOMER(vtu_file):
    recovered_point_pool, recovered_lexmaps = read_vtu_to_lexmaps(vtu_file)
    mesh_nodes = [MeshNode(pt) for pt in recovered_point_pool]
    mesh_elemens = [MeshElement(node_indexes=pts, basis_functions = [L3Basis]*3) for pts in recovered_lexmaps]
    return Mesh(nodes=mesh_nodes, elements=mesh_elemens)


