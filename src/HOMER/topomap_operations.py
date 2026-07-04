import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def global_nodes_from_ele_localnodes(local_points, connectivity, tol=1e-7):
    """
    Builds a global indexing for points of interest across a hypercube mesh.
    
    Args:
        local_points: (N, D) array of local coordinates.
        connectivity: (m, D, 2) array. -1 means no connection.
                      connectivity[e1, d, side] = e2
        tol: Tolerance for floating point coordinate matching.
        
    Returns:
        unique_representatives: (U, 2) array of (element_idx, local_point_idx) 
                                for each unique global point.
        per_simplex_idx: (m, N) array mapping each local point to its global ID.
    """
    m, D, _ = connectivity.shape
    N = local_points.shape[0]
    total_points = m * N
    
    # We will build a list of connected node pairs (edges) for the graph
    edges_i = []
    edges_j = []
    # 1. Find all pairs of local points (p, q) that match across dimension d
    diff = local_points[:, None, :] - local_points[None, :, :]
    for d in range(D):
        # Check the coordinate shift on the connecting dimension
        cond_d = np.abs(diff[:, :, d] - 1.0) < tol
        
        # Check that all other dimensions are perfectly aligned
        cond_other = np.ones((N, N), dtype=bool)
        for d_prime in range(D):
            if d_prime != d:
                cond_other &= (np.abs(diff[:, :, d_prime]) < tol)
                
        # Boolean mask of valid (p, q) pairs, and their indices
        valid_pairs = cond_d & cond_other
        p_indices, q_indices = np.where(valid_pairs)
        
        # 2. Find all elements that are connected on the positive side (side 1) of dim d
        e_indices = np.where(connectivity[:, d, 1] != -1)[0]
        neighbor_indices = connectivity[e_indices, d, 1]
        
        if len(e_indices) > 0 and len(p_indices) > 0:
            # 3. Vectorized creation of global edges 
            # node_i corresponds to point p in element e
            # node_j corresponds to point q in the neighbor element
            node_i = e_indices[:, None] * N + p_indices[None, :]
            node_j = neighbor_indices[:, None] * N + q_indices[None, :]
            
            edges_i.append(node_i.ravel())
            edges_j.append(node_j.ravel())
            
    # 4. Construct the sparse graph
    if edges_i:
        edges_i = np.concatenate(edges_i)
        edges_j = np.concatenate(edges_j)
        graph = coo_matrix(
            (np.ones_like(edges_i), (edges_i, edges_j)), 
            shape=(total_points, total_points)
        )
    else:
        graph = coo_matrix((total_points, total_points))
        
    # 5. Extract Connected Components
    # directed=False ensures that if A connects to B, B connects to A.
    # labels array maps every single local point to a dense sequential global ID (0 to U-1)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )
    
    # (b) Per simplex index into the unique points
    per_simplex_idx = labels.reshape(m, N)
    
    # (a) List of unique points
    # Extract the first global occurrence of each unique point
    _, unique_flat_indices = np.unique(labels, return_index=True)
    
    # Map back to which element and which local point it came from
    unique_elements = unique_flat_indices // N
    unique_local_pts = unique_flat_indices % N
    unique_representatives = np.column_stack((unique_elements, unique_local_pts))
    
    return unique_representatives, per_simplex_idx

def refine_connectivity(connectivity, R):
    """
    Refines an m x D x 2 connectivity array by a length D refinement array R.
    Uses Fortran ordering (dimension 0 changes fastest) for sub-hypercubes.
    
    Parameters:
    connectivity (np.ndarray): Shape (m, D, 2). Values are -1 or neighbor index.
    R (list or np.ndarray): Length D array containing the subdivision factors.
    
    Returns:
    tuple: 
        - new_conn (np.ndarray): Refined connectivity of shape (m * prod(R), D, 2).
        - local_coords (np.ndarray): The Fortran-ordered grid indices inside the 
                                     parent element, shape (m * prod(R), D).
    """
    m, D, _ = connectivity.shape
    R = np.array(R, dtype=int)
    K = np.prod(R)
    
    # --- 1. Map Global IDs (Fortran-order for local grid) ---
    local_ids = np.arange(K).reshape(R, order='F')
    shape_m = [m] + [1] * D
    global_ids = np.arange(m).reshape(shape_m) * K + local_ids
    
    # --- 2. Generate Local Grid Indices ---
    # np.unravel_index gets the (xi0, xi1, ...) coords for flat indices 0 to K-1
    base_coords = np.column_stack(np.unravel_index(np.arange(K), R, order='F'))
    # Tile this for all m parent elements so shape is (m * K, D)
    local_coords = np.tile(base_coords, (m, 1))
    
    # --- 3. Initialize FLAT new connectivity ---
    new_conn = np.full((m * K, D, 2), -1, dtype=int)
    
    # --- 4. Wire connections directly into the flat array ---
    for d in range(D):
        # A. Wire INTERNAL connections
        if R[d] > 1:
            slice_curr = [slice(None)] * (D + 1)
            slice_curr[d + 1] = slice(1, None)
            
            slice_prev = [slice(None)] * (D + 1)
            slice_prev[d + 1] = slice(None, -1)
            
            # Extract the correct flat indices from our F-ordered mapping
            origin_ids = global_ids[tuple(slice_curr)]
            target_ids = global_ids[tuple(slice_prev)]
            
            # Wire face 0 and face 1 simultaneously
            new_conn[origin_ids, d, 0] = target_ids
            new_conn[target_ids, d, 1] = origin_ids
            
        # B. Wire EXTERNAL connections
        
        # Face 0
        e_left = connectivity[:, d, 0]
        valid_left = (e_left != -1)
        if np.any(valid_left):
            slice_target = [slice(None)] * (D + 1)
            slice_target[0] = e_left[valid_left]
            slice_target[d + 1] = R[d] - 1
            
            slice_origin = [slice(None)] * (D + 1)
            slice_origin[0] = valid_left
            slice_origin[d + 1] = 0
            
            origin_ids = global_ids[tuple(slice_origin)]
            target_ids = global_ids[tuple(slice_target)]
            new_conn[origin_ids, d, 0] = target_ids
            
        # Face 1
        e_right = connectivity[:, d, 1]
        valid_right = (e_right != -1)
        if np.any(valid_right):
            slice_target = [slice(None)] * (D + 1)
            slice_target[0] = e_right[valid_right]
            slice_target[d + 1] = 0
            
            slice_origin = [slice(None)] * (D + 1)
            slice_origin[0] = valid_right
            slice_origin[d + 1] = R[d] - 1
            
            origin_ids = global_ids[tuple(slice_origin)]
            target_ids = global_ids[tuple(slice_target)]
            new_conn[origin_ids, d, 1] = target_ids
            
    return new_conn, local_coords
