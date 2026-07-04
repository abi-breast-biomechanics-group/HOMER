"""
utils.py – Miscellaneous utility functions for HOMER.

Provides small helper functions used internally across the library:

* :func:`jax_aknn` – JAX approximate k-nearest-neighbour search.
* :func:`block_diagonal_jacobian` – construct a block-diagonal sparse matrix.
* :func:`all_pairings` – Fortran-ordered Cartesian product of lists.
* :func:`h_tform` – homogeneous 4 × 4 rigid-body transform for point arrays.
* :func:`vol_tet`, :func:`vol_hexahedron` – geometric volume computations.
* :func:`make_tiling` – generate a hexagonal tiling for surface visualisation.
"""

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import itertools
from copy import copy
from scipy.sparse import csr_array
import jax
import functools

def all_pairings(*lists):
    return [t[::-1] for t in itertools.product(*reversed(copy(lists)))]

def validate_and_extract_topology_2d(m):
    """
    Validates if the mapping array m represents a valid 2D grid homotopic to a square,
    and returns the grid shape and mapping tables.
    """
    m = np.asarray(m)
    num_elements = m.shape[0]

    # 1. Find the origin element (must have -1 on the start faces of both axes)
    is_origin = np.all(m[:, :, 0] == -1, axis=1)
    origins = np.where(is_origin)[0]

    if len(origins) != 1:
        raise ValueError(f"Not homotopic to a square: Found {len(origins)} origin elements (expected 1).")
    origin = origins[0]

    # 2. Determine grid dimensions (N0, N1)
    N = np.zeros(2, dtype=int)
    for axis in range(2):
        curr = origin
        while curr != -1:
            N[axis] += 1
            curr = m[curr, axis, 1]

    if N[0] * N[1] != num_elements:
        raise ValueError(f"Not homotopic to a square: Grid bounds {N} mismatch total elements {num_elements}.")

    # 3. Build lookup tables
    grid_to_e = np.full(N, -1, dtype=int)
    e_to_grid = np.full((num_elements, 2), -1, dtype=int)

    # Traverse and populate the expected grid
    for i in range(N[0]):
        for j in range(N[1]):
            if i == 0 and j == 0:
                e = origin
            elif j > 0:
                e = m[grid_to_e[i, j-1], 1, 1]
            else:
                e = m[grid_to_e[i-1, 0], 0, 1]

            if e == -1 or e_to_grid[e, 0] != -1:
                raise ValueError("Not homotopic to a square: Cycle or premature boundary detected.")

            grid_to_e[i, j] = e
            e_to_grid[e] = [i, j]

    # 4. Rigorous verification of all edges
    for i in range(N[0]):
        for j in range(N[1]):
            e = grid_to_e[i, j]
            expected = [
                [grid_to_e[i-1, j] if i > 0 else -1, grid_to_e[i+1, j] if i < N[0]-1 else -1],
                [grid_to_e[i, j-1] if j > 0 else -1, grid_to_e[i, j+1] if j < N[1]-1 else -1]
            ]
            if not np.array_equal(m[e], expected):
                raise ValueError(f"Not homotopic to a square: Neighbor mismatch at element {e}.")

    return N, grid_to_e, e_to_grid

def build_square_mappings(m):
    """
    Consumes mapping array `m` and returns JAX-compatible forward and inverse mapping functions for 2D.
    """
    # Validate and build discrete mappings purely in NumPy first
    N_np, grid_to_e_np, e_to_grid_np = validate_and_extract_topology_2d(m)
    
    # Cast necessary arrays to JAX format
    N = jnp.array(N_np, dtype=jnp.float32)
    N_ints = jnp.array(N_np, dtype=jnp.int32)
    grid_to_e = jnp.array(grid_to_e_np, dtype=jnp.int32)
    e_to_grid = jnp.array(e_to_grid_np, dtype=jnp.float32)

    @jax.jit
    def macro_to_local(uv):
        """
        Maps [u, v] coordinates in the macro-square [0-1, 0-1]
        to the local element index (e) and its local [x, y] coordinates.
        """
        # Ensure uv is clamped to the valid domain
        uv_clamped = jnp.clip(uv, 0.0, 1.0)
        
        # Scale by grid size to find macro indices
        scaled = uv_clamped * N
        
        # Determine 2D grid index (handling the 1.0 boundary edge case)
        indices = jnp.floor(scaled).astype(jnp.int32)
        indices = jnp.clip(indices, 0, N_ints - 1)
        
        # Find exact local coordinates [x, y] inside the element
        local_xy = scaled - indices
        
        # Fetch the element index mapping
        e = grid_to_e[indices[:, 0], indices[:, 1]]
        
        return e, local_xy

    @jax.jit
    def local_to_macro(e, local_xy):
        """
        Maps an element index (e) and its local [x, y] coordinates [0-1, 0-1]
        back to the macro-square [u, v] coordinates.
        """
        grid_pos = e_to_grid[e]
        
        # Add local offset and normalize back to [0, 1] range
        uv = (grid_pos + local_xy) / N
        return uv

    return macro_to_local, local_to_macro

def validate_and_extract_topology(m):
    """
    Validates if the mapping array m represents a valid grid homotopic to a cube,
    and returns the grid shape and mapping tables.
    """
    m = np.asarray(m)
    num_elements = m.shape[0]

    # 1. Find the origin element (must have -1 on the start faces of all 3 axes)
    is_origin = np.all(m[:, :, 0] == -1, axis=1)
    origins = np.where(is_origin)[0]

    if len(origins) != 1:
        raise ValueError(f"Not homotopic to a cube: Found {len(origins)} origin elements (expected 1).")
    origin = origins[0]

    # 2. Determine grid dimensions (N0, N1, N2)
    N = np.zeros(3, dtype=int)
    for axis in range(3):
        curr = origin
        while curr != -1:
            N[axis] += 1
            curr = m[curr, axis, 1]

    if N[0] * N[1] * N[2] != num_elements:
        raise ValueError(f"Not homotopic to a cube: Grid bounds {N} mismatch total elements {num_elements}.")

    # 3. Build lookup tables
    grid_to_e = np.full(N, -1, dtype=int)
    e_to_grid = np.full((num_elements, 3), -1, dtype=int)

    # Traverse and populate the expected grid
    for i in range(N[0]):
        for j in range(N[1]):
            for k in range(N[2]):
                if i == 0 and j == 0 and k == 0:
                    e = origin
                elif k > 0:
                    e = m[grid_to_e[i, j, k-1], 2, 1]
                elif j > 0:
                    e = m[grid_to_e[i, j-1, 0], 1, 1]
                else:
                    e = m[grid_to_e[i-1, 0, 0], 0, 1]

                if e == -1 or e_to_grid[e, 0] != -1:
                    raise ValueError("Not homotopic to a cube: Cycle or premature boundary detected.")

                grid_to_e[i, j, k] = e
                e_to_grid[e] = [i, j, k]

    # 4. Rigorous verification of all faces
    for i in range(N[0]):
        for j in range(N[1]):
            for k in range(N[2]):
                e = grid_to_e[i, j, k]
                expected = [
                    [grid_to_e[i-1, j, k] if i > 0 else -1, grid_to_e[i+1, j, k] if i < N[0]-1 else -1],
                    [grid_to_e[i, j-1, k] if j > 0 else -1, grid_to_e[i, j+1, k] if j < N[1]-1 else -1],
                    [grid_to_e[i, j, k-1] if k > 0 else -1, grid_to_e[i, j, k+1] if k < N[2]-1 else -1]
                ]
                if not np.array_equal(m[e], expected):
                    raise ValueError(f"Not homotopic to a cube: Neighbor mismatch at element {e}.")

    return N, grid_to_e, e_to_grid

def build_cube_mappings(m):
    """
    Consumes mapping array `m` and returns JAX-compatible forward and inverse mapping functions.
    """
    # Validate and build discrete mappings purely in NumPy first
    N_np, grid_to_e_np, e_to_grid_np = validate_and_extract_topology(m)
    
    # Cast necessary arrays to JAX format
    N = jnp.array(N_np, dtype=jnp.float32)
    N_ints = jnp.array(N_np, dtype=jnp.int32)
    grid_to_e = jnp.array(grid_to_e_np, dtype=jnp.int32)
    e_to_grid = jnp.array(e_to_grid_np, dtype=jnp.float32)

    @jax.jit
    def macro_to_local(uvw):
        """
        Maps [u, v, w] coordinates in the macro-cube [0-1, 0-1, 0-1]
        to the local element index (e) and its local [x, y, z] coordinates.
        """
        # Ensure uvw is clamped to the valid domain
        uvw_clamped = jnp.clip(uvw, 0.0, 1.0)
        
        # Scale by grid size to find macro indices
        scaled = uvw_clamped * N
        
        # Determine 3D grid index (handling the 1.0 boundary edge case)
        indices = jnp.floor(scaled).astype(jnp.int32)
        indices = jnp.clip(indices, 0, N_ints - 1)
        
        # Find exact local coordinates [x, y, z] inside the element
        local_xyz = scaled - indices
        
        # Fetch the element index mapping
        e = grid_to_e[indices[0], indices[1], indices[2]]
        # breakpoint()
        
        return e, local_xyz

    @jax.jit
    def local_to_macro(e, local_xyz):
        """
        Maps an element index (e) and its local [x, y, z] coordinates [0-1, 0-1, 0-1]
        back to the macro-cube [u, v, w] coordinates.
        """
        grid_pos = e_to_grid[e]
        
        # Add local offset and normalize back to [0, 1] range
        uvw = (grid_pos + local_xyz) / N
        return uvw

    return macro_to_local, local_to_macro

@jax.jit
def spherical_to_hex_surface(angles):
    """
    Maps (theta, phi) spherical coordinates to the surface of a 
    hexahedral element with coordinates in [0, 1].
    
    Args:
        angles: A JAX array of shape (..., 2) containing (theta, phi).
                theta is the polar angle [0, pi].
                phi is the azimuthal angle [0, 2pi).
                
    Returns:
        A JAX array of shape (..., 3) containing the mapped (xi, eta, zeta)
        coordinates on the surface of the [0, 1]^3 hexahedron.
    """
    # 1. Unpack angles
    theta = angles[..., 0]
    phi = angles[..., 1]
    
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    
    points = jnp.stack([x, y, z], axis=-1)
    max_abs = jnp.max(jnp.abs(points), axis=-1, keepdims=True)
    cube_coords = points / max_abs
    xi_coords = (cube_coords + 1.0) / 2.0
    
    return xi_coords

@jax.jit
def hex_surface_to_spherical(xi_coords):
    """
    Maps coordinates on the surface of a [0, 1]^3 hexahedral element 
    back to (theta, phi) spherical coordinates.
    
    Args:
        xi_coords: A JAX array of shape (..., 3) containing (xi, eta, zeta)
                   coordinates on the surface of the [0, 1]^3 hexahedron.
                   
    Returns:
        A JAX array of shape (..., 2) containing (theta, phi).
            theta is the polar angle [0, pi].
            phi is the azimuthal angle [0, 2pi).
    """
    # 1. Reverse the mapping from [0, 1]^3 back to the [-1, 1]^3 cube centered at origin
    cube_coords = xi_coords * 2.0 - 1.0
    
    x = cube_coords[..., 0]
    y = cube_coords[..., 1]
    z = cube_coords[..., 2]
    
    # 2. Compute the radial distance (norm) of the vectors on the cube surface
    r = jnp.linalg.norm(cube_coords, axis=-1)
    
    # 3. Recover theta (polar angle)
    # Clip z/r to [-1.0, 1.0] to prevent NaN values in arccos due to floating-point imprecision
    theta = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))
    
    # 4. Recover phi (azimuthal angle)
    phi = jnp.arctan2(y, x)
    
    # jnp.arctan2 returns values in [-pi, pi].
    # Wrap them using modulo to match the original [0, 2pi) domain.
    phi = jnp.mod(phi, 2.0 * jnp.pi)
    
    return jnp.stack([theta, phi], axis=-1)

def skew_symmetric(w):
    """Returns the 3x3 skew-symmetric matrix of a 3D vector."""
    return jnp.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ])

@jax.jit
def rodrigues_exp(w):
    """
    Computes the SO(3) matrix exponential using Rodrigues' formula.
    
    Args:
        w: A 3D array representing the rotation vector (axis * angle).
           The direction is the axis of rotation, and the L2 norm is the angle.
           
    Returns:
        A 3x3 rotation matrix.
    """
    theta2 = jnp.sum(w**2)
    theta = jnp.sqrt(jnp.maximum(theta2, 1e-12))
    
    K = skew_symmetric(w)
    K2 = K @ K
    I = jnp.eye(3)
    
    is_small = theta < 1e-4
    A = jnp.where(
        is_small, 
        1.0 - theta2 / 6.0, 
        jnp.sin(theta) / theta
    )
    B = jnp.where(
        is_small, 
        0.5 - theta2 / 24.0, 
        (1.0 - jnp.cos(theta)) / theta2
    )
    R = I + A * K + B * K2
    return R

def surface_normal_mapping(mesh, eles, xis, derivs):
    """A default mapping function that can be used for evaluation of strain over 2D surfaces."""
    normal = mesh.evaluate_normals(eles, xis)
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return jnp.concatenate((normal[..., None], derivs), axis=-1)

def spheres_to_polydata(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """
    Build a single PolyData from many 'sphere' instances.
    The sphere doesn't actually need to be spherical, just keeps similar connectvitiy.

    Parameters
    ----------
    verts : np.ndarray, shape (M, N, 3)
        M spheres, each with N vertices (x, y, z).
    faces : np.ndarray, shape (F,)
        Shared connectivity in PyVista flat format: [3, i, j, k, 3, ...].
        Must be triangles (all polygons size 3).

    Returns
    -------
    pv.PolyData
        A single merged mesh representing all M spheres.
    """
    M, N, _ = verts.shape

    # Reshape to (num_triangles, 4) so we can offset only the index columns
    face_block = faces.reshape(-1, 4)       # [[3, i, j, k], ...]
    offsets = (np.arange(M) * N).reshape(M, 1, 1)          # (M, 1, 1)
    face_block_tiled = np.tile(face_block, (M, 1, 1))       # (M, F, 4)
    face_block_tiled[:, :, 1:] += offsets  # shift only i,j,k — leave the '3' alone

    all_faces = face_block_tiled.reshape(-1)
    all_verts = verts.reshape(-1, 3)

    return pv.PolyData(all_verts, all_faces)

@functools.partial(jax.jit, static_argnames=["k"])
def jax_aknn(d0, d1, k):
    """
    Jax implementation of approximate nearest neighbours. 
    Trust in jax that it's actualy not as inefficient as it appears!
    """
    test_data = jax.numpy.linalg.norm(d0[:, None] - d1[None, :], axis=-1)
    # print(test_data.shape)
    p0, p1 = jax.lax.approx_min_k(test_data, reduction_dimension=1, k=k)
    return p0, p1

def block_diagonal_jacobian(n: int, m: int, num_blocks: int) -> csr_array:
    """
    Build a block-diagonal sparse matrix with `num_blocks` dense blocks,
    each of shape (n, m), filled with placeholder 1s.

    Parameters
    ----------
    n           : number of rows per block
    m           : number of columns per block
    num_blocks  : number of blocks along the diagonal

    Returns
    -------
    csr_array of shape (n * num_blocks, m * num_blocks)
    """
    nnz = n * m * num_blocks

    block_rows, block_cols = np.mgrid[0:n, 0:m]
    block_rows = block_rows.ravel()
    block_cols = block_cols.ravel()

    k = np.repeat(np.arange(num_blocks), n * m)
    rows = k * n + np.tile(block_rows, num_blocks)
    cols = k * m + np.tile(block_cols, num_blocks)
    data = np.ones(nnz, dtype=np.float64)

    shape = (n * num_blocks, m * num_blocks)
    return csr_array((data, (rows, cols)), shape=shape)

def all_pairings(*lists):
    """
    Convinience function for Fortran ordered product of lists
    """
    return [t[::-1] for t in itertools.product(*reversed(copy(lists)))]

def h_tform(points: np.ndarray, transform:np.ndarray, fill=1) -> np.ndarray:
    """
    Performms a homogenous transformation on data
    :param points: the points to transform
    :param transform: the 4x4 transformation
    :param fill: 1 for points, 0 for vectors.
    :return pts: the transformed points
    """
    if points.ndim == 1:
        points = points[None, ...]

    homogenous_points = np.concatenate(
        [points, np.ones((len(points), 1))*fill], axis=-1
    )[..., None]
    new_points = (transform[None, ...] @ homogenous_points)[..., 0] #always 0 on this axis
    if fill==1:
        new_points = (
                new_points[:, :-1] / new_points[:, -1][..., None]
        )
    else:
        new_points = new_points[:,:-1]
    return new_points.squeeze()

def vol_tet(p0, p1, p2, p3):
    return jnp.abs( 1/6 * jnp.dot( p1 - p0, jnp.cross(p2 - p0, p3 - p0)))


VERTS = [[]]

def vol_hexahedron(pts):
    tetrahedrons = [
        [pts[0], pts[1], pts[3], pts[5]],  # A, B, D, E
        [pts[0], pts[2], pts[3], pts[6]],  # B, D, E, F
        [pts[0], pts[4], pts[5], pts[6]],  # D, F, E, H
        [pts[0], pts[3], pts[5], pts[6]],  # B, C, D, F
        [pts[5], pts[6], pts[7], pts[3]]   # F, H, C, G
    ]
    
    total_volume = 0.0
    for tet in tetrahedrons:

        # s.add_mesh(draw_tet(tet),
        #            # style='wireframe',
        #            )
        v1 = tet[1] - tet[0]  # Vector AB
        v2 = tet[2] - tet[0]  # Vector AD
        v3 = tet[3] - tet[0]  # Vector AE
        
        cross = jnp.cross(v2, v3)
        dot = jnp.dot(v1, cross)
        volume = abs(dot) / 6.0
        total_volume += volume
    return total_volume

def draw_tet(pts):
    tet = pv.PolyData(np.array(pts), faces = [3, 0, 1, 2,   3, 0, 1, 3,   3, 0, 2, 3,   3, 1,2,3 ])
    return tet

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
