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
