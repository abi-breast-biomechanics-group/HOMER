import jax.numpy as jnp
import numpy as np
import pyvista as pv

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
