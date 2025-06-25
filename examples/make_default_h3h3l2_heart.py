from pathlib import Path
from HOMER import mesh
from HOMER.basis_definitions import H3Basis
from HOMER.optim import jax_comp_kdtree_distance_query
from HOMER.compat_functions.load_exelem import load_mesh
from HOMER.io import save_mesh
import pyvista as pv
import numpy as np

def load_heart_points(floc: Path):
    data = np.genfromtxt(floc, skip_header=1)[:, 1:4]
    return data


if __name__ == "__main__":
    inner_wall = load_heart_points('bin/endo.ipdata')
    outer_wall = load_heart_points('bin/epi.ipdata')
    meshobj = load_mesh(Path('bin/cyl.ipnode'), Path('bin/cyl.ipelem'), 
                     )

    node_dx = []
    for node in meshobj.nodes:
        for k in node.keys():
            node[k] = np.zeros(3)
        node_dx.append(np.linalg.norm(node.loc[1:]))

    for elm in meshobj.elements:
        for idn, node_id in enumerate(elm.nodes):
            if idn < 4:
                s = 15
            elif idn < 8:
                s = 25
            else:
                s = 35
            node = meshobj.get_node(node_ids=node_id)
            uv_loc = node.loc[1:]/np.linalg.norm(node.loc[1:])
            new_loc = s * uv_loc
            node.loc[1:] = new_loc
    
    meshobj.generate_mesh()
    meshobj.plot(labels=True, node_size=0)
    save_mesh(meshobj, 'bin/heart_default.json')

