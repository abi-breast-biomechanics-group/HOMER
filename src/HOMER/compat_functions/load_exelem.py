from os import PathLike
import re
from pathlib import Path
import numpy as np

from HOMER.mesher import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L2Basis

def process_node(node_str, keys, dim=3):
    node_num = re.findall(r"[-+]?(?:\d*\.*\d+)", node_str[0])[-1]
    num_properties = (len(node_str) - 1)//dim
    
    node_data = [[] for _ in range(num_properties)]
    for idl, l in enumerate(node_str[1:]):
        prop_num = idl % num_properties
        datum = float(re.findall(r"[-+]?(?:\d*\.*\d+)", l)[-1])
        node_data[prop_num].append(datum)

    node_loc = np.array(node_data[0])
    node_keys = {k:np.array(v) for k, v in zip(keys, node_data[1:])}

    if np.any([len(node_datum) != 3 for node_datum in node_data]):
        breakpoint()

    
    node = MeshNode(node_loc, id=node_num, **node_keys)

    return node

def load_node(loc: PathLike, keys):
    if not isinstance(loc, Path):
        loc = Path(loc)
    if not loc.exists():
        raise ValueError(f"file {loc} doesn't exist")

    with open(loc, "r") as f:

        for idl, line in enumerate(f):
            if idl < 3:
                continue

            if idl == 3:
                nums = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                num_nodes = int(nums[-1])
                node_data = [[] for _ in range(num_nodes)]
                node_index = -1
                continue

            if idl < 12:
                continue
            
            if line == ' \n':
                node_index += 1


            else:
                node_data[node_index].append(line)

    return [process_node(node_datum, keys) for node_datum in node_data]
 
def process_elem(elem_data, basis_def):
    id = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[0])[-1]
    inds = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[-1])
    no_dupe = len(inds) == len(set(inds))
    # print(no_dupe)
    elem = MeshElement(node_ids=inds[2:], basis_functions=basis_def, id=id), no_dupe
    return elem

def load_elem(loc, basis_def):

    with open(loc, "r") as f:

        for idl, line in enumerate(f):
            if idl < 3:
                continue

            if idl == 3:
                nums = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                num_elems = int(nums[-1])
                elem_data = [[] for _ in range(num_elems)]
                elem_index = -1
                continue
            
            if line == ' \n':
                elem_index += 1

            else:
                elem_data[elem_index].append(line)

    elem = [process_elem(elem_datum, basis_def) for elem_datum in elem_data]
    return [e for e, t in elem]

def load_mesh(ipnode, ipelem, basis=(H3Basis, H3Basis, L2Basis), keys=('du', 'dv', 'dudv')):
    nodes = load_node(ipnode, keys = keys)
    elems = load_elem(ipelem, basis_def=basis)

    meshObj = Mesh(nodes, elems)
    meshObj._clean_pts()
    return meshObj

if __name__ == "__main__":
    ipnode = Path("bin/cyl.ipnode")
    ipelem = Path("bin/cyl.ipelem")
    nodes = load_node(ipnode, keys = ['du', 'dv', 'dudv'])
    elems = load_elem(ipelem, basis_def=(H3Basis, H3Basis, L2Basis))

    meshObj = mesh(nodes, elems)
    meshObj._clean_pts()
    meshObj.plot()

