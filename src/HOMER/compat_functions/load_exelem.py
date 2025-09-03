from os import PathLike
import re
from pathlib import Path
import numpy as np
from functools import reduce

from HOMER.mesher import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L2Basis

def extract_numbers(text):
    pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
    return re.findall(pattern, text)

def process_node(node_str, keys, dim=3):

    node_num = re.findall(r"[-+]?(?:\d*\.*\d+)", node_str[0])[-1]
    node_versions = ' The number of ver' == node_str[1][:18]

    if not node_versions: #simple node, just return it 
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

        node = MeshNode(node_loc, id=str(node_num), **node_keys)
        return [node]
    #else we have multiple versions, return multiple nodes.
    num_versions = int(re.findall(r"[-+]?(?:\d*\.*\d+)", node_str[1])[-1])
    node_data = [[[] for _ in range(3)] for _ in range(num_versions)]
    node_version = 1
    l_since_reset = 0
    dim = -1
    for idl, l in enumerate(node_str[1:]):
        # print(l)
        if l[:18] == ' The number of ver':
            temp_v = int(re.findall(r"[-+]?(?:\d*\.*\d+)", l)[-1]) #get the current node version
            if not temp_v == num_versions:
                raise NotImplementedError("HOMER doesn't support heterogenous node version numbers per properties")
            dim += 1 #increment the dim here


        elif l[:4] == ' For': #this highlights the node number.
            node_version = int(re.findall(r"[-+]?(?:\d*\.*\d+)", l)[-1]) #get the current node version
            # print(node_version)
            l_since_reset = 0
        else:
            datum = float(extract_numbers(l)[-1])
            node_data[node_version - 1][dim].append(datum)
            l_since_reset += 1
        # if num_versions > 1:

    nodes = []
    for idv, vnode_data in enumerate(node_data):
        vnode_data = np.array(vnode_data).T
        node_loc = np.array(vnode_data[0])
        node_keys = {k:np.array(v) for k, v in zip(keys, vnode_data[1:])}
        node_id = str(node_num) + f"_{idv+1}" if num_versions != 1 else str(node_num)
        node = MeshNode(node_loc, id=node_id, **node_keys)
        nodes.append(node)

    # if len(nodes) > 1:
    #     breakpoint()
    return nodes


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

    nodes = []
    for node_datum in node_data:
        nodes.extend(process_node(node_datum, keys))
    return nodes
  
def vnum_parse(l):
    is_vnum = ' The v' == l[:6]
    if not is_vnum:
        return False, []
    ints = re.findall(r"[-+]?(?:\d*\.*\d+)", l)
    local_pts = [int(ints[l]) for l in [0, 1, -1]]
    return True, local_pts

def vnum_checks(ls):
    parsed = [vnum_parse(l)[1] for l in ls if vnum_parse(l)[0]]
    if len(parsed) == 0:
        return False, None
    #assertion check here is that every node occurrence has the same final value 
    link_dict = {}
    for p in parsed:
        v = link_dict.get((p[0], p[1]), None)
        if v is None:
            link_dict[(p[0], p[1])] = p[2]
        elif v is not None:
            if not v == p[2]:
                raise NotImplementedError("Element had non heterogenous versions for occurences")
    return True, link_dict

def count_occurrences(nums):
    count_dict = {}
    result = []
    for num in nums:
        result.append(count_dict.setdefault(num, 0))
        count_dict[num] += 1
    return result

def rename_conditional(a, b, edict):
    if not (b+1, int(a)) in edict:
        return a
    return str(a) + f"_{edict[(b+1, int(a))]}"

def re_ind_elem_nodes(ls, nodes):
    ordering, edict = vnum_checks(ls)
    if not ordering:
        return nodes
    occurs = count_occurrences(nodes)
    new_nodes = [rename_conditional(a,b,edict) for a, b in zip(nodes, occurs)]
    print(new_nodes)
    return new_nodes
 
def process_elem(elem_data, basis_def):
    id = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[0])[-1]
    inds = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[-1])[2:]
    no_dupe = len(inds) == len(set(inds))
    
    inds = re_ind_elem_nodes(elem_data, inds)

    # print(no_dupe)
    elem = MeshElement(node_ids=inds, basis_functions=basis_def, id=id), no_dupe
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
    ipnode = Path("bin/heart/BB001_RC_Cubic_59.ipnode")
    ipelem = Path("bin/heart/BB001_RC_Cubic_59.ipelem")
    # ipelem = Path("bin/cyl.ipelem")
    nodes = load_node(ipnode, 
                      # keys = ['du', 'dv', 'dudv'],
                      keys = ['du', 'dv', 'dudv', 'dw', 'dudw', 'dvdw', 'dudvdw'],
                      )
    elems = load_elem(ipelem, basis_def=(H3Basis, H3Basis, H3Basis))

    meshObj = Mesh(nodes, elems)
    meshObj._clean_pts()
    meshObj.plot(node_size=0, labels=True)

    node = meshObj.get_node('3')
    print(node)

