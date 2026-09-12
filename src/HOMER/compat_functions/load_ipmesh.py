from os import PathLike
import re
from pathlib import Path
import numpy as np
from functools import reduce

from HOMER.mesh import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3, L2, L1

def extract_numbers(text):
    """Pull every number out of a line of an ipnode/ipelem file.

    :param text:
        The line to scan.

    :returns:
        The numbers as strings, in the order they appear, matching integers,
        decimals and exponent notation.
    """
    pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
    return re.findall(pattern, text)

def process_node(node_str, keys, dim=3):
    """Parse one node block of an ipnode file into MeshNode objects.

    :param node_str:
        The block's lines, the first carrying the node number.
    :param keys:
        Derivative field names to attach, in file order, e.g.
        ``('du', 'dv', 'dudv')``.
    :param dim:
        Spatial dimensions per property.

    :returns:
        A list of nodes: one for a plain node, or one per version for a node
        the file gives multiple versions of, ids suffixed ``_<version>``.

    :raises ValueError:
        If a property does not have three components, which usually means
        *keys* does not match the fields the file declares.
    """
    node_num = re.findall(r"[-+]?(?:\d*\.*\d+)", node_str[0])[-1]
    node_versions = ' The number of ver' == node_str[1][:18]

    if not node_versions: #simple node, just return it 
        num_properties = (len(node_str) - 1)//dim
        node_data = [[] for _ in range(num_properties)]
        for idl, l in enumerate(node_str[1:]):
            prop_num = idl % num_properties
            # datum = re.findall(r"[-+]?(?:\d*\.*\d+)", l)[-1]
            datum = extract_numbers(l)[-1]
            datum = float(datum)
            node_data[prop_num].append(datum)

        node_loc = np.array(node_data[0])
        node_keys = {k:np.array(v) for k, v in zip(keys, node_data[1:])}

        if np.any([len(node_datum) != 3 for node_datum in node_data]):
            raise ValueError(
                f"Node {node_num}: expected 3 components per property, got "
                f"{[len(d) for d in node_data]}. The file may declare a "
                f"different set of derivative fields than the keys given "
                f"({', '.join(keys)})."
            )

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
            datum = extract_numbers(l)[-1]
            datum=float(datum)
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

    return nodes


def load_nodes_ipdata(loc: PathLike, keys):
    """Read nodes from an ipdata file, one node per whitespace-separated row.

    :param loc:
        Path to the ipdata file.  Its first line is skipped as a header.
    :param keys:
        Derivative field names for the columns after the coordinates, each
        consuming three columns.

    :returns:
        The nodes, ids taken from the first column.
    """
    data = np.genfromtxt(loc, skip_header=1)
    nodes = []
    for datum in data:
        node_keys = {key:keylet for key, keylet in zip(keys, datum[4:].reshape(-1,3))}
        new_node = MeshNode(datum[1:4], id=str(int(datum[0])), **node_keys)
        nodes.append(new_node)

    return nodes


def load_node(loc: PathLike, keys):
    """Read nodes from an ipnode file.

    :param loc:
        Path to the ipnode file.
    :param keys:
        Derivative field names to attach, in file order.

    :returns:
        The nodes, in file order, expanded so a versioned node contributes one
        node per version.

    :raises ValueError:
        If *loc* does not exist.
    """
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
    """Parse a single ``The version number...`` line of an ipelem file.

    :param l:
        The line to try.

    :returns:
        ``(matched, [local_node, occurrence, version])`` -- ``(False, [])``
        when the line is not a version line.
    """
    is_vnum = ' The v' == l[:6]
    if not is_vnum:
        return False, []

    ints = re.findall(r"[-+]?(?:\d*\.*\d+)", l)
    local_pts = [int(ints[l]) for l in [0, 1, -1]]
    return True, local_pts


def vnum_checks(ls):
    """Collect an element's version lines into a lookup.

    :param ls:
        The element's lines.

    :returns:
        ``(found, link_dict)`` mapping ``(local_node, occurrence)`` to its
        version; ``(False, None)`` when the element declares no versions.

    :raises NotImplementedError:
        If one occurrence of a node is given two different versions, which
        this reader cannot represent.
    """
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
    """Number each repeat of a value, in order of appearance.

    :param nums:
        The values to number.

    :returns:
        A parallel list giving 0 for each value's first appearance, 1 for the
        second, and so on -- which is what turns a repeated node id in a
        collapsed element into a distinct occurrence.
    """
    count_dict = {}
    result = []
    for num in nums:
        result.append(count_dict.setdefault(num, 0))
        count_dict[num] += 1
    return result


def rename_conditional(a, b, edict):
    """Suffix a node id with its version, when it has one.

    :param a:
        The node id as read from the file.
    :param b:
        Which occurrence of that id this is, zero-based.
    :param edict:
        The lookup from :func:`vnum_checks`.

    :returns:
        ``a`` unchanged, or ``"<a>_<version>"`` when the element declares a
        version for this occurrence.
    """
    if not (b+1, int(a)) in edict:
        return a
    return str(a) + f"_{edict[(b+1, int(a))]}"

def re_ind_elem_nodes(ls, nodes):
    """Rewrite an element's node ids to match the versioned node names.

    :param ls:
        The element's lines, used to find its version declarations.
    :param nodes:
        The element's node ids in file order.

    :returns:
        The ids, each suffixed with its version where one is declared, so they
        match the ids :func:`process_node` gave the versioned nodes.  Returned
        unchanged when the element declares no versions.
    """
    ordering, edict = vnum_checks(ls)
    if not ordering:
        return nodes

    occurs = count_occurrences(nodes)
    new_nodes = [rename_conditional(a,b,edict) for a, b in zip(nodes, occurs)]
    # print(new_nodes)
    return new_nodes

def process_elem_legacy(elem_data, basis_def):
    """Parse an element block in the older ipelem layout.

    Superseded by :func:`process_elem`, which parses the block with explicit
    patterns rather than positional regex.  Kept for files this reader still
    handles better; nothing in HOMER calls it.

    :param elem_data:
        The element block's lines.
    :param basis_def:
        The bases to give the element, one per parametric direction.

    :returns:
        The element, with node ids suffixed by version where the block
        declares them.
    """
    id = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[0])[-1]
    inds = re.findall(r"[-+]?(?:\d*\.*\d+)", elem_data[-1])[2:]
    no_dupe = len(inds) == len(set(inds))

    inds = re_ind_elem_nodes(elem_data, inds)
    # print(inds)
    # print(no_dupe)
    elem = MeshElement(node_ids=inds, basis_functions=basis_def, id=id) #, no_dupe
    return elem

def process_elem(lines: list[str], basis_def) -> MeshElement:
    """
    Parses a list of lines representing a single element block.

    :param lines:
        The element block's lines.
    :param basis_def:
        The bases to give the element, one per parametric direction.

    :returns:
        A :class:`~HOMER.mesh.element.MeshElement` capturing this topology.

    :raises ValueError:
        If njj version values differ for the same node occurrence.
    """
    elem_pattern = re.compile(r"Element number\s*\[.*?\]:\s*(\d+)")
    nodes_pattern = re.compile(r"Enter the \d+.*numbers for basis.*:\s*(.*)")
    version_pattern = re.compile(
        r"The version number for occurrence\s+(\d+)\s+of node\s+(\d+),\s+njj=(\d+)\s+is.*?:\s+(\d+)"
    )

    element_number = None
    nodes = []
    node_versions = {}  # Key: (node, occurrence) -> Value: {njj: version_value}

    for line in lines:
        if element_number is None:
            m_elem = elem_pattern.search(line)
            if m_elem:
                element_number = int(m_elem.group(1))

        m_nodes = nodes_pattern.search(line)
        if m_nodes and not nodes:
            nodes = [int(x) for x in m_nodes.group(1).split()]

        m_ver = version_pattern.search(line)
        if m_ver:
            occ = int(m_ver.group(1))
            node = int(m_ver.group(2))
            njj = int(m_ver.group(3))
            ver = int(m_ver.group(4))

            key = (node, occ)
            if key not in node_versions:
                node_versions[key] = {}
            node_versions[key][njj] = ver

    if element_number is None:
        raise ValueError("No element number found in the provided lines.")

    # Track occurrences for each node and resolve versions
    occ_count = {}
    versions = []

    for node in nodes:
        occ_count[node] = occ_count.get(node, 0) + 1
        occ = occ_count[node]
        key = (node, occ)

        if key in node_versions:
            unique_vers = set(node_versions[key].values())
            if len(unique_vers) > 1:
                raise ValueError(
                    f"Conflicting njj versions found for node {node} (occurrence {occ}): {node_versions[key]}"
                )
            versions.append(next(iter(unique_vers)))
        else:
            versions.append(0)

    node_inds = [str(n) + f"_{v}" if v != 0 else str(n) for n,v in zip(nodes, versions)] 

    elem = MeshElement(node_ids=node_inds, basis_functions=basis_def, id=element_number)
    return elem


def load_elem(loc, basis_def):
    """Read elements from an ipelem file.

    :param loc:
        Path to the ipelem file.
    :param basis_def:
        The bases to give every element, one per parametric direction.

    :returns:
        The elements, referencing nodes by the ids :func:`load_node` assigned.

    :raises ValueError:
        If the file holds element data before any element block begins.
    """
    with open(loc, "r") as f:
        for idl, line in enumerate(f):
            if idl < 3:
                continue

            if idl == 3:
                # breakpoint()
                nums = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                num_elems = max([int(n) for n in nums])
                # elem_data = [[] for _ in range(num_elems)]
                elem_data = []
                # elem_index = -1
                continue

            if line.strip() == '':
                # elem_index += 1
                elem_data.append([])
                # pass

            else:
                if not elem_data:
                    raise ValueError(
                        f"{loc}: element data on line {idl + 1} before any "
                        f"element block began. The file may be malformed, or an "
                        f"ipelem variant this reader does not handle."
                    )
                elem_data[-1].append(line)

    elem = [process_elem(elem_datum, basis_def) for elem_datum in elem_data if not len(elem_datum) == 0]
    return elem
    # return [e for e, t in elem]


def load_mesh(ipnode, ipelem, basis=(H3, H3, L2), keys=('du', 'dv', 'dudv')):
    """Read an OpenCMISS ipnode/ipelem pair into a mesh.

    :param ipnode:
        Path to the ipnode file, holding the node coordinates and derivatives.
    :param ipelem:
        Path to the matching ipelem file, holding the connectivity.
    :param basis:
        The bases to give every element, one per parametric direction.  The
        default suits the bicubic-Hermite-by-quadratic-Lagrange meshes these
        files usually carry.
    :param keys:
        Derivative field names to read off each node, in file order.

    :returns:
        The assembled mesh, with nodes no element references dropped.
    """
    nodes = load_node(ipnode, keys = keys)
    elems = load_elem(ipelem, basis_def=basis)

    meshObj = Mesh(nodes, elems)
    meshObj._clean_pts()
    return meshObj
