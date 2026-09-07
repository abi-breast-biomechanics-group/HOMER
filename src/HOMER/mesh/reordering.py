"""
reordering.py - giving a mesh a predictable node numbering.

Refining, rebasing and most other manipulations rebuild the node list from
scratch, and the order they happen to build it in is an artefact of the
algorithm - for a refined mesh it is "first time each node was touched while
sweeping the sub-elements of each parent in turn", which bears no relation to
where the node sits.  :func:`reorder_nodes` replaces that with an ordering
derived from the mesh itself, so the same mesh built two different ways
numbers its nodes the same way.

An operation that does *not* change the node set is a special case, and is
handled first: when every new node sits on a distinct old node and none is
left over - an H3 to L1 rebase, a refinement by a factor of one - the old
numbering is reproduced exactly rather than replaced, so node indices a caller
is holding stay valid.  That is what :func:`preserving_permutation` decides,
from the same parent-node map that carries fixed parameters across.  It only
recognises a one-to-one correspondence: after a real refinement the old nodes
are a strict *subset* of the new ones, and putting them back at their old
indices would leave the new ones in the arbitrary order this module exists to
get rid of, so that case is ordered by strategy like any other.

Three orderings are available, named by :data:`STRATEGIES`:

``'lattice'`` (the default)
    Sort by the node's position in the mesh's *parametric* lattice: walk the
    element connectivity to give every element an integer grid coordinate,
    add the basis node locations to place each node within its element, and
    sort that coordinate with the last parametric direction slowest.  For a
    subdivided cube this is exactly the lexicographic ``(z, y, x)`` ordering
    :func:`~HOMER.geometry.cubeMNO` builds by hand, but it is computed from
    the topology rather than the coordinates, so it is equally meaningful for
    a secondary field (whose ``loc`` holds field values, not positions) and is
    unaffected by how the mesh is posed or deformed.

``'spatial'``
    Lexicographic sort of the nodal coordinates themselves, last coordinate
    slowest - i.e. ``np.lexsort((x, y, z))``.  Only meaningful for a geometric
    field, and sensitive to the pose of the mesh.

``'bandwidth'``
    Reverse Cuthill-McKee over the node adjacency graph.  Makes no assumption
    that the mesh is a lattice, and minimises the bandwidth of the parameter
    couplings, which is what an unstructured mesh wants.

Reordering is a pure renumbering: node objects, elements, bases and every
value stored on a node (including :attr:`~HOMER.mesh.node.MeshNode.fixed_params`)
are carried across untouched, so the field evaluates identically before and
after.
"""

from collections import deque
from typing import Optional, TYPE_CHECKING

import numpy as np

from HOMER.utils import all_pairings

if TYPE_CHECKING:
    from HOMER.mesh.field import MeshField


#: the ordering used when a caller passes ``reorder_nodes=True``.  Set this to
#: ``False`` to turn the automatic reordering off across the whole session.
DEFAULT_NODE_ORDERING = 'lattice'

STRATEGIES = ('lattice', 'spatial', 'bandwidth')

#: parametric coordinates are snapped to this many decimals before sorting.
#: They are integers plus the basis node locations, so this only has to absorb
#: the representation error of a node location like 1/3.
SORT_ROUNDING = 9

#: the ``'spatial'`` ordering quantises coordinates to this fraction of the
#: mesh's largest extent before sorting.  Nodal coordinates come out of a
#: least-squares fit and carry its noise, so an unquantised sort would let a
#: part in a million decide which of two nodes in the same plane comes first.
SPATIAL_TOLERANCE = 1e-4


def resolve_strategy(strategy) -> Optional[str]:
    """Map a ``reorder_nodes=`` argument onto a strategy name, or ``None``.

    ``None`` means "leave the numbering alone", which is what a caller checks
    when it wants to know whether to do the work of building a parent-node map
    at all.
    """
    if strategy is None or strategy is False:
        return None
    if strategy is True:
        strategy = DEFAULT_NODE_ORDERING
        if strategy is None or strategy is False:
            return None
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown node ordering {strategy!r}; expected one of {STRATEGIES}, "
                         "or True/False")
    return strategy


def _element_node_lists(field: 'MeshField') -> np.ndarray:
    """``(n_elements, n_nodes_per_element)`` array of *index* references."""
    field._update_id_mappings()
    rows = []
    for element in field.elements:
        if element.used_index:
            rows.append([int(n) for n in element.nodes])
        else:
            rows.append([field.node_id_to_ind[node_id] for node_id in element.nodes])
    return np.array(rows, dtype=int)


def element_lattice_coords(topo_lookup) -> tuple[np.ndarray, np.ndarray]:
    """Give every element an integer coordinate in the mesh's element lattice.

    Breadth-first from element 0, stepping ``+1`` in direction *d* for the
    neighbour across the ``xi_d = 1`` face and ``-1`` for the one across
    ``xi_d = 0``.  Elements that cannot be reached start a new component at the
    origin, so a mesh in several pieces is ordered a piece at a time.

    This assumes neighbouring elements agree on which parametric direction is
    which - true of anything HOMER builds by refining or rebasing.  Where it is
    not true the coordinates are still deterministic, just no longer a lattice.

    Parameters
    ----------
    topo_lookup:
        ``(n_elements, ndim, 2)`` neighbour array, as built by
        :meth:`~HOMER.mesh.topology._explore_topology`; ``-1`` means no
        neighbour on that face.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(component, coords)`` - the connected component index of each
        element, and its ``(n_elements, ndim)`` integer coordinate.
    """
    lookup = np.asarray(topo_lookup)
    n_ele, ndim = lookup.shape[0], lookup.shape[1]

    coords = np.zeros((n_ele, ndim), dtype=int)
    component = np.zeros(n_ele, dtype=int)
    seen = np.zeros(n_ele, dtype=bool)

    next_component = 0
    for start in range(n_ele):
        if seen[start]:
            continue
        seen[start] = True
        component[start] = next_component
        queue = deque([start])
        while queue:
            ele = queue.popleft()
            for d in range(ndim):
                for side, step in ((0, -1), (1, 1)):
                    neighbour = int(lookup[ele, d, side])
                    if neighbour < 0 or seen[neighbour]:
                        continue
                    coords[neighbour] = coords[ele]
                    coords[neighbour, d] += step
                    component[neighbour] = next_component
                    seen[neighbour] = True
                    queue.append(neighbour)
        next_component += 1

    return component, coords


def _lattice_key(field: 'MeshField', ele_nodes: np.ndarray, topo_lookup) -> np.ndarray:
    """Parametric lattice coordinate of every node, ``(n_nodes, ndim + 1)``.

    The first ``ndim`` columns are the node's position in the lattice and the
    last is its connected component, which is the column order
    :func:`numpy.lexsort` wants: it takes its last key as the primary one.  A
    node shared by several elements gets the smallest coordinate any of them
    gives it, which is *the* coordinate whenever the elements agree.
    Unreferenced nodes get ``inf`` and so sort last.
    """
    component, coords = element_lattice_coords(topo_lookup)

    basis = field.elements[0].basis_functions
    #direction 0 fastest, matching the local node ordering of an element
    offsets = np.array(all_pairings(*[np.asarray(b.node_locs, dtype=float) for b in basis]))

    n_ele, n_local = ele_nodes.shape
    if offsets.shape[0] != n_local:
        raise ValueError(f"element node lists hold {n_local} nodes, but the basis has "
                         f"{offsets.shape[0]} local nodes")

    #(n_ele, n_local, ndim) parametric position, plus the component alongside
    positions = coords[:, None, :] + offsets[None, :, :]
    positions = np.round(positions, SORT_ROUNDING)
    stacked = np.concatenate(
        [positions, np.broadcast_to(component[:, None, None], (n_ele, n_local, 1)).astype(float)],
        axis=-1,
    )

    key = np.full((len(field.nodes), stacked.shape[-1]), np.inf)
    flat_nodes = ele_nodes.ravel()
    flat_keys = stacked.reshape(-1, stacked.shape[-1])
    #np.minimum.at gives the per-element minimum, which is the shared value
    #wherever the elements agree on where the node is.
    np.minimum.at(key, flat_nodes, flat_keys)
    return key


def _bandwidth_permutation(field: 'MeshField', ele_nodes: np.ndarray) -> np.ndarray:
    """Reverse Cuthill-McKee ordering of the node adjacency graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    n_nodes = len(field.nodes)
    rows = np.repeat(ele_nodes, ele_nodes.shape[1], axis=1).ravel()
    cols = np.tile(ele_nodes, (1, ele_nodes.shape[1])).ravel()
    graph = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return np.asarray(reverse_cuthill_mckee(graph, symmetric_mode=True), dtype=int)


def preserving_permutation(parent_of_new, n_old: int) -> Optional[np.ndarray]:
    """The permutation that puts every new node back where its old one was.

    Only defined when the operation did not change the node set: every new node
    has an old counterpart, no two share one, and no old node is left without a
    successor.  Anything else returns ``None``, because there is then no old
    ordering to reproduce.

    Parameters
    ----------
    parent_of_new:
        For each new node, the index of the old node it coincides with, or
        ``-1`` for none - as built by
        :func:`~HOMER.mesh.refinement._parent_node_map`.
    n_old:
        How many nodes the mesh had before the operation.

    Returns
    -------
    numpy.ndarray or None
        A permutation of the new nodes, or ``None`` if the correspondence is
        not one-to-one.
    """
    parent = np.asarray(parent_of_new, dtype=int)
    if parent.shape[0] != n_old or (parent < 0).any():
        return None
    if np.unique(parent).shape[0] != n_old:
        return None
    #parent is a permutation, so its inverse is the ordering we want
    return np.argsort(parent)


def node_permutation(field: 'MeshField', strategy=True, topo_lookup=None,
                     parent_of_new=None, n_old: Optional[int] = None) -> Optional[np.ndarray]:
    """Work out the new node order without applying it.

    Returns ``None`` when *strategy* disables reordering, otherwise an index
    array *perm* such that ``field.nodes[perm[i]]`` is the node that should end
    up at position *i*.  Ties keep their current relative order.

    Parameters
    ----------
    field:
        The field to order.  Only its nodes and elements are read.
    strategy:
        One of :data:`STRATEGIES`, ``True`` for :data:`DEFAULT_NODE_ORDERING`,
        or ``False``/``None`` to do nothing.
    topo_lookup:
        Neighbour array to use for the ``'lattice'`` ordering.  Defaults to the
        field's own ``_topo_lookup``; pass it explicitly when the field's copy
        is stale, as it is midway through a refinement.
    parent_of_new, n_old:
        The map from each new node to the old node it coincides with, and the
        old node count.  When the two describe a one-to-one correspondence the
        old numbering is reproduced and *strategy* is not consulted - see
        :func:`preserving_permutation`.
    """
    strategy = resolve_strategy(strategy)
    if strategy is None or not field.nodes or not field.elements:
        return None

    if parent_of_new is not None and n_old is not None:
        preserved = preserving_permutation(parent_of_new, n_old)
        if preserved is not None:
            return preserved

    ele_nodes = _element_node_lists(field)

    if strategy == 'spatial':
        locs = np.array([np.asarray(node.loc, dtype=float) for node in field.nodes])
        extent = float(np.max(np.ptp(locs, axis=0))) if locs.size else 0.0
        tolerance = extent * SPATIAL_TOLERANCE if extent > 0 else 1.0
        return np.lexsort(np.round(locs / tolerance).T)  #last coordinate slowest

    if strategy == 'bandwidth':
        return _bandwidth_permutation(field, ele_nodes)

    if topo_lookup is None:
        topo_lookup = getattr(field, '_topo_lookup', None)
    if topo_lookup is None:
        raise ValueError("the 'lattice' ordering needs the element connectivity: generate the "
                         "mesh first, or pass topo_lookup explicitly")
    key = _lattice_key(field, ele_nodes, topo_lookup)
    return np.lexsort(key.T)  #component slowest, then the last direction, direction 0 fastest


def apply_node_permutation(field: 'MeshField', perm: np.ndarray, generate=True) -> None:
    """Renumber *field*'s nodes into the order given by *perm*, in place.

    Elements that reference their nodes by index are rewritten; ones that
    reference them by id need no rewriting, since the id travels with the node.
    """
    perm = np.asarray(perm, dtype=int)
    if perm.shape != (len(field.nodes),):
        raise ValueError(f"permutation has {perm.shape[0]} entries for {len(field.nodes)} nodes")

    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.shape[0])

    field.nodes = [field.nodes[i] for i in perm]
    for element in field.elements:
        if element.used_index:
            element.nodes = [int(inverse[n]) for n in element.nodes]

    if generate:
        field.generate_mesh()
    else:
        field._update_id_mappings()


def reorder_nodes(field: 'MeshField', strategy=True, topo_lookup=None,
                  generate=True, parent_of_new=None,
                  n_old: Optional[int] = None) -> Optional[np.ndarray]:
    """Renumber a field's nodes into a predictable order, in place.

    A convenience wrapper over :func:`node_permutation` and
    :func:`apply_node_permutation`.  Call it after any manipulation that
    rebuilds the node list; :meth:`~HOMER.mesh.field.MeshField.refine` and
    :meth:`~HOMER.mesh.field.MeshField.rebase` already do.

    Parameters
    ----------
    field:
        The :class:`~HOMER.mesh.field.MeshField` (or :class:`~HOMER.mesh.mesh.Mesh`)
        to renumber.  A ``Mesh``'s secondary fields are *not* touched - each
        field owns its own node list, and reordering one has no bearing on the
        others.
    strategy:
        One of :data:`STRATEGIES`, ``True`` for :data:`DEFAULT_NODE_ORDERING`,
        or ``False``/``None`` to do nothing.
    topo_lookup:
        Neighbour array for the ``'lattice'`` ordering; defaults to the field's
        own.
    generate:
        Rebuild the mesh afterwards.  Pass ``False`` when the caller is about
        to call :meth:`~HOMER.mesh.field.MeshField.generate_mesh` anyway.
    parent_of_new, n_old:
        Where the new nodes came from, if the caller knows - see
        :func:`node_permutation`.  Passed by
        :meth:`~HOMER.mesh.field.MeshField.refine` and
        :meth:`~HOMER.mesh.field.MeshField.rebase` so that an operation which
        leaves the node set alone leaves the numbering alone too.

    Returns
    -------
    numpy.ndarray or None
        The permutation applied, so a caller can carry an index list of its own
        across the renumbering (``inverse[old_index]`` gives the new index,
        where ``inverse[perm] = arange(n)``); ``None`` if nothing was done.

    Examples
    --------
    Put an arbitrarily numbered mesh back into lattice order::

        from HOMER.mesh.reordering import reorder_nodes

        reorder_nodes(mesh)                      # the default 'lattice' order
        reorder_nodes(mesh, 'spatial')           # lexicographic in x, y, z
        reorder_nodes(mesh, 'bandwidth')         # reverse Cuthill-McKee
    """
    perm = node_permutation(field, strategy=strategy, topo_lookup=topo_lookup,
                            parent_of_new=parent_of_new, n_old=n_old)
    if perm is None:
        return None
    apply_node_permutation(field, perm, generate=generate)
    return perm
