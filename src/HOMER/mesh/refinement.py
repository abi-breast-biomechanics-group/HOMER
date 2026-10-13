"""
refinement.py - subdividing a mesh, and moving it to a different basis.

:func:`refine`, :func:`refine_elements` and :func:`rebase` are methods of
:class:`~HOMER.mesh.field.MeshField` that live in this module; the first
argument is the field itself, and :mod:`HOMER.mesh.field` binds them into the
class.  The private helpers around them work out which node of the refined
mesh came from which node of the original, which is what lets a pinned
parameter survive the operation.
"""

import itertools
import logging
from copy import deepcopy
from fractions import Fraction
from typing import Optional, TYPE_CHECKING

import numpy as np

from HOMER.basis_definitions import Basis, BasisGroup
from HOMER.mesh.node import MeshNode
from HOMER.mesh.element import MeshElement
from HOMER.mesh.hierarchy import Hierarchy, prolongation
from HOMER.mesh.reordering import (reorder_nodes as _reorder_nodes, resolve_strategy,
                                   _element_node_lists, element_lattice_coords)
from HOMER.topomap_operations import global_nodes_from_ele_localnodes, refine_connectivity
from HOMER.utils import all_pairings

if TYPE_CHECKING:
    from HOMER.mesh.field import MeshField


MAX_XI_DENOMINATOR = 10**6


def _basis_node_fractions(basis: Basis, max_denominator: int = MAX_XI_DENOMINATOR) -> list[Fraction]:
    """Exact rational node locations of a 1-D basis.

    ``node_locs`` are fixed constants of the basis, so the snap to a rational is exact
    in intent.  Note that they are not required to lie inside ``[0, 1]``:
    :class:`~HOMER.basis_definitions.B3` places its shared control points
    at ``[-1, 0, 1, 2]``.
    """
    return [Fraction(float(loc)).limit_denominator(max_denominator) for loc in basis.node_locs]


def _as_fractions(values, max_denominator: int = MAX_XI_DENOMINATOR) -> list[Fraction]:
    """Snap a sequence of parametric breakpoints to exact rationals."""
    return [Fraction(float(v)).limit_denominator(max_denominator) for v in values]


def _basis_slot_correspondence(old_bases, new_bases, xi_breaks, refinement) -> list[np.ndarray]:
    """Per-direction map from ``(sub-element index, new basis node)`` to old basis node.

    Entry ``[c, j]`` is the index of the old basis node that the new basis node
    *j* of sub-element *c* sits exactly on top of, or ``-1`` when the new node
    has no counterpart.  The comparison is exact rational arithmetic on the
    basis node locations - never on nodal coordinates.

    old_bases, new_bases:
        1-D bases before and after the operation, one per direction.
    :param xi_breaks:
        Per-direction list of :class:`~fractions.Fraction` sub-element
        boundaries within the parent element, length ``refinement[d] + 1``.
    :param refinement:
        Number of sub-elements per direction (all ones for a rebase).
    """
    maps = []
    for d, (old_basis, new_basis) in enumerate(zip(old_bases, new_bases)):
        f_old = _basis_node_fractions(old_basis)
        f_new = _basis_node_fractions(new_basis)

        lookup = {}
        for i, loc in enumerate(f_old):
            lookup.setdefault(loc, i)

        n_sub = int(refinement[d])
        breaks = xi_breaks[d]
        slot_map = np.full((n_sub, len(f_new)), -1, dtype=int)
        for c in range(n_sub):
            lo, span = breaks[c], breaks[c + 1] - breaks[c]
            for j, loc in enumerate(f_new):
                slot_map[c, j] = lookup.get(lo + loc * span, -1)
        maps.append(slot_map)
    return maps


def _require_conforming(field: 'MeshField', operation: str) -> None:
    """Refuse *operation* on a mesh with hanging nodes, which it reads as a lattice."""
    if getattr(field, 'hanging', None):
        raise NotImplementedError(f"{operation} reads the mesh as a lattice of elements, which a "
                                  f"mesh with {len(field.hanging)} hanging faces is not")


def _element_widths(element: MeshElement) -> np.ndarray:
    """The width each direction of *element* scales its derivatives by.

    A node derivative is taken along a parametrisation shared by every element
    it serves; an element of width *w* in that parametrisation sees it scaled
    by *w*, which is what its map records on any slot that is a plain scaling of
    its node's field.  ``1`` along a direction with no derivatives.

    :raises NotImplementedError:
        If a direction has derivatives but no slot that plainly scales one.
    """
    widths = np.ones(element.ndim)
    if element.param_map is None:
        return widths
    orders = element.derivative_orders
    for d, first in enumerate(np.eye(element.ndim, dtype=int)):
        along = [element.param_map[element.slots[s]] for s in np.flatnonzero((orders == first).all(1))]
        plain = [terms[0][2] for terms in along if len(terms) == 1]
        if along and not plain:
            raise NotImplementedError(f"no slot of this element plainly scales a derivative along "
                                      f"direction {d}, so its width cannot be read off its map")
        if plain:
            widths[d] = plain[0]
    return widths


def _width_map(element: MeshElement, widths: np.ndarray) -> Optional[dict]:
    """The map scaling each derivative slot by the widths of its directions.

    ``None`` -- the identity -- when every scale is 1.
    """
    scale = np.prod(np.where(element.derivative_orders, widths, 1.0), axis=1)
    if np.all(scale == 1):
        return None
    return {slot: [(slot[0], slot[1], float(w))] for slot, w in zip(element.slots, scale)}


def _parent_node_map(old_field: 'MeshField', old_bases, new_bases, new_ele_nodes: np.ndarray,
                     parent_coords: np.ndarray, refinement, xi_breaks, n_new_nodes: int) -> np.ndarray:
    """Map every new global node to the old global node it coincides with.

    Element identity is preserved by both refinement and rebasing, so a new
    node is identified by *which basis slot of which parent element* it
    occupies.  That is resolved purely with integer indices and exact rational
    node locations, so nothing here depends on the mesh coordinates or their
    dtype.

    :returns:
        numpy.ndarray
            Length ``n_new_nodes``, holding the old node index for each new node,
            or ``-1`` where the new node has no counterpart.
    """
    old_shape = [len(b.node_locs) for b in old_bases]
    new_shape = [len(b.node_locs) for b in new_bases]
    ndim = len(new_shape)
    n_new_local = int(np.prod(new_shape))

    maps = _basis_slot_correspondence(old_bases, new_bases, xi_breaks, refinement)

    # all_pairings orders local points with direction 0 fastest (Fortran order),
    # matching the sub-element ordering used by refine_connectivity.
    new_local_j = np.unravel_index(np.arange(n_new_local), new_shape, order='F')
    old_strides = np.concatenate([[1], np.cumprod(old_shape[:-1])]).astype(int)

    n_elements = new_ele_nodes.shape[0]
    parent_ele = np.arange(n_elements) // int(np.prod(refinement))

    old_local = np.zeros((n_elements, n_new_local), dtype=int)
    valid = np.ones((n_elements, n_new_local), dtype=bool)
    for d in range(ndim):
        matched = maps[d][parent_coords[:, d], :][:, new_local_j[d]]
        valid &= matched >= 0
        old_local += np.where(matched >= 0, matched, 0) * old_strides[d]

    parent_of_new = np.full(n_new_nodes, -1, dtype=int)
    if valid.any():
        old_node_of = _element_node_lists(old_field)
        rows = np.repeat(parent_ele[:, None], n_new_local, axis=1)[valid]
        parent_of_new[new_ele_nodes[valid]] = old_node_of[rows, old_local[valid]]
    return parent_of_new


def _hold_pinned_locations(old_nodes, new_nodes, parent_of_new: np.ndarray) -> None:
    """Pin the coincident new nodes to their old locations, before the fit runs.

    A ``loc`` constraint on an interpolatory basis is the one kind whose value
    is known ahead of the least-squares solve, so it is the one kind the solve
    can honour: the landmark is held and the parameters around it are fitted
    to the best they can be given it.  Everything else -- derivative
    constraints, whose magnitude is element-scale dependent (a refined
    element's ``du`` is legitimately the parent's divided by the refinement
    factor), and ``loc`` on a control net, where the node is not on the
    surface -- takes the value the fit produces, so :func:`_transfer_fixed_params`
    flags it afterwards.
    """
    for new_index, old_index in enumerate(parent_of_new):
        if old_index < 0:
            continue
        inds = np.asarray(old_nodes[old_index].fixed_params.get('loc', []), dtype=int)
        if inds.size == 0:
            continue
        new_nodes[new_index].fix_parameter('loc', values=np.asarray(old_nodes[old_index].loc)[inds],
                                           inds=inds)


def _transfer_fixed_params(old_nodes, new_nodes, parent_of_new: np.ndarray) -> tuple[int, int, int]:
    """Carry :attr:`MeshNode.fixed_params` across to the coincident new nodes.

    The flags only: a pinned location was already held through the fit by
    :func:`_hold_pinned_locations`, and every other constraint is asserted at
    whatever value the fit gave it.

    :returns:
        tuple
            ``(constraints transferred, constraints dropped, fixed old nodes with
            no counterpart)``.
    """
    transferred = 0
    dropped = 0
    matched_old = set()

    for new_index, old_index in enumerate(parent_of_new):
        if old_index < 0:
            continue
        old_node = old_nodes[old_index]
        if not old_node.fixed_params:
            continue
        matched_old.add(int(old_index))

        new_node = new_nodes[new_index]
        for param, inds in old_node.fixed_params.items():
            inds = np.asarray(inds, dtype=int)
            if inds.size == 0:
                continue
            if param != 'loc' and param not in new_node:
                dropped += 1 #the new basis carries no such derivative
                continue
            new_node.fix_parameter(param, inds=inds)
            transferred += 1

    unmatched = sum(1 for i, node in enumerate(old_nodes) if node.fixed_params and i not in matched_old)
    return transferred, dropped, unmatched


def _report_fixed_param_transfer(operation: str, stats: tuple[int, int, int], interpolatory: bool) -> None:
    """Warn about constraints that could not be carried through *operation*."""
    transferred, dropped, unmatched = stats
    if dropped or unmatched:
        logging.warning(
            f"{operation}: {transferred} fixed nodal parameter(s) preserved, but {dropped} were dropped "
            f"(no equivalent parameter in the new basis) and {unmatched} fixed node(s) had no counterpart "
            f"in the new mesh. Those degrees of freedom are now optimisable."
        )
    if transferred and not interpolatory:
        logging.warning(
            f"{operation}: the basis is not interpolatory, so fixed parameters were re-applied to the "
            f"lattice-coincident control points at their fitted values. This approximates the original "
            f"constraint, which is spread over several control points after {operation}."
        )


def refine(self, refinement_factor: Optional[int|list[int]]=None, by_xi_refinement: Optional[tuple[np.ndarray]] =  None,
           clean_nodes = True, preserve_fixed_params = True, reorder_nodes = True):
    """Subdivide every element, increasing the mesh resolution.

    Each existing element is replaced by ``refinement_factor ** ndim``
    (or the equivalent for *by_xi_refinement*) smaller elements sharing
    intermediate nodes.  Derivative values at the new nodes are obtained
    by evaluating the current basis functions.

    Exactly one of *refinement_factor* or *by_xi_refinement* must be
    provided.

    :param refinement_factor:
        Integer ≥ 2 that subdivides each parametric direction uniformly.
        For example, ``refinement_factor=2`` splits a single element into
        8 sub-elements in 3-D (2 × 2 × 2).
    :param by_xi_refinement:
        Tuple of 1-D arrays, one per parametric direction, specifying the
        xi values at which to place the new element boundaries.  Each
        array must start with 0 and end with 1.
    :param clean_nodes:
        When ``True`` (default), remove unreferenced nodes after
        refinement.
    :param preserve_fixed_params:
        When ``True`` (default), any node of the refined mesh that sits
        exactly on an existing node inherits that node's
        :attr:`MeshNode.fixed_params`.  For an interpolatory basis a pinned
        ``loc`` is held *through* the fit, so the landmark keeps its value
        exactly and the rest of the mesh is fitted around it; constraints
        with no counterpart in the refined mesh are dropped and reported.
    :param reorder_nodes:
        Refinement rebuilds the node list, and the order it falls out in is
        an artefact of the sub-element sweep.  When ``True`` (default) the
        refined nodes are renumbered by
        :func:`~HOMER.mesh.reordering.reorder_nodes` using
        :data:`~HOMER.mesh.reordering.DEFAULT_NODE_ORDERING`; pass ``False``
        to keep the raw ordering, or a strategy name to choose another.
        A refinement that does not actually add nodes - a factor of one in
        every direction - keeps the numbering it had, so node indices survive
        it; a refinement that does add nodes renumbers, and indices into the
        coarse mesh do not carry over.

    :raises AssertionError: If both *refinement_factor* and *by_xi_refinement* are given, or
        if *refinement_factor* < 2.
    """
    assert not(refinement_factor is not None and by_xi_refinement is not None), "Refinement factor and refining by defined xi are mutually exclusive."
    _require_conforming(self, 'refine')

    #input handling, map both refinement factors and values to the same array. 
    if refinement_factor is not None:
        if isinstance(refinement_factor, int):
            refinement_factor = [refinement_factor] * self.ndim
        xi_locs = [np.linspace(0,1,rf+1) for rf in refinement_factor]
        f_xi_locs = [[Fraction(k, rf) for k in range(rf+1)] for rf in refinement_factor]
    elif by_xi_refinement is not None:
        refinement_factor = [len(xival) - 1 for xival in by_xi_refinement]
        xi_locs = by_xi_refinement
        f_xi_locs = [_as_fractions(xival) for xival in by_xi_refinement]
    else:
        raise ValueError("one of refinement factor and by_xi_refinement must be defined")
    xi_locs = [np.array(x) for x in xi_locs]
    scales = [np.diff(x) for x in xi_locs]

    basis = self.elements[0].basis_functions
    used_fields = self.elements[0].used_node_fields

    new_topo_lookup, parent_connectivity = refine_connectivity(self._topo_lookup, refinement_factor)
    eval_pts = np.array(all_pairings(*[b.node_locs for b in basis]))
    unique_nodes, ele_indexes = global_nodes_from_ele_localnodes(local_points=eval_pts, connectivity=new_topo_lookup)

    S = np.column_stack([x[p] for x, p in zip(xi_locs, parent_connectivity.T)])
    D = np.column_stack([x[p] for x, p in zip(scales, parent_connectivity.T)])
    parent = np.arange(ele_indexes.shape[0]) // np.prod(refinement_factor)

    #a child is its parent's width times its share of it.  Measured against the
    #mean, an even refinement leaves every element at width 1 and so writes no
    #maps at all; an uneven one scales the derivatives each child sees, which is
    #what lets one node derivative serve two children of different widths
    widths = np.array([_element_widths(e) for e in self.elements])[parent] * D
    widths = widths / widths.mean(axis=0)
    widths[np.isclose(widths, 1, rtol=1e-9, atol=0)] = 1.0

    new_pts = [MeshNode(loc=np.zeros(self.fdim), **{uf:np.zeros(self.fdim) for uf in used_fields}) for _ in unique_nodes]
    template = MeshElement(node_indexes=ele_indexes[0], basis_functions=basis)
    new_elements = [MeshElement(node_indexes=ele_pts, basis_functions=basis,
                                param_map=_width_map(template, w))
                    for ele_pts, w in zip(ele_indexes, widths)]
    from HOMER.mesh.field import MeshField  #deferred: field.py imports this module
    new_mesh = MeshField(nodes=new_pts, elements=new_elements)

    #Sample each child over the whole of its own [0, 1], not an interior
    #window: the sample count is fixed at order+2 per direction, so pulling
    #the samples inwards makes the fit extrapolate to the element ends,
    #which is what costs the accuracy.  Widening improves the conditioning
    #of every basis (L3 69 -> 3, H3 4.4e4 -> 1.1e4, B3 1.6e5 -> 4.1e4) and
    #leaves the system full rank, at the price of duplicate rows where
    #neighbouring children meet.
    to_eval = [b.order+2 for b in basis]
    xi_grid = np.column_stack([xi.ravel() for xi in np.mgrid[*[slice(0.0, 1.0, e*1j) for e in to_eval]]])

    new_eles = np.repeat(np.arange(ele_indexes.shape[0]), xi_grid.shape[0])
    new_xi = np.tile(xi_grid, (ele_indexes.shape[0], 1))
    old_eles = parent[new_eles]

    offsets = S[:, None] + xi_grid[None] * D[:, None]
    old_xis = offsets.reshape(-1, self.ndim)
    #new_eles and shape need to line up

    targets = self.evaluate_embeddings_ele_xi_pair(old_eles, old_xis)

    #must happen while self still holds the old nodes.  Both the constraint
    #transfer and the reordering are answering the same question - which old
    #node is this new node - so the map is built once for whichever wants it,
    #and before the fit, which is what the pinned locations constrain.
    parent_of_new = None
    n_old = len(self.nodes)
    interpolatory = basis.interpolatory
    if preserve_fixed_params or resolve_strategy(reorder_nodes) is not None:
        parent_of_new = _parent_node_map(self, basis, basis, ele_indexes, parent_connectivity,
                                         refinement_factor, f_xi_locs, len(new_mesh.nodes))
    if preserve_fixed_params and interpolatory:
        _hold_pinned_locations(self.nodes, new_mesh.nodes, parent_of_new)
        new_mesh.generate_mesh()

    weights, columns = new_mesh.get_xi_weight_blocks(new_eles, new_xi)
    new_mesh.linear_fit(targets, weights, sparse_columns=columns)

    if preserve_fixed_params:
        stats = _transfer_fixed_params(self.nodes, new_mesh.nodes, parent_of_new)
        _report_fixed_param_transfer('refine', stats, interpolatory)

    self.elements = new_mesh.elements


    # spatial_hash = {tuple(np.round(node.loc, ref_res).tolist()):idn for idn, node in enumerate(self.nodes)} 

    self.nodes = new_mesh.nodes
    #new_topo_lookup is the connectivity the refinement built, which is what the
    #lattice ordering needs; self._topo_lookup is still the pre-refinement one
    #until generate_mesh runs below.
    _reorder_nodes(self, reorder_nodes, topo_lookup=new_topo_lookup, generate=False,
                   parent_of_new=parent_of_new, n_old=n_old)
    self.generate_mesh()
    return


def _element_boxes(field: 'MeshField') -> list[tuple]:
    """Every element's ``(component, origin, size)``, reading a conforming mesh's lattice when unset."""
    if all(e.box is not None for e in field.elements):
        return [e.box for e in field.elements]
    _require_conforming(field, 'refine_elements on elements without a box')
    component, coords = element_lattice_coords(field._topo_lookup)
    return [(int(c), tuple(Fraction(int(x)) for x in xs), Fraction(1))
            for c, xs in zip(component, coords)]


def _box_arrays(boxes: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(components, origins, sizes)`` of *boxes*, as float arrays.

    Every box is dyadic -- a lattice cell halved some number of times -- so
    these are exact, and so is comparing them.
    """
    return (np.array([b[0] for b in boxes]),
            np.array([[float(x) for x in b[1]] for b in boxes]),
            np.array([float(b[2]) for b in boxes]))


def _overlapping(arrays, box: tuple, reach: int) -> np.ndarray:
    """The elements overlapping *box* grown by *reach* of its own widths on every side."""
    components, origins, sizes = arrays
    c, origin, size = box
    lo = np.array([float(x) for x in origin]) - reach * float(size)
    hi = lo + (2 * reach + 1) * float(size)
    overlaps = (origins < hi).all(1) & (origins + sizes[:, None] > lo).all(1)
    return np.flatnonzero(overlaps & (components == c))


def _reach(element: MeshElement) -> int:
    """How many cells either side the functions non-zero on a cell reach."""
    return max(int(np.ceil(max(b.node_locs) - min(b.node_locs))) for b in element.basis_functions)


def _balanced(boxes: list[tuple], chosen: set, reach: int) -> set:
    """*chosen*, grown until the mesh refined from it is admissible.

    Refining an element first refines every coarser element overlapping its
    support extension -- the cells the functions non-zero on it reach -- so at
    most two consecutive levels act on any element.  For a basis whose
    functions reach one cell, that is every coarser element touching it,
    across a face, an edge or a corner.
    """
    arrays = _box_arrays(boxes)
    chosen, frontier = set(chosen), set(chosen)
    while frontier:
        grown = {int(n) for e in frontier for n in _overlapping(arrays, boxes[e], reach)
                 if n not in chosen and boxes[n][2] > boxes[e][2]}
        chosen |= grown
        frontier = grown
    return chosen


def refine_elements(self, element_ids) -> np.ndarray:
    """Halve the chosen elements in every direction, leaving the rest as they are.

    The refined mesh carries the truncated hierarchical (THB) space of its
    basis -- see :mod:`HOMER.mesh.hierarchy`.  A function of some level is a
    DOF where its support is refined to that level and no further, and a coarse
    one gives way to the finer ones wherever they take over, so the functions
    still sum to one.  For a nodal basis that leaves the points a child would
    put on a face shared with a coarser element *hanging*: no node, and values
    drawn from the coarser element.  Either way the field is continuous across
    every face, as smooth as its basis, and the children reproduce their
    parents exactly, so there is no fit.

    Refinement is kept admissible: an element is refined together with every
    coarser one its functions reach, so at most two levels act on any element.
    For a basis reaching one cell that is a 2:1 balance across faces, edges
    and corners; ``B3`` reaches three.  Refining again goes another level deep.

    A node of a nodal basis is a value at a point, so it stays: new nodes are
    appended, and every index into the node list survives.  A ``B3`` node is
    the coefficient of one function, which goes once refinement covers its
    whole support, and the node goes with it.  Node derivatives keep their
    parametrisation; a child sees them scaled by its width, as an unevenly
    refined element does.

    :param element_ids:
        Indices of the elements to refine; an empty list only gives every
        element its :attr:`~HOMER.mesh.element.MeshElement.box`.

    :returns:
        The indices of the elements refined, the chosen ones and those the
        balance added, in increasing order.
    """
    template = self.elements[0]
    bases = template.basis_functions
    boxes = _element_boxes(self)
    reach = _reach(template)
    chosen = _balanced(boxes, {int(e) for e in np.atleast_1d(element_ids)}, reach)
    offsets = [o[::-1] for o in itertools.product((0, 1), repeat=template.ndim)]
    bp_inds = tuple(map(tuple, template.BasisProductInds))

    #the refined mesh, an element at a time: kept, or replaced by its children
    new_boxes, sources = [], []
    for ide, (c, origin, size) in enumerate(boxes):
        if ide not in chosen:
            new_boxes.append(boxes[ide])
            sources.append((ide, None))
            continue
        for offset in offsets:
            new_boxes.append((c, tuple(x + size * o / 2 for x, o in zip(origin, offset)), size / 2))
            sources.append((ide, offset))

    #what can change: the children, and every element a function they touch reaches
    old_arrays = _box_arrays(boxes)
    near = {int(n) for e in chosen for n in _overlapping(old_arrays, boxes[e], reach)}
    rebuild = [i for i, (src, offset) in enumerate(sources) if offset is not None or src in near]

    widths = {}
    for box, element in zip(boxes, self.elements):
        c, level, index = Hierarchy.cell(box)
        widths[(c, tuple(i >> level for i in index))] = _element_widths(element) * 2 ** level
    hierarchy = Hierarchy(template, new_boxes, widths)

    #the nodes there are: each is the function its value slot is, alone, in the
    #coarsest element it is so in
    node_of = {}
    by_place = {}
    ele_nodes = _element_node_lists(self)
    for ide in sorted(range(len(boxes)), key=lambda e: -boxes[e][2]):
        c, level, index = Hierarchy.cell(boxes[ide])
        for j, n in enumerate(ele_nodes[ide]):
            if n >= 0:
                place = tuple(Fraction(i) + x for i, x in
                              zip(index, hierarchy.slot_offset[j * len(template.slot_fields)]))
                node_of.setdefault((c, level, place), int(n))
                by_place.setdefault((c, tuple(p / 2 ** level for p in place)), int(n))

    def node_for(key):
        """An existing node for this function, by function, or by place for a nodal basis."""
        c, level, place = key
        if key in node_of:
            return node_of[key]
        if bases.interpolatory:
            return by_place.get((c, tuple(p / 2 ** level for p in place)))
        return None

    #every rebuilt element's slots as THB functions, and the field's own
    #coefficients there, from which each function's value is read where it
    #stands alone
    values, tables = {}, {}
    for i in rebuild:
        src, offset = sources[i]
        local = self.get_element_params(src).reshape(-1, self.fdim)
        if offset is not None:
            local = prolongation(bases, bp_inds, offset) @ local
        table, functions = hierarchy.extraction(Hierarchy.cell(new_boxes[i]))
        tables[i] = (table, functions)
        alone = (table != 0).sum(1) == 1
        for s in np.flatnonzero(alone):
            t = int(np.flatnonzero(table[s])[0])
            values.setdefault(functions[t], local[s] / table[s, t])

    created = {}
    def index_of(key):
        node = node_for(key)
        if node is not None:
            return node
        if key not in created:
            created[key] = len(self.nodes) + len(created)
        return created[key]

    elements = [self.elements[src] for src, _ in sources]
    for i in rebuild:
        table, functions = tables[i]
        pool, param_map = [], {}
        for s, (j, f) in enumerate(template.slots):
            terms = []
            for t in np.flatnonzero(table[s]):
                node = index_of(functions[t][0])
                if node not in pool:
                    pool.append(node)
                terms.append((pool.index(node), functions[t][1], float(table[s, t])))
            param_map[(j, f)] = terms
        identity = all(terms == [(j, f, 1.0)] for (j, f), terms in param_map.items())
        old = self.elements[sources[i][0]]
        elements[i] = MeshElement(node_indexes=pool, basis_functions=bases,
                                  param_map=None if identity else param_map, box=new_boxes[i],
                                  id=old.id if sources[i][1] is None else None)

    #every function's value, on the node that is its own
    nodes = list(self.nodes) + [None] * len(created)
    for key, n in created.items():
        nodes[n] = MeshNode(**{f: values[(key, f)] for f in template.slot_fields})
    for (key, f), value in values.items():
        node = node_for(key)
        if node is not None:
            if f == 'loc':
                nodes[node].loc = np.asarray(value, dtype=np.asarray(nodes[node].loc).dtype)
            else:
                nodes[node][f] = np.asarray(value, dtype=np.asarray(nodes[node][f]).dtype)

    for element, box in zip(elements, new_boxes):
        element.box = box
    self.nodes, self.elements = nodes, elements
    #a coefficient whose function refinement retired has no element left
    self._clean_pts()
    return np.array(sorted(chosen), dtype=int)


def rebase(self, new_basis: BasisGroup, in_place=False, res=10, preserve_fixed_params=True,
           reorder_nodes=True) -> 'MeshField':
    """Convert the mesh to a different set of basis functions.

    Constructs a new :class:`MeshField` with *new_basis*, sampling the
    current mesh on a dense xi grid and linearly fitting the new nodal
    parameters to match the sampled geometry.  This allows, for example,
    converting a trilinear (``L1``) mesh into a cubic-Hermite
    (``H3``) mesh without losing the shape.

    The three-step algorithm is:

    1. Determine the new node locations by evaluating the current mesh at
       the basis node positions of *new_basis*.
    2. Sample a fine xi grid in the current basis to get dense geometry
       samples.
    3. Linearly fit the new nodal parameters to these samples.

    This code explicitely nops if the rebasing is of the same type as the initial mesh.

    :param new_basis:
        The new 1-D bases, one per parametric direction; a
        :class:`~HOMER.basis_definitions.BasisGroup` such as ``H3 * 3``,
        or any list or tuple of bases.
    :param in_place:
        When ``True``, replace this field's nodes and elements with the
        rebased ones, regenerate it, and return *self*.  The default builds
        and returns a new field, leaving this one untouched.
    :param res:
        Number of xi grid points per direction used for the linear fit.
    :param preserve_fixed_params:
        When ``True`` (default), a node of the rebased mesh that sits
        exactly on an existing node inherits that node's
        :attr:`MeshNode.fixed_params`, and a pinned ``loc`` is held through
        the fit when both bases are interpolatory.  Only parameters that exist
        in both bases carry across - rebasing H3 to L1 necessarily drops the
        derivative constraints - and the dropped ones are reported.
    :param reorder_nodes:
        A rebase rebuilds the node list from the new basis, so the order it
        comes back in is an artefact of that, not the order the mesh had.
        When ``True`` (default) the rebased nodes are renumbered by
        :func:`~HOMER.mesh.reordering.reorder_nodes`: a rebase between bases
        with the same nodes - H3 to L1, say - reproduces the numbering the
        mesh had, and one that changes the node set uses
        :data:`~HOMER.mesh.reordering.DEFAULT_NODE_ORDERING`.  Pass ``False``
        to keep the raw ordering, or a strategy name to choose another.  A
        rebase to the basis the mesh already has returns a copy untouched,
        and so is unaffected either way.

    :returns:
        MeshField
            New mesh with the requested basis functions.
    """
    new_basis = BasisGroup(new_basis)
    new_mesh = deepcopy(self)
    if new_basis == self.elements[0].basis_functions:
        return new_mesh
    _require_conforming(self, 'rebase')

    s_hash = {}
    list_locs = [b.node_locs for b in new_basis]
    eval_pts = np.array(all_pairings(*list_locs))
    new_elements = []
    new_pts = []

    used_fields = MeshElement(node_ids=[np.arange(eval_pts.shape[0])], basis_functions=new_basis).used_node_fields

    eval_pts = np.array(eval_pts)

    unique_nodes, ele_indexes = global_nodes_from_ele_localnodes(eval_pts, self._topo_lookup)
    for _ in unique_nodes:
        new_pts.append(MeshNode(loc=np.zeros(self.fdim), **{uf:np.zeros(self.fdim) for uf in used_fields}))
    for ele_pts in ele_indexes:
        new_elements.append(MeshElement(node_indexes=ele_pts, basis_functions=new_basis))
    from HOMER.mesh.field import MeshField  #deferred: field.py imports this module
    new_mesh = MeshField(nodes=new_pts, elements=new_elements)

    egrid = self.xi_grid(res=res, boundary_points=False)
    el = (np.ones((1, res**self.ndim)) * np.arange(len(self.elements))[:, None]).flatten().astype(int)
    xi = np.tile(egrid.reshape(-1, self.ndim), (len(self.elements), 1))
    #rebasing keeps the element topology, so the parent element is the element.
    #The map serves the constraint transfer and the reordering alike, and is
    #built before the fit, which is what the pinned locations constrain.
    parent_of_new = None
    old_basis = self.elements[0].basis_functions
    interpolatory = old_basis.interpolatory and new_basis.interpolatory
    if preserve_fixed_params or resolve_strategy(reorder_nodes) is not None:
        parent_of_new = _parent_node_map(self, old_basis, new_basis, ele_indexes,
                                         np.zeros((len(new_elements), self.ndim), dtype=int),
                                         [1] * self.ndim,
                                         [[Fraction(0), Fraction(1)]] * self.ndim,
                                         len(new_mesh.nodes))
    if preserve_fixed_params and interpolatory:
        _hold_pinned_locations(self.nodes, new_mesh.nodes, parent_of_new)
        new_mesh.generate_mesh()

    weights, columns = new_mesh.get_xi_weight_blocks(el, xi)
    locs = self.evaluate_embeddings_ele_xi_pair(el, xi)
    new_mesh.linear_fit(weight_mat=weights, targets=locs, sparse_columns=columns)

    if preserve_fixed_params:
        stats = _transfer_fixed_params(self.nodes, new_mesh.nodes, parent_of_new)
        _report_fixed_param_transfer('rebase', stats, interpolatory)

    #a rebase keeps the element topology, so this mesh's connectivity is the
    #new one too - and it is populated, which new_mesh's is not yet.
    _reorder_nodes(new_mesh, reorder_nodes, topo_lookup=self._topo_lookup, generate=False,
                   parent_of_new=parent_of_new, n_old=len(self.nodes))
    new_mesh.generate_mesh()

    if in_place:
        self.nodes = new_mesh.nodes
        self.elements = new_mesh.elements
        self.generate_mesh()
        return self
    return new_mesh
