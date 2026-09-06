"""
refinement.py - subdividing a mesh, and moving it to a different basis.

:func:`refine` and :func:`rebase` are methods of
:class:`~HOMER.mesh.field.MeshField` that live in this module; the first
argument is the field itself, and :mod:`HOMER.mesh.field` binds them into the
class.  The private helpers around them work out which node of the refined
mesh came from which node of the original, which is what lets a pinned
parameter survive the operation.
"""

import logging
from copy import deepcopy
from fractions import Fraction
from typing import Optional, TYPE_CHECKING

import numpy as np
import pyvista as pv

from HOMER.basis_definitions import Basis, BasisGroup
from HOMER.mesh.node import MeshNode
from HOMER.mesh.element import MeshElement
from HOMER.topomap_operations import global_nodes_from_ele_localnodes, refine_connectivity
from HOMER.utils import all_pairings

if TYPE_CHECKING:
    from HOMER.mesh.field import MeshField


MAX_XI_DENOMINATOR = 10**6


def _basis_node_fractions(basis: Basis, max_denominator: int = MAX_XI_DENOMINATOR) -> list[Fraction]:
    """Exact rational node locations of a 1-D basis.

    ``node_locs`` are fixed constants of the basis, so the snap to a rational is exact
    in intent.  Note that they are not required to lie inside ``[0, 1]``:
    :class:`~HOMER.basis_definitions.B3Basis` places its shared control points
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

    Parameters
    ----------
    old_bases, new_bases:
        1-D bases before and after the operation, one per direction.
    xi_breaks:
        Per-direction list of :class:`~fractions.Fraction` sub-element
        boundaries within the parent element, length ``refinement[d] + 1``.
    refinement:
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


def _element_node_indices(field: 'MeshField') -> np.ndarray:
    """``(n_elements, n_nodes_per_element)`` array of global node indices."""
    field._update_id_mappings()
    rows = []
    for element in field.elements:
        if element.used_index:
            rows.append(list(element.nodes))
        else:
            rows.append([field.node_id_to_ind[node_id] for node_id in element.nodes])
    return np.array(rows, dtype=int)


def _parent_node_map(old_field: 'MeshField', old_bases, new_bases, new_ele_nodes: np.ndarray,
                     parent_coords: np.ndarray, refinement, xi_breaks, n_new_nodes: int) -> np.ndarray:
    """Map every new global node to the old global node it coincides with.

    Element identity is preserved by both refinement and rebasing, so a new
    node is identified by *which basis slot of which parent element* it
    occupies.  That is resolved purely with integer indices and exact rational
    node locations, so nothing here depends on the mesh coordinates or their
    dtype.

    Returns
    -------
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
        old_node_of = _element_node_indices(old_field)
        rows = np.repeat(parent_ele[:, None], n_new_local, axis=1)[valid]
        parent_of_new[new_ele_nodes[valid]] = old_node_of[rows, old_local[valid]]
    return parent_of_new


def _transfer_fixed_params(old_nodes, new_nodes, parent_of_new: np.ndarray, interpolatory: bool) -> tuple[int, int, int]:
    """Carry :attr:`MeshNode.fixed_params` across to the coincident new nodes.

    ``loc`` constraints are re-asserted with the original value for
    interpolatory bases, so a pinned landmark survives the least-squares fit
    exactly.  Derivative constraints keep their fitted value, because their
    magnitude is element-scale dependent (a refined element's ``du`` is
    legitimately the parent's divided by the refinement factor).  For a
    control-net basis no value is restored at all - only the flag.

    Returns
    -------
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
            if param == 'loc':
                values = np.asarray(old_node.loc)[inds] if interpolatory else None
                new_node.fix_parameter('loc', values=values, inds=inds)
            elif param in new_node:
                new_node.fix_parameter(param, inds=inds)
            else:
                dropped += 1 #the new basis carries no such derivative
                continue
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
           clean_nodes = True, plot=False, preserve_fixed_params = True):
    """Subdivide every element, increasing the mesh resolution.

    Each existing element is replaced by ``refinement_factor ** ndim``
    (or the equivalent for *by_xi_refinement*) smaller elements sharing
    intermediate nodes.  Derivative values at the new nodes are obtained
    by evaluating the current basis functions.

    Exactly one of *refinement_factor* or *by_xi_refinement* must be
    provided.

    Parameters
    ----------
    refinement_factor:
        Integer ≥ 2 that subdivides each parametric direction uniformly.
        For example, ``refinement_factor=2`` splits a single element into
        8 sub-elements in 3-D (2 × 2 × 2).
    by_xi_refinement:
        Tuple of 1-D arrays, one per parametric direction, specifying the
        xi values at which to place the new element boundaries.  Each
        array must start with 0 and end with 1.
    clean_nodes:
        When ``True`` (default), remove unreferenced nodes after
        refinement.
    preserve_fixed_params:
        When ``True`` (default), any node of the refined mesh that sits
        exactly on an existing node inherits that node's
        :attr:`MeshNode.fixed_params`.  Fixed ``loc`` values are restored
        verbatim for interpolatory bases so pinned landmarks do not drift
        with the fit; constraints with no counterpart in the refined mesh
        are dropped and reported.

    Raises
    ------
    AssertionError
        If both *refinement_factor* and *by_xi_refinement* are given, or
        if *refinement_factor* < 2.
    """
    assert not(refinement_factor is not None and by_xi_refinement is not None), "Refinement factor and refining by defined xi are mutually exclusive."

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

    new_pts = [MeshNode(loc=np.zeros(self.fdim), **{uf:np.zeros(self.fdim) for uf in used_fields}) for _ in unique_nodes]
    new_elements = [MeshElement(node_indexes=ele_pts, basis_functions=basis) for ele_pts in ele_indexes]
    from HOMER.mesh.field import MeshField  #deferred: field.py imports this module
    new_mesh = MeshField(nodes=new_pts, elements=new_elements)
    new_mesh.generate_mesh()

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
    old_eles = new_eles // np.prod(refinement_factor)
    S = np.column_stack([x[p] for x, p in zip(xi_locs, parent_connectivity.T)])
    D = np.column_stack([x[p] for x, p in zip(scales, parent_connectivity.T)])

    offsets = S[:, None] + xi_grid[None] * D[:, None]
    old_xis = offsets.reshape(-1, self.ndim)
    #new_eles and shape need to line up

    targets = self.evaluate_embeddings_ele_xi_pair(old_eles, old_xis)

    if plot:
        test = pv.lines_from_points(np.array(targets))
        test['data'] = np.arange(targets.shape[0])
        test.plot(render_lines_as_tubes=True, cmap='jet', line_width=15)

    w_mat = new_mesh.get_xi_weight_mat(new_eles, new_xi)
    # plt.imshow(w_mat);plt.show()
    new_mesh.linear_fit(targets, w_mat) 

    if preserve_fixed_params: #must happen while self still holds the old nodes
        parent_of_new = _parent_node_map(self, basis, basis, ele_indexes, parent_connectivity,
                                         refinement_factor, f_xi_locs, len(new_mesh.nodes))
        interpolatory = basis.interpolatory
        stats = _transfer_fixed_params(self.nodes, new_mesh.nodes, parent_of_new, interpolatory)
        _report_fixed_param_transfer('refine', stats, interpolatory)

    self.elements = new_mesh.elements


    # spatial_hash = {tuple(np.round(node.loc, ref_res).tolist()):idn for idn, node in enumerate(self.nodes)} 

    self.nodes = new_mesh.nodes
    self.generate_mesh()
    return


def rebase(self, new_basis: BasisGroup, in_place=False, res=10, preserve_fixed_params=True) -> 'MeshField':
    """Convert the mesh to a different set of basis functions.

    Constructs a new :class:`MeshField` with *new_basis*, sampling the
    current mesh on a dense xi grid and linearly fitting the new nodal
    parameters to match the sampled geometry.  This allows, for example,
    converting a trilinear (``L1Basis``) mesh into a cubic-Hermite
    (``H3Basis``) mesh without losing the shape.

    The three-step algorithm is:

    1. Determine the new node locations by evaluating the current mesh at
       the basis node positions of *new_basis*.
    2. Sample a fine xi grid in the current basis to get dense geometry
       samples.
    3. Linearly fit the new nodal parameters to these samples.

    This code explicitely nops if the rebasing is of the same type as the initial mesh.

    Parameters
    ----------
    new_basis:
        The new 1-D bases, one per parametric direction; a
        :class:`~HOMER.basis_definitions.BasisGroup` such as ``H3Basis * 3``,
        or any list or tuple of bases.
    in_place:
        Currently unused (future: modify *self* rather than returning a
        new object).
    res:
        Number of xi grid points per direction used for the linear fit.
    preserve_fixed_params:
        When ``True`` (default), a node of the rebased mesh that sits
        exactly on an existing node inherits that node's
        :attr:`MeshNode.fixed_params`.  Only parameters that exist in both
        bases carry across - rebasing H3 to L1 necessarily drops the
        derivative constraints - and the dropped ones are reported.

    Returns
    -------
    MeshField
        New mesh with the requested basis functions.
    """
    new_basis = BasisGroup(new_basis)
    new_mesh = deepcopy(self)
    if new_basis == self.elements[0].basis_functions:
        return new_mesh

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
    w_mat = new_mesh.get_xi_weight_mat(el, xi)
    locs = self.evaluate_embeddings_ele_xi_pair(el, xi)
    new_mesh.linear_fit(weight_mat=w_mat, targets=locs)

    if preserve_fixed_params: #rebasing keeps the element topology, so the parent element is the element
        old_basis = self.elements[0].basis_functions
        parent_of_new = _parent_node_map(self, old_basis, new_basis, ele_indexes,
                                         np.zeros((len(new_elements), self.ndim), dtype=int),
                                         [1] * self.ndim,
                                         [[Fraction(0), Fraction(1)]] * self.ndim,
                                         len(new_mesh.nodes))
        interpolatory = old_basis.interpolatory and new_basis.interpolatory
        stats = _transfer_fixed_params(self.nodes, new_mesh.nodes, parent_of_new, interpolatory)
        _report_fixed_param_transfer('rebase', stats, interpolatory)

    new_mesh.generate_mesh()

    if in_place:
        self.nodes = new_mesh.nodes
        self.elements = new_mesh.elements
        self.generate_mesh()
        return self
    return new_mesh
