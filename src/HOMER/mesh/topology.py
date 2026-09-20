"""
topology.py - working out how a mesh is connected, and what that connectivity
implies.

These are the methods of :class:`~HOMER.mesh.field.MeshField` that discover
shared nodes, faces and surfaces, keep the id lookups in step, and colour the
mesh for sparse evaluation.  The first argument is the field itself, and
:mod:`HOMER.mesh.field` binds them into the class.  Unlike the evaluation and
plotting modules these do mutate the field - they are what fills in
``faces``, ``topomap``, ``bmap`` and the id maps.
"""

import itertools
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from HOMER.utils import build_full_lookup 
from HOMER.mesh.reordering import _element_node_lists


def associated_node_index(self, index_list:list, nodes_to_gather: Optional[list] = None, node_by_id = False):
    """
    Given an index list, returns the associated indexes of features in that index in the input param array.
    Used to perform manipulations, and identify which features to fix 

    :param index_list:
        The node field names to gather, e.g. ``['loc', 'du']``.  A node that
        lacks one raises :exc:`ValueError`.
    :param nodes_to_gather:
        Restrict the search to these nodes, given as indices into
        :attr:`nodes` or, with *node_by_id*, as user-assigned node ids.
        ``None`` gathers from every node.
    :param node_by_id:
        Read *nodes_to_gather* as node ids rather than positions.

    :returns:
        One list per gathered node, holding the parameter-vector indices of
        each requested field, in the order *index_list* gives them.

    :raises ValueError:
        If a gathered node does not carry one of the requested fields.

    The parameter vector is briefly overwritten with index values and
    restored before returning, so this is not safe to call from inside a
    traced region.
    """
    true_param_array = np.concatenate([np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()
    self.update_from_params(np.arange(true_param_array.shape[-1]), generate=False)

    if nodes_to_gather is None:
        nodes_to_iter = self.nodes
    else: 
        if node_by_id:
            nodes_to_iter = [self.nodes[self.node_id_to_ind[e]] for e in nodes_to_gather] 
        else:
            nodes_to_iter = [self.nodes[e] for e in nodes_to_gather] 

    param_ids = []
    for idn, node in enumerate(nodes_to_iter):
        node_data = []

        for field in index_list: 
            if field == "loc":
                node_data.append(node.loc)
            else:
                try:
                    node_data.append(node[field].flatten())
                except KeyError:
                    if nodes_to_gather is not None:
                        ele_name = nodes_to_gather[idn]
                    else:
                        ele_name = idn

                    raise ValueError(f"Node {ele_name} did not have the required field '{field}'")
        param_ids.append(node_data)

    self.update_from_params(true_param_array, generate=False)
    return param_ids


def _explore_topology(self, rounding_res=5):
    """
    Explores the mesh topology, finding how neighbouring points connet to each other

    What it finds depends only on where the nodes are and which elements hold
    them, so a regeneration that moved neither has nothing new to say.  That
    is worth checking because :meth:`~HOMER.mesh.field.MeshField.generate_mesh`
    is called far more often than a mesh changes -- once per element added,
    several times per refinement -- and this is the expensive half of it.
    """
    signature = (self.true_param_array.tobytes(),
                 np.asarray(self.ele_map).tobytes(),
                 self.ndim, len(self.elements))
    if (getattr(self, "_topo_signature", None) == signature
            and getattr(self, "topomap", None) is not None):
        return

    if self.ndim == 2:
        xi_l = np.array([
            [0, 0.5], [1, 0.5],
            [0.5, 0], [0.5, 1],
        ])
        tzip = ((0,0), (0,1), (1, 0), (1,1))
    else:
        xi_l = np.array([
            [0, 0.5, 0.5], [1, 0.5, 0.5],
            [0.5, 0, 0.5], [0.5, 1, 0.5],
            [0.5, 0.5, 0], [0.5, 0.5, 1],
        ])
        tzip = ((0,0), (0,1), (1, 0), (1,1), (2, 0), (2, 1))
    n_test = len(xi_l)

    #pinned to the mesh's own parameters rather than whatever is ambient: this
    #can be reached from inside a jit trace, and the topology is a property of
    #where the mesh actually is, not of the tangent being pushed through it
    concrete = np.asarray(self.optimisable_param_array)
    locs = np.round(
        np.asarray(self.evaluate_embeddings_in_every_element(xi_l, fit_params=concrete)),
        rounding_res)
    _, idx, inv, cnt = np.unique(
        locs, axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    faces = []
    bmap = {}

    lookup_arr = np.ones((len(self.elements), self.ndim, 2), dtype=int) * -1

    #a location seen once is a face, seen twice is a boundary between two
    #elements.  Sorting by which location each test point landed on groups the
    #repeats in one pass, rather than rescanning every point per location.
    order = np.argsort(inv.ravel(), kind="stable")
    starts = np.cumsum(cnt) - cnt

    if self.ndim == 3: #undefined behaviour otherwise, what even is a face for a 2D object
        for single in idx[cnt == 1]:
            faces.append((int(single) // n_test,) + tzip[int(single) % n_test])

    shared = np.flatnonzero(cnt == 2)
    if shared.size:
        #only a shared boundary reads the jacobians, and a mesh of one element
        #has none, so the pass that finds them is left until it is wanted
        l_jacs = np.asarray(
            self.evaluate_jacobians_in_every_element(xi_l, fit_params=concrete))

        for first, second in zip(order[starts[shared]], order[starts[shared] + 1]):
            ele_a, side_a = int(first) // n_test, tzip[int(first) % n_test]
            ele_b, side_b = int(second) // n_test, tzip[int(second) % n_test]

            rel_dirs = np.sum(l_jacs[first] * l_jacs[second], axis=0) > 0
            bmap[(ele_a,) + side_a] = [(ele_b,) + side_b, rel_dirs]
            bmap[(ele_b,) + side_b] = [(ele_a,) + side_a, rel_dirs]

            lookup_arr[ele_a, side_a[0], side_a[1]] = ele_b
            lookup_arr[ele_b, side_b[0], side_b[1]] = ele_a

    lookup_arr = jnp.asarray(lookup_arr)
    #face if once, 
    # test_faces = self.get_faces()
    self.faces = faces
    self.bmap = bmap
    self._topo_lookup = lookup_arr
    # print('.')
    # raise ValueError

    # @jax.jit
    def topomap(ele, xi):
        """
        Applies topology mapping using lookup_arr.
        Assumes at most one xi component is out of bounds at a time.
        """
        xi = jnp.asarray(xi)
        ele = jnp.asarray(ele, dtype=jnp.int32)
        # return ele, xi, False
        xi_clipped = jnp.clip(xi, 0.0, 1.0)
        b_lo, b_hi = xi < 0, xi > 1.0
        crossed = b_lo | b_hi
        map_valid = jnp.sum(crossed.astype(jnp.int32), axis=-1) == 1 # Only one bound transition allowed
        where_bound = jnp.atleast_1d(jnp.argmax(crossed.astype(jnp.int32), axis= -1).astype(jnp.int32))
        # jax.debug.print("elem {elem}, xi {xi}, maps {maps}, valid {valid}, where {where}", elem=ele, xi=xi, maps=crossed, valid=map_valid, where=where_bound)
        # side = jnp.where(b_hi[where_bound], 1, 0).astype(jnp.int32)
        side = jnp.take_along_axis(jnp.atleast_2d(b_hi), where_bound[:, None], axis=-1)[..., 0].astype(int)
        new_ele = lookup_arr[ele, where_bound, side]
        map_valid = map_valid & (new_ele != -1)

        xi_mapped = xi + b_lo.astype(xi.dtype) - b_hi.astype(xi.dtype)
        map_valid &= ~jnp.any((xi_mapped < 0) | ( xi_mapped > 1.0)) #check the new location is a valid index.
        out_ele = jnp.where(map_valid, new_ele, ele)
        out_xi = jnp.where(map_valid[:, None], xi_mapped, xi_clipped)
        return out_ele.squeeze(), out_xi.squeeze(), map_valid.squeeze() #squeeze back down to support the 1D outputs

    lookup_full = build_full_lookup(lookup_arr)

    masks_list = list(itertools.product([True, False], repeat=self.ndim))
    masks_list.sort(key=lambda x: (sum(x), x), reverse=True)
    masks = jnp.array(masks_list) 

    @jax.jit
    def topomap_fast_subset(ele, xi):
        """
        Fast O(1) topological mapping that gracefully degrades to sub-constraint 
        mappings if the full diagonal/corner mapping does not exist.
        """
        xi = jnp.asarray(xi)
        ele = jnp.asarray(ele, dtype=jnp.int32)

        ndim = xi.shape[-1]
        batch_shape = xi.shape[:-1]
        K = masks.shape[0]

        b_lo, b_hi = xi < 0.0, xi > 1.0
        shifts = jnp.where(b_lo, 0, jnp.where(b_hi, 2, 1))

        # Identify which dimensions are actually trying to cross a boundary
        is_active = (shifts != 1) 
        needs_map = jnp.sum(is_active, axis=-1) > 0

        # Reshape masks and inputs to broadcast across the K candidate dimension
        # masks_resh: [K, 1..., ndim], shifts_resh: [1, batch..., ndim]
        masks_resh = masks.reshape((K,) + (1,) * len(batch_shape) + (ndim,))
        shifts_resh = jnp.expand_dims(shifts, axis=0)

        # Generate all candidate shift vectors by reverting masked-out dimensions to 1 (inside)
        cand_shifts = jnp.where(masks_resh, shifts_resh, 1)

        # Calculate how many active constraints each candidate actually satisfies
        is_active_resh = jnp.expand_dims(is_active, axis=0)
        satisfied = masks_resh & is_active_resh
        num_satisfied = jnp.sum(satisfied, axis=-1) # Shape: [K, batch...]

        # Evaluate ALL candidates in a single heavily-optimized memory gather
        ele_resh = jnp.expand_dims(ele, axis=0)
        cand_tuple = tuple(cand_shifts[..., i] for i in range(ndim))
        cand_eles = lookup_full[(ele_resh,) + cand_tuple] # Shape: [K, batch...]

        # Score candidates: -1 if invalid mapping, otherwise equal to constraints satisfied
        is_valid_map = (cand_eles != -1)
        scores = jnp.where(is_valid_map, num_satisfied, -1)

        # Find the index of the first candidate with the highest score
        best_idx = jnp.argmax(scores, axis=0) # Shape: [batch...]

        # Extract the winning element, shift vector, and score
        best_ele = jnp.take_along_axis(cand_eles, best_idx[None, ...], axis=0).squeeze(0)
        best_score = jnp.take_along_axis(scores, best_idx[None, ...], axis=0).squeeze(0)

        # Expand best_idx to extract the winning shift vector
        best_idx_expanded = jnp.expand_dims(best_idx, axis=(0, -1))
        best_shift = jnp.take_along_axis(cand_shifts, best_idx_expanded, axis=0).squeeze(0)

        final_valid = jnp.where(needs_map, best_score > 0, True)
        out_ele = jnp.where(final_valid, best_ele, ele)

        did_shift_lo = (best_shift == 0)
        did_shift_hi = (best_shift == 2)

        unmapped_active = is_active & (best_shift == 1)
        xi_base = jnp.where(unmapped_active, jnp.clip(xi, 0.0, 1.0), xi)

        xi_mapped = xi_base + did_shift_lo.astype(xi.dtype) - did_shift_hi.astype(xi.dtype)
        out_xi = jnp.where(final_valid[..., None], xi_mapped, jnp.clip(xi, 0.0, 1.0))

        return out_ele.squeeze(), out_xi.squeeze(), final_valid.squeeze()
    self.topomap = topomap_fast_subset
    self._topo_signature = signature


def get_xi_surface_nodes(self, xi_dim, bound_val):
    """
    Given a xi dim and the boundary value, uses the known mesh topology to find all
    elements which have no neighbouring element at that boundary, and the nodes whose
    basis functions have support on the face those elements expose.

    The bases are a tensor product, so a local node contributes to the
    ``xi_dim = bound_val`` face exactly when its own 1-D basis function is non-zero
    there - the other directions sweep the whole face and drop out.  The face is
    therefore a property of one 1-D basis and the element's node ordering, and needs
    no mesh evaluation.  For an interpolatory basis this is the single layer of nodes
    sitting on the face; for a control-net basis such as
    :data:`~HOMER.basis_definitions.B3` it is every layer with support there,
    which is the set that actually controls the surface.

    :param xi_dim:
        Parametric direction whose boundary to look at, 0-based.
    :param bound_val:
        Which end of that direction: ``0`` for xi = 0, ``1`` for xi = 1.

    :returns:
        ``(valid_elements, valid_nodes)`` -- the indices of the elements with no
        neighbour on that boundary, and the indices of the nodes their bases give
        support on it.
    """
    valid_elements = np.where(self._topo_lookup[:, xi_dim, bound_val] == -1)[0]

    bases = self.elements[0].basis_functions
    basis = bases[xi_dim]
    #the weights of a direction are grouped by node, so a Hermite node's value and
    #derivative weights collapse back onto the one node
    supported = np.asarray(basis.fn(jnp.array([float(bound_val)]))).ravel() != 0
    supported = supported.reshape(len(basis.node_locs), -1).any(axis=1)

    #local nodes are the Fortran-ordered lattice of the 1-D bases, direction 0 fastest
    nodes_per_dim = [len(b.node_locs) for b in bases]
    stride = int(np.prod(nodes_per_dim[:xi_dim]))
    local = np.arange(int(np.prod(nodes_per_dim)))
    on_face = np.where(supported[(local // stride) % nodes_per_dim[xi_dim]])[0]

    ele_nodes = _element_node_lists(self)
    return valid_elements, np.unique(ele_nodes[np.ix_(valid_elements, on_face)])


def get_faces(self, rounding_res = 5) -> list[tuple[int]]:
    """
    Returns all external faces of the current mesh.
    Faces are indicated as tuples (elem_id, dim, {0,1}).
    By definition, A manifold is a face, indicated as (elem_id, -1, -1).
    Faces are determined by spatial hashing of the face center i.e (0.5,0.5, {0,1})

    :param rounding_res:
        Decimal places the face centres are rounded to before hashing, so two
        faces that meet are recognised as the same point.  Only used on the
        first call; afterwards the cached :attr:`faces` is returned.

    :returns:
        One tuple per external face.
    """
    if self.faces is not None:
        return self.faces

    hash_space = {}

    elem_eval = np.array([
        [0, 0.5, 0.5], [1, 0.5, 0.5],
        [0.5, 0, 0.5], [0.5, 1, 0.5],
        [0.5, 0.5, 0], [0.5, 0.5, 1],
    ])
    tzip = ((0,0), (0,1), (1, 0), (1,1), (2, 0), (2, 1))
    faces = []
    for ide, element in enumerate(self.elements):
        if element.ndim == 2:
            faces.append((ide, -1, -1))
            continue

        pts = self.evaluate_embeddings(np.array([ide]), xis=elem_eval)
        for pt, tested in zip(pts, tzip):
            tp = tuple(np.round(np.asarray(pt), rounding_res).tolist())
            space = hash_space.setdefault(tp, [])
            space.append((ide,) + tested)

    calc_face = faces + [k[0] for k in hash_space.values() if len(k) == 1]
    # self.shared_boundaries = [k[0] for k in hash_space.values() if len(k) > 1]
    self.faces = calc_face
    return self.faces


def topo_chain_check(self, ele, xi, at_lo, at_hi):
    """
    quickly iterates through a given point, trying to validly map the point.
    If a point fails, it leaves the boundary active, then moves onto the next point.
    Has a for loop, but XLA compiles down to appropriate quick checks when used in a vmap
    boundary states returns if the point has an active boundary

    :param ele:
        Element index the point sits in.
    :param xi:
        Parametric coordinate of the point.
    :param at_lo:
        Per-direction flags for sitting on the ``xi = 0`` boundary.
    :param at_hi:
        Per-direction flags for sitting on the ``xi = 1`` boundary.

    :returns:
        A boolean per parametric direction, ``True`` where the point is on a
        boundary :func:`topomap` could not map across -- an active boundary,
        with no neighbour on the far side.
    """
    boundary_states = []
    for i_range in range(self.ndim): #iterate over the topological dimension.
        on_boundary = at_hi[i_range] | at_lo[i_range]
        xi_test = jnp.clip(xi, 0, 1).at[i_range].add(0.1 * at_hi[i_range] - 0.1 *at_lo[i_range]) #clip to force only testing pooint of interest.
        _, _, valid = self.topomap(ele, xi_test)
        boundary_states.append((~valid) & on_boundary) 
    return jnp.array(boundary_states)


def _update_id_mappings(self):
    self.node_id_to_ind = {}
    self.element_id_to_ind = {}
    for e, n in [(e, n) for  e , n in enumerate(self.nodes) if n.id is not None]:
        key_in = self.node_id_to_ind.get(n.id, None)
        if key_in is not None:
            raise ValueError(f"Duplicate nodes with the id: {n.id} were added to the mesh")
        self.node_id_to_ind[n.id] = e 

    for e, el in [(e, el) for  e, el in enumerate(self.elements) if el.id is not None]:
        key_in = self.element_id_to_ind.get(el.id, None)
        if key_in is not None:
            raise ValueError(f"Duplicate nodes with the id: {el.id} were added to the mesh")
        self.element_id_to_ind[el.id] = e 


def _clean_pts(self):
    """
    Removes nodes unreferenced by all elements, and then reorderers the associated nodes of each element.
    """

    self._update_id_mappings()

    used_ids = []
    used_points = []
    for element in self.elements:
        if element.used_index:
            used_points.extend(element.nodes)
        else: 
            used_points.extend([self.node_id_to_ind[id] for id in element.nodes])
            used_ids.extend(element.nodes)

    # print(np.sort(np.unique(used_ids)))
    bool_array = np.zeros(len(self.nodes), dtype=bool)
    bool_array[used_points] = True
    new_inds = np.array([0] + np.cumsum(bool_array).tolist())

    for element in self.elements:
        if element.used_index:
            element.nodes = [new_inds[n] for n in element.nodes]

    self.nodes = [n for idn, n in enumerate(self.nodes) if bool_array[idn]]

    self._update_id_mappings()
    self.generate_mesh()


def _colour_ranks(colour_of, num_colours):
    """Where each parameter sits within its own colour, and the way back.

    The index pass of a coloured Jacobian is decoded by division, so its seed
    weights set how sharp the recovered integer is.  Weighting by a global
    parameter index costs precision that weighting by the parameter's rank
    within its colour does not, and the rank identifies it just as well once
    the colour is known -- which it is, because the colour is the row.

    :param colour_of:
        The colour of each parameter, indexed by parameter.
    :param num_colours:
        How many colours the mesh needed.

    :returns:
        ``(rank, members)`` -- every parameter's rank, and a
        ``(num_colours, widest colour)`` table giving the parameter holding
        each rank, padded with -1.
    """
    order = np.argsort(colour_of, kind="stable")
    widths = np.bincount(colour_of, minlength=num_colours)
    ranks_sorted = np.arange(colour_of.size) - np.repeat(np.cumsum(widths) - widths, widths)

    rank = np.empty(colour_of.size, dtype=int)
    rank[order] = ranks_sorted

    members = np.full((num_colours, max(widths.max(), 1)), -1, dtype=int)
    members[colour_of[order], ranks_sorted] = order
    return rank, members


def _greedy_colour_largest_first(adj_matrix):
    """Greedily colour a parameter adjacency graph, highest degree first.

    ``networkx.coloring.greedy_color`` gives an equivalent answer, but building
    the graph object it wants costs an order of magnitude more than colouring
    it does.  Walking the CSR rows directly skips that.

    :param adj_matrix:
        Symmetric ``csr_array`` with the self-loops already removed; an entry
        means the two parameters meet in some element.

    :returns:
        The colour of each parameter, as an integer array.
    """
    indptr, indices = adj_matrix.indptr, adj_matrix.indices
    colours = np.full(adj_matrix.shape[0], -1, dtype=int)

    for node in np.argsort(-np.diff(indptr), kind="stable"):
        used = colours[indices[indptr[node]:indptr[node + 1]]]
        used = used[used >= 0]
        #the smallest colour no neighbour has taken; only the first
        #len(used) + 1 can be free, so nothing beyond that need be considered
        free = np.ones(used.size + 1, dtype=bool)
        free[used[used <= used.size]] = False
        colours[node] = np.argmax(free)

    return colours


def get_colouring_dict(self, fields_seperable=False, seed_matrix=False):
    """
    Returns a colouring dict which describes which mesh parameters will never effect the same output variable.
    The fields seperable option notes if the output of fields produce seperate responses. (e.g. embedding evaluation is seperable, but local jac det is not).

    :param fields_seperable:
        Treat each field component as producing its own response, giving one
        graph row per component rather than one per element.  Embedding
        evaluation is separable this way; a local Jacobian determinant is not.
    :param seed_matrix:
        Also build the seed matrices a coloured Jacobian is reconstructed
        from.

    :returns:
        The colouring as ``{parameter index: colour}``, or, with
        *seed_matrix*, a tuple of that dict, a unit-valued seed matrix and one
        weighted by each parameter's rank within its colour -- both ``BCOO``
        of shape ``(n_parameters, n_colours)``.  :func:`_colour_ranks` turns
        those ranks back into parameters.
    """
    import scipy.sparse

    sf = self.fdim if fields_seperable else 1

    # One row per element, or per field component of an element when separable,
    # holding the optimisable parameters that row can reach.  Fixed parameters
    # map to -1 and drop out.
    ele_map = np.asarray(self.ele_map).astype(int)
    n_rows = ele_map.shape[0] * sf
    slots = ele_map.reshape(ele_map.shape[0], -1, sf).transpose(0, 2, 1).reshape(n_rows, -1)

    optimisable = np.asarray(self.optimisable_param_bool)
    position = np.full(len(self.true_param_array), -1)
    position[optimisable] = np.arange(optimisable.sum())

    columns = position[slots].ravel()
    rows = np.repeat(np.arange(n_rows), slots.shape[1])[columns >= 0]
    columns = columns[columns >= 0]

    incidence = scipy.sparse.csr_array(
        (np.ones(columns.size), (rows, columns)),
        shape=(n_rows, int(optimisable.sum())),
    )
    adj_matrix = (incidence.T @ incidence).tocsr()
    adj_matrix.setdiag(0) # Remove self-loops for coloring
    adj_matrix.eliminate_zeros()

    colouring = _greedy_colour_largest_first(adj_matrix)
    colouring_dict = {int(node): int(colour) for node, colour in enumerate(colouring)}
    num_colours = int(colouring.max()) + 1

    if not seed_matrix:
        return colouring_dict

    num_vars = max(colouring_dict.keys()) + 1
    nodes = np.array(list(colouring_dict.keys()))
    colours = np.array(list(colouring_dict.values()))

    # The coordinates (row, color) remain identical for both matrices
    indices = jnp.column_stack((nodes, colours))

    # 1. Standard Value Seed Matrix (S1) - Data is all 1s
    data_vals = jnp.ones(len(nodes), dtype=jnp.float32)
    seed_matrix_vals = jax.experimental.sparse.BCOO(
        (data_vals, indices), shape=(num_vars, num_colours)
    )

    # 2. Rank-Weighted Seed Matrix (S2) - Data is the rank within the colour,
    # which is smaller than the parameter index by a factor of the colour
    # count, and so is recovered by the decode's division far more sharply.
    colour_of = np.empty(num_vars, dtype=int)
    colour_of[nodes] = colours
    rank, _ = _colour_ranks(colour_of, num_colours)

    data_idxs = jnp.array(rank[nodes], dtype=jnp.float32)
    seed_matrix_idxs = jax.experimental.sparse.BCOO(
        (data_idxs, indices), shape=(num_vars, num_colours)
    )

    return colouring_dict, seed_matrix_vals, seed_matrix_idxs
