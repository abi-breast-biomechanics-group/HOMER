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
import networkx as nx

from HOMER.utils import build_full_lookup 


def associated_node_index(self, index_list:list, nodes_to_gather: Optional[list] = None, node_by_id = False):
    """
    Given an index list, returns the associated indexes of features in that index in the input param array.
    Used to perform manipulations, and identify which features to fix 
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
    Explores the mesh topology, finding how neighbouring points connet to each other"""
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
    locs = self.evaluate_embeddings_in_every_element(xi_l)
    l_jacs = self.evaluate_jacobians_in_every_element(xi_l)
    n_ele = len(self.elements)
    n_test = len(xi_l)



    locs = np.round(locs, rounding_res)
    _, idx, inv, cnt = np.unique(
        locs, axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True
    )
    faces = []
    bmap = {}


    lookup_arr = np.ones((len(self.elements), self.ndim, 2), dtype=int) * -1

    for idu, cn in enumerate(cnt): #undefined behaviour here, what even is a face for a 2D object
        if cn == 1 and self.ndim == 3: #this region appeared once, so it's a "face"
            ele = idx[idu]//n_test
            test_n = idx[idu]%n_test
            faces.append((int(ele),) + tzip[test_n])
        if cn == 2: #this point appeared multiple times, and defines a transition boundary.
            inds = np.where(inv == idu)[0]
            ele = inds//n_test
            test_n = inds%n_test
            tested = [tzip[t] for t in test_n]
            rel_jac = [l_jacs[t] for t in inds]
            rel_dirs = np.sum(rel_jac[0]*rel_jac[1], axis=0) > 0
            bmap[(ele[0],) + tested[0]] = [(ele[1],) + tested[1], rel_dirs]
            bmap[(ele[1],) + tested[1]] = [(ele[0],) + tested[0], rel_dirs]

            #bmap is extra
            lookup_arr[ele[0], tested[0][0], tested[0][1]] = ele[1]
            # print(ele, tested)
            lookup_arr[ele[1], tested[1][0], tested[1][1]] = ele[0]

        # elif cn > 2:
        #     raise ValueError("Mesh had multiple elements intersecting at a single point")
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


def get_xi_surface_nodes(self, xi_dim, bound_val):
    """
    Given a xi dim, and the boundary value, uses the known mesh topology to find all elements 
    which have no neighbouring elements at that boundary.
    Then uses the xi_weight mat to find the relative weightings of values in the mesh.
    """
    if self.ndim == 2:
        # find the elements
        valid_elements = np.where(self._topo_lookup[:, xi_dim, bound_val] == -1)[0]

        # find the nodes along the 1D edge
        xiq_grid = (np.arange(5) / 4.0).reshape(-1, 1)
        xiq_pt = np.ones(5) * bound_val
        xi_query = np.insert(xiq_grid, xi_dim, xiq_pt, axis=1)
        xi_query = np.tile(xi_query, (len(valid_elements), 1))
        eles_to_q = np.repeat(valid_elements, 5)
        mat = self.get_xi_weight_mat(eles_to_q, xi_query) 
        pams = np.repeat(np.any(mat > 0, axis=0), 3)

        valid_nodes = []
        for idn, node in enumerate(self.nodes):
            append = False 
            sval, pams = pams[:self.fdim], pams[self.fdim:]
            if np.any(sval):
                append = True
            for key, value in node.items():
                l_val = value.flatten().shape[0]
                sval, pams = pams[:l_val], pams[l_val:] 
                if np.any(sval):
                    append = True
            if append:
                valid_nodes.append(idn)

        return valid_elements, valid_nodes
        raise ValueError("everything on a 2d mesh is a surface, but requested to find surface elements")

    #find the elements
    valid_elements = np.where(self._topo_lookup[:, xi_dim, bound_val] == -1)[0]


    #find the nodes
    xiq_grid = np.mgrid[:5,:5]
    xiq_grid = np.column_stack((xiq_grid[0].flatten(), xiq_grid[1].flatten()))/4
    xiq_pt = np.ones((5**2)) * bound_val
    xi_query = np.insert(xiq_grid, xi_dim, xiq_pt, axis=1)
    xi_query = np.tile(xi_query, (len(valid_elements), 1))

    eles_to_q= np.repeat(valid_elements, 5**2)
    mat = self.get_xi_weight_mat(eles_to_q, xi_query) #this should be 1/3rd of the params
    pams = np.repeat(np.any(mat > 0, axis=0), 3)

    valid_nodes = []
    for idn, node in enumerate(self.nodes):
        append = False 
        sval, pams = pams[:self.fdim], pams[self.fdim:]
        if np.any(sval):
            append = True
        for key, value in node.items():
            l_val = value.flatten().shape[0]
            sval, pams = pams[:l_val], pams[l_val:] 
            if np.any(sval):
                append = True
        if append:
            valid_nodes.append(idn)

    return valid_elements, valid_nodes


def get_faces(self, rounding_res = 5) -> list[tuple[int]]:
    """
    Returns all external faces of the current mesh.
    Faces are indicated as tuples (elem_id, dim, {0,1}).
    By definition, A manifold is a face, indicated as (elem_id, -1, -1).
    Faces are determined by spatial hashing of the face center i.e (0.5,0.5, {0,1})
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


def get_colouring_dict(self, fields_seperable=False, seed_matrix=False):
    """
    Returns a colouring dict which describes which mesh parameters will never effect the same output variable.
    The fields seperable option notes if the output of fields produce seperate responses. (e.g. embedding evaluation is seperable, but local jac det is not).
    """
    from sparsejac.sparsejac import _greedy_color, _input_connectivity_from_sparsity
    from jax.experimental.sparse import BCOO
    import scipy

    sf = self.fdim if fields_seperable else 1


    graph_struct = np.zeros((len(self.elements) * sf, len(self.true_param_array)))
    for ide, emap in enumerate(self.ele_map):
        for i in range(sf):
            graph_struct[ide * sf + i, emap.astype(int)[i::sf]] = 1
    graph_struct = graph_struct[:, self.optimisable_param_bool] #remove non-optimisable params
    jax_sparse = BCOO.fromdense(graph_struct)

    graph_struct = scipy.sparse.csr_array(
                (jax_sparse.data, (jax_sparse.indices[:, 0], jax_sparse.indices[:, 1])),
                shape=jax_sparse.shape,
                )

    jacobian = graph_struct #csr_array(graph_struct)
    adj_matrix = (jacobian.T @ jacobian).tocsr()
    adj_matrix.setdiag(0) # Remove self-loops for coloring
    adj_matrix.eliminate_zeros()

    G = nx.from_scipy_sparse_array(adj_matrix)
    colouring_dict = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colours = max(colouring_dict.values()) + 1

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

    # 2. Index-Weighted Seed Matrix (S2) - Data is the node indices
    data_idxs = jnp.array(nodes, dtype=jnp.float32)
    seed_matrix_idxs = jax.experimental.sparse.BCOO(
        (data_idxs, indices), shape=(num_vars, num_colours)
    )

    return colouring_dict, seed_matrix_vals, seed_matrix_idxs
    # td = seed_matrix.todense()
    # breakpoint()
    # seed_matrix = BCOO.fromdense(basis)
    return colouring_dict, seed_matrix
