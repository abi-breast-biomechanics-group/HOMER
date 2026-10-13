"""
element.py - :class:`MeshElement`, the tensor-product connectivity of a mesh.

An element links a set of :class:`~HOMER.mesh.node.MeshNode` objects through a
product of 1-D bases, one per parametric direction, and works out the ordering
of the weights that the evaluation functions expect.  Its parameter map says
how each of those weights is drawn from the nodes.
"""

from typing import Optional
from functools import reduce
from itertools import groupby

import numpy as np

from HOMER.basis_definitions import BasisGroup, DERIV_ORDER


class MeshElement:
    """A single high-order mesh element linking nodes through tensor-product basis functions.

    A :class:`MeshElement` combines a list of :class:`MeshNode` references with
    a *group* of 1-D basis functions (one per parametric direction) to define a
    2-D manifold surface element or a 3-D volume element.

    The number of nodes required per element equals the product of the numbers
    of 1-D basis nodes:

    * H3 × H3 → 2 × 2 = 4 nodes (2-D)
    * H3 × H3 × H3 → 2 × 2 × 2 = 8 nodes (3-D)
    * L2 × L2 → 3 × 3 = 9 nodes (2-D)

    :param basis_functions:
        The 1-D bases of the element, one per parametric direction, defining the
        parametric-direction interpolation.  E.g.
        ``H3 * 2`` for a 2-D cubic-Hermite element.
    :param node_indexes:
        Zero-based integer indices into the parent mesh's ``nodes`` list.
        Exactly one of *node_indexes* or *node_ids* must be given.
    :param node_ids:
        User-supplied node identifiers (alternative to *node_indexes*).
    :param BP_inds:
        Pre-computed basis-product index pairs.  Computed automatically
        when ``None``; supply a cached value to skip recomputation.
    :param param_map:
        How each element parameter is drawn from the nodes; see
        :attr:`param_map`.
    :param id:
        Optional element identifier.

    :ivar ndim: int
        Parametric dimensionality (2 or 3).
    :ivar nodes: list
        The element's local node pool, as indexes or ids.  Without a
        :attr:`param_map` it is the basis node lattice, direction 0 fastest.
    :ivar param_map: dict | None
        The general linear map from the element's parameters to its nodes'
        parameters.  An element parameter -- a *slot* -- is a ``(j, field)``
        pair: lattice position *j* and a field of :attr:`slot_fields`.  Each
        slot maps to a list of ``(k, field, weight)`` terms, and takes the
        weighted sum of those fields of the nodes at pool positions *k*.  A
        weight scales every component of the field alike.  ``None`` is the
        identity, ``(j, field) -> [(j, field, 1.0)]``; otherwise every slot
        must be given.
    :ivar slot_fields: list[str]
        ``'loc'`` followed by :attr:`used_node_fields`, the fields a lattice
        position holds a slot for.
    :ivar box: tuple | None
        ``(component, origin, size)``, where the element sits in the lattice of
        the mesh it was locally refined from: the connected component, the
        rational coordinate of its ``xi = 0`` corner and its rational width,
        the same in every direction.  Set by
        :func:`~HOMER.mesh.refinement.refine_elements`; ``None`` until then.
    :ivar basis_functions: BasisGroup
        The 1-D bases, one per parametric direction.
    :ivar used_node_fields: list[str]
        Derivative field names (``'du'``, ``'dv'``, …) that each node must
        carry for this element's basis.
    :ivar BasisProductInds: list[tuple[int, ...]]
        Ordered index pairs/triplets defining the tensor-product weight
        computation.
    :ivar num_nodes: int
        Total number of nodes in this element.
    """

    def __init__(self, basis_functions: BasisGroup, node_indexes: Optional[list[int]] = None,
                 node_ids: Optional[list] = None, BP_inds: Optional = None,
                 param_map: Optional[dict] = None, box: Optional[tuple] = None, id=None):
        """Initialise a :class:`MeshElement`.

        :param basis_functions:
            The 1-D bases of the element, one per parametric direction (1, 2
            or 3 of them).  Accepts a :class:`~HOMER.basis_definitions.BasisGroup`
            (``H3**2 * L1``), a list or tuple of bases, or a single
            basis for a 1-D element.
        :param node_indexes:
            Zero-based indices into the parent mesh's node list.
        :param node_ids:
            User-supplied node identifiers.
        :param BP_inds:
            Pre-computed basis-product index pairs (optional optimisation).
        :param param_map:
            The element's parameter map, or ``None`` for the identity.
        :param box:
            Where the element sits in the lattice it was refined from.
        :param id:
            Optional element identifier.

        :raises ValueError: If neither *node_indexes* nor *node_ids* is provided, or if
            both are provided, or if *param_map* does not give exactly the
            element's slots.
        """
        if node_ids is None and node_indexes is None:
            raise ValueError("An element must be associated with a list of nodes, either by index or node id")
        elif node_ids is not None and node_indexes is not None:
            raise ValueError("Both node indexes and node ids were provided - only one should be given.")

        nodes = node_indexes if node_indexes is not None else node_ids
        self.used_index = node_indexes is not None

        self.nodes = nodes
        #accepts a BasisGroup, a list/tuple of bases, or a bare basis for 1-D
        self.basis_functions = BasisGroup(basis_functions)
        if not 1 <= len(self.basis_functions) <= 3:
            raise ValueError("An element is a tensor product of 1, 2 or 3 bases; "
                             f"got {len(self.basis_functions)}: {self.basis_functions!r}")
        self.ndim: int = len(self.basis_functions)
        self.n_in_dim = [sum([l[0]=='x' for l in b.weights]) for b in self.basis_functions]

        self.get_used_fields()
        self.BasisProductInds = self._calc_basis_product_inds() if BP_inds is None else BP_inds
        self.id = id
        self.num_nodes = int(np.prod([len(b.node_locs) for b in self.basis_functions]))
        self.slot_fields = ['loc'] + list(self.used_node_fields)

        if param_map is not None and set(param_map) != set(self.slots):
            raise ValueError("A param_map gives terms for every slot of the element and no "
                             f"others; expected the slots {self.slots}")
        self.param_map = param_map
        self.box = box

    @property
    def slots(self) -> list[tuple[int, str]]:
        """Every ``(j, field)`` slot, in the order the evaluators take parameters."""
        return [(j, f) for j in range(self.num_nodes) for f in self.slot_fields]

    def terms(self, slot: tuple[int, str]) -> list[tuple[int, str, float]]:
        """The ``(k, field, weight)`` terms *slot* is the weighted sum of."""
        if self.param_map is None:
            return [(slot[0], slot[1], 1.0)]
        return self.param_map[slot]

    @property
    def derivative_orders(self) -> np.ndarray:
        """``(n_slots, ndim)``, 1 where a slot is a derivative along that direction.

        Read off the 1-D weight each slot takes in each direction, so it holds for
        any mix of bases without decoding field names.
        """
        return np.array([[b.weights[i].startswith('d') for b, i in zip(self.basis_functions, inds)]
                         for inds in self.BasisProductInds], dtype=int)

    @property
    def lattice_nodes(self) -> list[Optional[int]]:
        """The pool position of the node at each lattice position, or ``None``.

        A node sits at lattice position *j* when the value there is that node's
        value and nothing else.  A position whose value is a combination -- a
        hanging node -- has no node of its own.
        """
        nodes = []
        for j in range(self.num_nodes):
            terms = self.terms((j, 'loc'))
            whole = len(terms) == 1 and terms[0][1] == 'loc' and terms[0][2] == 1
            nodes.append(terms[0][0] if whole else None)
        return nodes

    def get_used_fields(self):
        """
        Calculates the used node fields for field objects.
        This represents the increasing derivative pattern du -> du, dw, dudw -> du ... dudvdw
        """
        raw_fields = [b.node_fields for b in self.basis_functions if b.node_fields is not None]
        sorted_objects = sorted(raw_fields, key=lambda x: x.__class__.__name__)
        grouped = [list(group) for _, group in groupby(sorted_objects, key=lambda x: x.__class__)]
        if len(grouped) == 0:
            self.used_node_fields = []
            return
        fields = reduce(lambda x,y:x+y,[f.get_needed_fields() for f in [reduce(lambda x,y: x+y, g) for g in grouped]])
        self.used_node_fields = [fields] if isinstance(fields, str) else fields

    def _calc_basis_product_inds(self):
        """
        Given the definition of the basis functions, this creates the indexes used to populate the weighting matrix.
        The weighting matrix is defined as the outer product of the basis functions for each element.
        :params b_def: the definition of the parameters associated with the basis functions.
        """
        dim_step = [1] + np.cumprod(self.n_in_dim)[:-1].tolist()
        n_param  = [len(b.weights) for b in self.basis_functions]

        
        if len(self.basis_functions) == 3:
            w_mat = np.mgrid[:n_param[0], :n_param[1], :n_param[2]].astype(int) # this is the pairing.
        elif len(self.basis_functions) == 2:
            w_mat = np.mgrid[:n_param[0], :n_param[1]].astype(int) # this is the pairing.
        elif len(self.basis_functions) == 1:
            w_mat = np.arange(n_param[0]).astype(int)
            return [(i,) for i in np.arange(n_param[0])] #just cycle the params
        l_mat = np.column_stack([w.flatten() for w in w_mat])

        
        ind_names = [0] + np.cumsum([np.any([f[:2]=='dx' for f in bparam.weights]) for bparam in self.basis_functions]).tolist()
        
        keyvals = []
        for pairing in l_mat:
            id = 0
            deriv = []
            for idind, ind in enumerate(pairing):
                l_name = self.basis_functions[idind].weights[ind]
                id += int(l_name[-1]) * dim_step[idind] #encodes the surface representation
                if l_name[0] == 'd':
                    deriv.append(ind_names[idind])
            keyvals.append([id] + deriv)

        # breakpoint()


        sorted  = self.argsort_derivs(keyvals, DERIV_ORDER)
        new_ind_pairs = [tuple(l_mat[i].tolist()) for i in sorted]
        # print([keyvals[i] for i in sorted])
        return new_ind_pairs
        return [tuple(l_mat[i].tolist()) for i in range(len(keyvals))]

    def argsort_derivs(self, derivs_struct: list[list[str]], order_dict: dict[tuple]):
        """
        Given a derivs struct defined iternally, returns the canonical ordering according to a given order dict.

        :param derivs_struct: The calculated derivative pairs to evaluate.
        :param order_dict: Maps a derivative tuple to its rank, giving the
            ordering to follow.

        :returns: Indices that put *derivs_struct* into canonical order,
            sorted by derivative magnitude and then by *order_dict*.
        """

        indexed_keys = [
            (i, (abs(lst[0]), (order_dict[tuple(lst[1:])] if len(lst) > 1 else 0)))
            for i, lst in enumerate(derivs_struct)
        ]
        
        indexed_keys.sort(key=lambda x: x[1])
        return [i for i,  _ in indexed_keys]
