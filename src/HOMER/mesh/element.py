"""
element.py - :class:`MeshElement`, the tensor-product connectivity of a mesh.

An element links a set of :class:`~HOMER.mesh.node.MeshNode` objects through a
product of 1-D bases, one per parametric direction, and works out the ordering
of the weights that the evaluation functions expect.
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

    * H3Basis × H3Basis → 2 × 2 = 4 nodes (2-D)
    * H3Basis × H3Basis × H3Basis → 2 × 2 × 2 = 8 nodes (3-D)
    * L2Basis × L2Basis → 3 × 3 = 9 nodes (2-D)

    Parameters
    ----------
    basis_functions:
        The 1-D bases of the element, one per parametric direction, defining the
        parametric-direction interpolation.  E.g.
        ``H3Basis * 2`` for a 2-D cubic-Hermite element.
    node_indexes:
        Zero-based integer indices into the parent mesh's ``nodes`` list.
        Exactly one of *node_indexes* or *node_ids* must be given.
    node_ids:
        User-supplied node identifiers (alternative to *node_indexes*).
    BP_inds:
        Pre-computed basis-product index pairs.  Computed automatically
        when ``None``; supply a cached value to skip recomputation.
    id:
        Optional element identifier.

    Attributes
    ----------
    ndim : int
        Parametric dimensionality (2 or 3).
    nodes : list
        The ordered node references (indexes or ids).
    basis_functions : BasisGroup
        The 1-D bases, one per parametric direction.
    used_node_fields : list[str]
        Derivative field names (``'du'``, ``'dv'``, …) that each node must
        carry for this element's basis.
    BasisProductInds : list[tuple[int, ...]]
        Ordered index pairs/triplets defining the tensor-product weight
        computation.
    num_nodes : int
        Total number of nodes in this element.
    """

    def __init__(self, basis_functions: BasisGroup, node_indexes: Optional[list[int]] = None, 
                 node_ids: Optional[list] = None, BP_inds: Optional = None, id=None):
        """Initialise a :class:`MeshElement`.

        Parameters
        ----------
        basis_functions:
            The 1-D bases of the element, one per parametric direction (1, 2
            or 3 of them).  Accepts a :class:`~HOMER.basis_definitions.BasisGroup`
            (``H3Basis * 2 + L1Basis``), a list or tuple of bases, or a single
            basis for a 1-D element.
        node_indexes:
            Zero-based indices into the parent mesh's node list.
        node_ids:
            User-supplied node identifiers.
        BP_inds:
            Pre-computed basis-product index pairs (optional optimisation).
        id:
            Optional element identifier.

        Raises
        ------
        ValueError
            If neither *node_indexes* nor *node_ids* is provided, or if
            both are provided.
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

        self.scale_factors = None #classic H3 representation. 

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

        :params derivs_struct: The calculated derivative pairs to evaluate.
        :params order_dict: The ordering to follow
        """

        indexed_keys = [
            (i, (abs(lst[0]), (order_dict[tuple(lst[1:])] if len(lst) > 1 else 0)))
            for i, lst in enumerate(derivs_struct)
        ]
        
        indexed_keys.sort(key=lambda x: x[1])
        return [i for i,  _ in indexed_keys]
