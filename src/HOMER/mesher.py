"""
mesher.py – Core mesh data structures and operations for HOMER.

This module defines the four primary classes that make up every HOMER mesh:

* :class:`MeshNode` – a single mesh node storing a physical location and any
  Hermite derivative vectors required by the chosen basis functions.
* :class:`MeshElement` – connects a set of :class:`MeshNode` objects through a
  product of 1-D basis functions (one per parametric dimension).
* :class:`MeshField` – a collection of :class:`MeshNode` and
  :class:`MeshElement` objects that can evaluate and optimise any
  vector-valued field over the mesh topology.  The primary geometry field
  (world-space XYZ coordinates) is always a :class:`MeshField`.
* :class:`Mesh` – extends :class:`MeshField` and owns a dictionary of named
  secondary :class:`MeshField` objects (accessible via ``mesh['name']``).
  Secondary fields can represent fibre directions, velocities, stresses, or
  any other spatially varying quantity.

Typical import::

    from HOMER import Mesh, MeshNode, MeshElement
    from HOMER.basis_definitions import H3Basis, L1Basis
"""

import logging
from copy import copy
import itertools
import functools

from os import PathLike
from typing import Optional, Callable
import typing
import numpy as np
import jax.numpy as jnp
import jax
import pyvista as pv
from matplotlib import pyplot as plt
from copy import deepcopy

from functools import reduce, partial
from itertools import groupby, combinations_with_replacement, product

from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import coo_array

from HOMER.basis_definitions import N2_weights, N3_weights, AbstractBasis, BasisGroup, DERIV_ORDER, EVAL_PATTERN
from HOMER.jacobian_evaluator import jacobian
from HOMER.utils import spheres_to_polydata, vol_hexahedron, make_tiling, h_tform, all_pairings, block_diagonal_jacobian, jax_aknn
from HOMER.mesh_decorators import expand_wide_evals, wide_eval

pv.global_theme.allow_empty_mesh = True

class MeshNode(dict):
    """A mesh node that stores a physical location and associated derivative data.

    :class:`MeshNode` subclasses :class:`dict` so that derivative quantities
    (``du``, ``dv``, ``dw``, ``dudv``, …) required by higher-order basis
    functions can be stored as named entries.  All values must be
    :class:`numpy.ndarray` objects of the same length as ``loc``.

    For a 2-D manifold mesh with cubic-Hermite basis in both directions
    (``H3Basis``, ``H3Basis``), each node must carry ``du``, ``dv``, and
    ``dudv`` derivatives::

        node = MeshNode(
            loc=np.array([0.0, 0.0, 1.0]),
            du=np.zeros(3),
            dv=np.zeros(3),
            dudv=np.zeros(3),
        )

    For a 3-D volume mesh with ``H3Basis`` in all three directions, the
    additional derivatives ``dw``, ``dudw``, ``dvdw``, and ``dudvdw`` are
    also required::

        node = MeshNode(
            loc=np.array([0.0, 0.0, 1.0]),
            du=np.zeros(3), dv=np.zeros(3), dw=np.zeros(3),
            dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3),
            dudvdw=np.zeros(3),
        )

    Parameters
    ----------
    loc:
        Physical-space coordinates of the node, shape ``(fdim,)``.
    id:
        Optional unique identifier.  When provided, nodes can be referenced
        by ID rather than list index in a :class:`MeshElement`.
    **kwargs:
        Named derivative arrays, e.g. ``du``, ``dv``, ``dw``, ``dudv``, …
        All values must be ``numpy.ndarray`` (or list / JAX array, which
        are automatically converted).

    Attributes
    ----------
    loc : numpy.ndarray
        Physical-space coordinates, shape ``(fdim,)``.
    id :
        The node identifier (or ``None``).
    fixed_params : dict
        Maps parameter name → array of fixed component indices.  Populated
        by :meth:`fix_parameter`.
    """

    def __init__(self, loc, id=None, **kwargs):
        """Initialise a :class:`MeshNode`.

        Parameters
        ----------
        loc:
            Physical-space coordinates, shape ``(fdim,)``.
        id:
            Optional unique identifier.
        **kwargs:
            Named derivative arrays (``du``, ``dv``, ``dw``, …).
            Each value must be an array of the same length as ``loc``.

        Raises
        ------
        ValueError
            If any keyword-argument value is not an array-like type.
        """
        self.loc = np.asarray(loc)
        self.id = id
        self.update(kwargs)
        self.fixed_params = {}

        for key, value in kwargs.items():
            if isinstance(value, list):
                self[key] = np.asarray(value).copy()
            elif isinstance(value, jnp.ndarray):
                self[key] = np.asarray(value).copy()
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"Only np.ndarray are valid additional data, but found key: {key}, value: {value} pair")
            else:
                self[key] = np.array(value).copy()

    def fix_parameter(self, param_names: list | str, values: Optional[list[np.ndarray]|np.ndarray]=None, inds: Optional[list[int]] = None) -> None:
        """Mark one or more node parameters as fixed (non-optimisable).

        Fixed parameters are excluded from the optimisable parameter vector
        exposed by :class:`MeshField`.  Optionally, the parameter can also be
        set to a specified value at the same time.

        Parameters
        ----------
        param_names:
            Name or list of names of the parameters to fix, e.g.
            ``'loc'``, ``'du'``, ``['loc', 'dv']``.
        values:
            Optional value(s) to assign at the time of fixing.  Must match
            the shape implied by ``inds`` (or the full parameter dimension
            when ``inds`` is ``None``).
        inds:
            Component indices to fix within the parameter array (e.g.
            ``[0, 2]`` to fix the *x* and *z* components of ``loc``).
            When ``None``, all components are fixed.
        """
        l_dim = self.loc.shape[0]

        if inds is not None:
            inds = np.array(inds).astype(int)
        if isinstance(param_names, str):
            param_names = [param_names]
        if not isinstance(values, list):
            values = [values] * len(param_names)

        for idp, param in enumerate(param_names):
            if inds is None:
                inds = np.arange(l_dim).astype(int)
            if param in self.fixed_params:
                self.fixed_params[param] = np.union1d(self.fixed_params[param], inds)
            else:
                self.fixed_params[param] = inds

            if values[idp] is not None:
                if param == 'loc':
                    self.loc[inds] = values[idp]
                else:
                    self[param][inds] = values[idp]

    def get_optimisability_arr(self):
        """
        Returns the optimisable status of all data stored on the node.
        """
        l_dim = self.loc.shape[0]
        free_loc = np.ones(l_dim) 
        free_loc[self.fixed_params.get('loc', [])] = 0
        list_data = [free_loc]
        for key in self.keys():
            free_key = np.ones(l_dim)
            free_key[self.fixed_params.get(key, [])] = 0
            list_data.append(free_key)
        return np.concatenate(list_data, axis=0)


    def plot(self, scene: Optional[pv.Plotter] = None) -> pv.Plotter | None:
        """
        Draws the node, and any quantities, to a pyvista plotter.
        :param scene: An existing pyvista scene to draw too - if given will not draw the plot.
        """
        s = pv.Plotter() if scene is None else scene
        
        s.add_mesh(pv.PolyData(self.loc), point_size=5, render_points_as_spheres=True)
        label_locs = []
        label_names = []
        for key, value in self.items():
            s.add_mesh(pv.lines_from_points(np.array((self.loc, self.loc + value))))
            label_locs.append(self.loc + value)
            label_names.append(key)
        s.add_point_labels(label_locs, labels=label_names)
        if scene is None:
            s.show()
            return None
        return s

    def unfix_params(self):
        self.fixed_params = {}


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
        A sequence of 1-D basis classes (length 2 or 3) defining the
        parametric-direction interpolation.  E.g.
        ``(H3Basis, H3Basis)`` for a 2-D cubic-Hermite element.
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
        The sequence of 1-D basis classes.
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
            A sequence of 1-D basis classes (length 2 or 3).
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
        self.basis_functions = basis_functions
        self.ndim: int = len(self.basis_functions)
        self.n_in_dim = [sum([l[0]=='x' for l in b.weights]) for b in self.basis_functions]

        self.get_used_fields()
        self.BasisProductInds = self._calc_basis_product_inds() if BP_inds is None else BP_inds
        self.id = id
        self.num_nodes = int(np.prod([len(b.node_locs) for b in self.basis_functions]))


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

        
        if   len(self.basis_functions) == 3:
            w_mat = np.mgrid[:n_param[0], :n_param[1], :n_param[2]].astype(int) # this is the pairing.
        elif len(self.basis_functions) == 2:
            w_mat = np.mgrid[:n_param[0], :n_param[1]].astype(int) # this is the pairing.
        l_mat = np.column_stack([w.flatten() for w in w_mat])

        
        ind_names  = [0] + np.cumsum([np.any([f[:2]=='dx' for f in bparam.weights]) for bparam in self.basis_functions]).tolist()
        
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



@expand_wide_evals
class MeshField:
    """A collection of :class:`MeshNode` and :class:`MeshElement` objects representing a single field.

    :class:`MeshField` is the base class for both the primary geometry of a
    :class:`Mesh` and for any secondary fields (fibre directions, stresses,
    velocities, etc.) created with :meth:`Mesh.new_field`.

    A :class:`MeshField` owns:

    * A **node list** storing the field's degrees of freedom (parameter values
      and, for Hermite bases, derivative vectors).
    * An **element list** defining how nodes are connected and which basis
      functions to use for interpolation.
    * **Compiled JAX functions** (built by :meth:`generate_mesh`) for fast
      evaluation, differentiation, and optimisation.

    The ``@expand_wide_evals`` decorator automatically adds
    ``*_in_every_element`` and ``*_ele_xi_pair`` variants for every method
    decorated with ``@wide_eval``.

    Parameters
    ----------
    nodes:
        List of :class:`MeshNode` objects.  May be ``None`` when building a
        mesh incrementally with :meth:`add_node`.
    elements:
        List (or single instance) of :class:`MeshElement` objects.
    jax_compile:
        When ``True``, JIT-compiles evaluation functions at construction time
        (recommended for iterative fitting loops).

    Attributes
    ----------
    nodes : list[MeshNode]
        All nodes belonging to this field.
    elements : list[MeshElement]
        All elements belonging to this field.
    fdim : int
        Physical dimensionality of the field values (e.g. 3 for XYZ).
    ndim : int
        Parametric dimensionality (2 or 3).
    true_param_array : numpy.ndarray
        Flat vector of *all* nodal parameters (free and fixed).
    optimisable_param_array : numpy.ndarray
        Subset of *true_param_array* that is not fixed.
    optimisable_param_bool : numpy.ndarray
        Boolean mask selecting optimisable parameters from
        *true_param_array*.
    ele_map : numpy.ndarray
        ``(n_elements, n_params_per_element)`` index array mapping element
        slots to positions in *true_param_array*.
    """

    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        """Initialise a :class:`MeshField`.

        Parameters
        ----------
        nodes:
            Node list (or ``None`` for incremental construction).
        elements:
            Element or list of elements (or ``None``).
        jax_compile:
            If ``True``, JIT-compile internal evaluation functions
            immediately after construction.
        """
        
        ######### topology of the mesh
        self.nodes: list[MeshNode] = [] if nodes is None else (nodes if isinstance(nodes, list) else [nodes])
        self.elements: list[MeshElement] = [] if elements is None else (elements if isinstance(elements, list) else [elements])

        self.node_id_to_ind = {}
        self.element_id_to_ind = {}

        ######### initialising values to be calculated
        self.elem_evals: Optional[Callable] = None
        self.elem_deriv_evals: Optional[Callable] = None

        self.generate_weight_matrix: Optional[Callable] = None

        self.faces = None

        ######### optimisation
        self.true_param_array: Optional[np.ndarray] = None
        self.optimisable_param_array: Optional[np.ndarray] = None
        self.optimisable_param_bool: Optional[np.ndarray] = None
        self.ele_map: Optional[np.ndarray] = None
        
        ######### field stuff
        self.fdim = None
        ######### Compilation flags
        self.compile = jax_compile
        if not len(self.nodes) == 0 and not len(self.elements) == 0:
            self.generate_mesh()
    
        
    ################################## MAIN FUNCTIONS
    @wide_eval
    def evaluate_embeddings(self, *a, **kw): #placeholder for later func definition
        """Evaluate the field at parametric coordinates within one or more elements.

        This is a placeholder that is replaced by a compiled JAX function when
        :meth:`generate_mesh` is called.  The full signature after
        initialisation is::

            evaluate_embeddings(element_ids, xis, fit_params=None) -> jnp.ndarray

        Parameters
        ----------
        element_ids:
            1-D array of integer element indices, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.
        fit_params:
            Override of the current :attr:`optimisable_param_array`.
            When ``None`` the stored parameter values are used.

        Returns
        -------
        jnp.ndarray
            Field values at the requested locations, shape ``(n_pts, fdim)``.

        Notes
        -----
        The ``@expand_wide_evals`` class decorator automatically creates two
        additional variants:

        * ``evaluate_embeddings_in_every_element(xis)`` – evaluates the same
          grid of xi points in *every* element and stacks the results.
        * ``evaluate_embeddings_ele_xi_pair(element_ids, xis)`` – evaluates
          each ``(element, xi)`` pair independently (equivalent signature to
          the base function but without batching).
        """
        if not typing:
            raise RuntimeError('Called evaluate_embeddings before initialisation')
        return

    @wide_eval
    def evaluate_deriv_embeddings(self, *a, **kw): #placeholder for later func definition
        """Evaluate a specified partial derivative of the field.

        This is a placeholder replaced at :meth:`generate_mesh` time.  The
        full signature is::

            evaluate_deriv_embeddings(element_ids, xis, derivs, fit_params=None)
                -> jnp.ndarray

        Parameters
        ----------
        element_ids:
            1-D integer array, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.
        derivs:
            Derivative order per parametric direction, e.g. ``[1, 0]`` for
            ∂/∂u in a 2-D element or ``[0, 0, 1]`` for ∂/∂w in a 3-D one.
        fit_params:
            Optional override of :attr:`optimisable_param_array`.

        Returns
        -------
        jnp.ndarray
            Derivative field values, shape ``(n_pts, fdim)``.
        """
        if not typing:
            raise RuntimeError('Called evaluate_deriv_embeddings before initialisation')
        return

    def evaluate_element_embeddings(self, element_id, xis, fit_params=None):
        """Evaluate the embedding for a single element identified by its ID.

        Parameters
        ----------
        element_id:
            The user-assigned element ID (not the list index).
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.
        fit_params:
            Optional parameter override.

        Returns
        -------
        jnp.ndarray
            Field values, shape ``(n_pts, fdim)``.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        return self.evaluate_embeddings([self.element_id_to_ind[element_id]], xis, fit_params=fit_params)
     
    @wide_eval
    def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray, fit_params=None) -> np.ndarray:
        """Return the surface normal vectors at parametric coordinates.

        Only valid for 2-D manifold meshes (``ndim == 2``).  The normal is
        computed as the cross product of the two surface tangent vectors.

        Parameters
        ----------
        element_ids:
            1-D integer array of element indices, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, 2)``.
        fit_params:
            Optional override of :attr:`optimisable_param_array`.

        Returns
        -------
        jnp.ndarray
            Normal vectors (not normalised), shape ``(n_pts, 3)``.

        Raises
        ------
        ValueError
            If called on a 3-D volume mesh.
        """

        if self.ndim == 3: 
            raise ValueError("Normals aren't defined on a volume mesh")
        if fit_params is None:
            fit_params = self.optimisable_param_array

        d0 = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1], fit_params) 
        d1 = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0], fit_params)
        return jnp.cross(d0, d1)

    @wide_eval
    def eval_numeric_jac(self, element_ids, xis, locals=None, step=2e-1, fit_params=None):
        """ 
        Evaluates the jacobian at a set of xis within an element
        Uses numeric derivatives, useful when the underlying mesh field has zero derivative boundaries.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array

        if locals is None:
            locals = self.evaluate_embeddings(element_ids, xis)
        
        flip = jnp.where(jnp.atleast_2d(xis) > 0.5, -1, 1)
        
        if self.ndim == 2:
            du = (self.evaluate_embeddings(element_ids, xis + jnp.array([step, 0])[None] * flip[:, 0], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 0][:, None, None]
            dv = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, step])[None] * flip[:, 1], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 1][:, None, None]
            jmats = jnp.concatenate((du, dv), axis=1)
        if self.ndim == 3:
            du = (self.evaluate_embeddings(element_ids, xis + jnp.array([step, 0, 0])[None] * flip[:, 0], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 0][:, None, None]
            dv = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, step, 0])[None] * flip[:, 1], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 1][:, None, None]
            dw = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, 0, step])[None] * flip[:, 2], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 2][:, None, None]

            jmats = jnp.concatenate((du, dv, dw), axis=1) / step
        # return jmats
        return jnp.swapaxes(jmats, -1,-2) #differing jacobin implementation.
        

    @wide_eval
    def evaluate_jacobians(self, element_ids, xis, fit_params=None):
        """Evaluate the Jacobian matrix of the embedding at parametric coordinates.

        Returns ∂x/∂ξ, the matrix mapping parametric-space tangent vectors to
        physical-space tangent vectors.  Rows correspond to physical directions
        (x, y, z) and columns to parametric directions (u, v[, w]).

        Parameters
        ----------
        element_ids:
            1-D integer array, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.
        fit_params:
            Optional override of :attr:`optimisable_param_array`.

        Returns
        -------
        jnp.ndarray
            Jacobian matrices, shape ``(n_pts, fdim, ndim)``.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array

        if self.ndim == 2:
            du = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
            dv = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1], fit_params=fit_params).reshape(-1, 1, self.fdim)
            jmats = jnp.concatenate((du, dv), axis=1)
        if self.ndim == 3:

            du = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
            dv = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
            dw = self.evaluate_deriv_embeddings(element_ids, xis, [0, 0, 1], fit_params=fit_params).reshape(-1, 1, self.fdim)
            jmats = jnp.concatenate((du, dv, dw), axis=1)
        # return jmats
        return jnp.swapaxes(jmats, -1,-2) #differing jacobin implementation.

    ################################## CONVENIENCE
    def xi_grid(self, res: int, dim=None, surface=False, boundary_points=True, lattice=None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return a regular grid of parametric (xi) coordinates.

        Creates a uniform Cartesian grid of xi points for use with
        :meth:`evaluate_embeddings_in_every_element` or for passing to
        :meth:`get_xi_weight_mat`.

        Parameters
        ----------
        res:
            Number of grid points along each parametric direction.  The
            total number of points is ``res ** ndim`` (or ``res ** 2`` when
            returning surface faces of a volume mesh).
        dim:
            Dimensionality of the grid (2 or 3).  Defaults to
            :attr:`ndim`.
        surface:
            For a 3-D mesh, return only points on the six element faces
            rather than the full interior grid.
        boundary_points:
            When ``False``, exclude xi = 0 and xi = 1 from the grid (useful
            to avoid double-counting shared element boundaries).
        lattice:
            Optional ``(xn, yn)`` tiling definition for hexagonal surface
            patterns.

        Returns
        -------
        numpy.ndarray
            Grid points, shape ``(res**ndim, ndim)`` (or, when *lattice* is
            provided and *surface* is ``True``, a ``(pts, connectivity)``
            tuple).
        """
        dim = self.ndim if dim is None else dim

        b_off = 0 if boundary_points else 1
        if not boundary_points:
            res = res + 1 #boundary points drops a res
        if dim == 2:
            if lattice is None:
                X,Y = (np.mgrid[
                    0:res - b_off,
                    0:res - b_off,
                ] + b_off * 0.5)/(res - 1)
                return np.column_stack((X.flatten(), Y.flatten()))
            else:
                return make_tiling(*lattice)
        else:
            if not surface:
                X,Y,Z = (np.mgrid[
                    0:res - b_off,
                    0:res - b_off,
                    0:res - b_off,
                ] + b_off * 0.5 )/(res - 1)
                return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            else:
                if lattice is None:
                    raw_x = np.array([x.flatten() for x in np.mgrid[:res, :res]/(res-1)])
                else:
                    raw_x, connectivity = make_tiling(*lattice) 
                    raw_x = raw_x.T

                zero_r = np.zeros(shape=raw_x[0].shape[0])
                ones_r = np.ones(shape=raw_x[0].shape[0])

                arrays = [
                        np.column_stack((zero_r, raw_x[0], raw_x[1])),
                        np.column_stack((ones_r, raw_x[0], raw_x[1])),
                        np.column_stack((raw_x[0], zero_r, raw_x[1])),
                        np.column_stack((raw_x[0], ones_r, raw_x[1])),
                        np.column_stack((raw_x[0], raw_x[1], zero_r)),
                        np.column_stack((raw_x[0], raw_x[1], ones_r)),
                ]
                if lattice is None:
                    return np.concatenate(arrays) 
                return np.concatenate(arrays), connectivity

    def gauss_grid(self, ng):
        """Return a tensor-product Gauss quadrature grid.

        Parameters
        ----------
        ng:
            * **int** – return 1-D Gauss points for a single direction.
            * **list[int]** – tensor-product grid; e.g. ``[3, 3]`` for a
              2-D surface integration or ``[3, 3, 3]`` for a 3-D volume.

        Returns
        -------
        Xi : numpy.ndarray
            Gauss point locations, shape ``(n_gauss, ndim)`` or ``(n_gauss,)``
            for the 1-D case.
        W : numpy.ndarray
            Corresponding quadrature weights, shape ``(n_gauss,)``.

        Raises
        ------
        ValueError
            If ``ng`` has more than 3 entries or is of an unsupported type.
        """

        if isinstance(ng, int):
            return GAUSS[ng]
        elif isinstance(ng, list):
            if len(ng) > 3:
                raise ValueError('Gauss points for 4 dimensions and above not supported')
            if len(ng) == 2:
                Xi1, W1 = self.gauss_grid(ng[0])
                Xi2, W2 = self.gauss_grid(ng[1])
                Xi1g, Xi2g = np.meshgrid(Xi1.flatten(), Xi2.flatten())
                Xi1 = np.array([Xi1g.flatten(), Xi2g.flatten()]).T
                W1g, W2g = np.meshgrid(W1.flatten(), W2.flatten())
                W1 = W1g.flatten() * W2g.flatten()
                return Xi1, W1
            elif len(ng) == 3:
                Xi1, W1 = self.gauss_grid(ng[0])
                Xi2, W2 = self.gauss_grid(ng[1])
                Xi3, W3 = self.gauss_grid(ng[2])
                gindex = np.mgrid[0:ng[0], 0:ng[1], 0:ng[2]]
                gindex = np.array([gindex[n].flatten() for n in [0,1,2]]).T #doesn't seem to work as default
                Xi = np.array([
                    Xi1[gindex[:, 0]], Xi2[gindex[:, 1]], Xi3[gindex[:, 2]]])[:, :, 0].T
                W = np.array([
                    W1[gindex[:, 0]], W2[gindex[:, 1]], W3[gindex[:, 2]]]).T.prod(1)
                return Xi, W
            
        raise ValueError('Invalid number of gauss points')

    def get_element_params(self, ele_num: int) -> np.ndarray:
        """
        returns the flat vector of node parameters associated with this element.
        """
        return self.true_param_array[self.ele_map[ele_num].astype(int)]

    def update_from_params(self, inp_params, generate=True):
        """
            Updates all nodes with data from an input param array.

            :param inp_params: the input params to update the mesh with
            :param generate: whether to rebuild the mesh after updating.
        """

        if len(inp_params) == len(self.optimisable_param_array):
            params = self.true_param_array.copy()
            params[self.optimisable_param_bool] = inp_params 
        elif len(inp_params) == len(self.true_param_array):
            params = inp_params
            # self.true_param_array = inp_params
        else:
            raise ValueError("Input param array was provided that did not match either that set of parameters, or the optimisable subset of parameters")

        for node in self.nodes:
            node.loc, params = params[:self.fdim], params[self.fdim:]
            for key, value in node.items():
                l_val = value.flatten().shape[0]
                flat_node = node[key].ravel()
                flat_node[:], params = params[:l_val], params[l_val:] 
        if generate:
            self.generate_mesh()

    ################################## MORPHIC INTERFACE COMPATIBILITY
    def generate_mesh(self) -> None:
        """
        Builds the mesh representation on call.

        This code is responsible for handling on-the-fly functions, and the generation of the
        'fast' pathway jax.numpy array representation.

        """

        self.fdim = self.nodes[0].loc.shape[0]
        self.ndim = self.elements[0].ndim
        self.true_param_array = np.concatenate([np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()
        self.optimisable_param_bool = np.concatenate([node.get_optimisability_arr() for node in self.nodes], axis=0).astype(bool)
        self.optimisable_param_array = self.true_param_array[self.optimisable_param_bool]


        ########## build the lookup from the input values.
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

        self.update_from_params(np.arange(self.true_param_array.shape[-1]), generate=False)

        ele_maps = []
        for ide, element in enumerate(self.elements):
            param_ids = []
            
            if element.used_index:
                nodes_to_iter = [self.nodes[e] for e in element.nodes]
            else:
                nodes_to_iter = [self.get_node(e) for e in element.nodes]

            for idn, node in enumerate(nodes_to_iter):
                param_ids.append(node.loc)
                for field in element.used_node_fields: 
                    try:
                        param_ids.append(node[field].flatten())
                    except KeyError:
                        raise ValueError(f"Node {idn} of element: {ide} did not have the required field '{field}'")
            ele_maps.append(np.concatenate(param_ids))
        self.ele_map = np.array(ele_maps)
        self.update_from_params(self.true_param_array, generate=False)

        self._generate_elem_functions()
        self._generate_elem_deriv_functions()
        self._generate_eval_function()
        self._generate_deriv_function()
        self._generate_weight_function()
        self._explore_topology()

    def add_node(self, node:MeshNode) -> None:
        """
        Add a node to the node list.
        """
        self.nodes.append(node)
        # self.generate_mesh()

    def add_element(self, element:MeshElement) -> None:
        """
        Adds an element to the element list.
        """
        self.elements.append(element)
        self.generate_mesh()

    def get_element(self, element_ids: list) -> list[MeshElement]:
        """
        Returns the element with the associated id.
        """
        if not isinstance(element_ids, list):
            return self.get_element([element_ids])[0]
        return [self.elements[self.element_id_to_ind[id]] for id in element_ids]

    def get_node(self, node_ids: list | int | str) -> list[MeshNode] | MeshNode:
        if not isinstance(node_ids, list):
            return self.get_node([node_ids])[0]
        return [self.nodes[self.node_id_to_ind[id]] for id in node_ids]

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

    def unfix_mesh(self):
        """
        Removes all fixed parameters in the mesh, and regenerates the mesh structure.
        """
        for node in self.nodes:
            node.unfix_params()
        self.generate_mesh()

    ################################## PLOTTING
    def get_surface(self, element_ids: Optional[np.ndarray] = None, res:int = 20, just_faces=False, tiling=None) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
        """
        Returns a set of points evaluated over the mesh surface.
        """
        ele_iter  = [element_ids] if not isinstance(element_ids, list) else element_ids
        elements_to_iter = self.elements if element_ids is None else ele_iter
        if not just_faces:
            grid = self.xi_grid(res=res, ndim=self.ndim, surface=True)
            if element_ids is not None:
                all_points = []
                for ne, e in enumerate(elements_to_iter):
                    all_points.append(self.evaluate_embeddings(np.array([ne]), grid))
                return np.concatenate(all_points, axis=0) 
            else:
                return self.evaluate_embeddings_in_every_element(grid)
        else:
            face_pts = []

            if self.ndim == 3:
                faces = self.get_faces()
                if tiling is None:
                    xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)
                    for face in faces:
                        grid_def = xi3grid[face[1], face[2]]
                        face_pts.append(self.evaluate_embeddings(np.array([face[0]]),grid_def))
                    return np.concatenate(face_pts, axis=0)
                
                c = []
                xi3grid, connectivity = self.xi_grid(res=res, dim=3, surface=True, lattice=tiling)
                connectivity = connectivity.reshape(-1, 3)
                xi3grid = xi3grid.reshape(3,2,-1,3)
                l_xi = xi3grid.shape[2]

                for idf, face in enumerate(faces):
                    grid_def = xi3grid[face[1], face[2]]
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]),grid_def))
                    c.append([[0, idf * l_xi, idf * l_xi]] + connectivity)
                return np.concatenate(face_pts, axis=0), np.concatenate(c, axis=0)
            else:
                if tiling is None:
                    xi2grid = self.xi_grid(res=res, dim=2)
                    return np.asarray(self.evaluate_embeddings_in_every_element(xi2grid))
                xi2grid, connectivity = self.xi_grid(res=res, dim=2, lattice=tiling)
                lc = len(xi2grid)
                c = np.concatenate([connectivity.reshape(-1, 3) + [[0, idc * lc, idc * lc]] for idc in range(len(self.elements))], axis=0)
                return np.asarray(self.evaluate_embeddings_in_every_element(xi2grid)), c


    def get_hex_surface(self, element_ids, tiling = (10, 6)) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns lines evaluating a hexagon tiling of the element surface

        :params tiling: the repetitions of the underlying unit surface (5/3 ratio "looks good")
        """
        surface_points, single_face_connectivity = self.get_surface(element_ids, just_faces=True, tiling=tiling)
        return surface_points, single_face_connectivity.astype(int)

    def get_triangle_surface(self, element_ids: Optional[np.ndarray] = None, res:int = 20) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a set of points evaluated over the mesh surface, and triangles to create the surface.

        :returns surface pts: Surface points evaluated over the mesh.
        :returns tris: the triangles creatign the mesh surface.
        """
        base_0 = np.array([0, 1, res])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(res - 1) * res)[:, None, None] 
        base_1 = np.array([res, 1, res + 1])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(res - 1) * res)[:, None, None] 
        surface_pts = self.get_surface(element_ids, just_faces=True, res=res)
        n_surfaces = surface_pts.shape[0]/(res**2)
        tris = (np.concatenate((base_0.reshape((-1,3)), base_1.reshape((-1,3))))[None] + np.arange(n_surfaces)[:, None, None] * res**2).reshape(-1,3)

        return surface_pts, tris


    def get_lines(self, element_ids: Optional[list[int]|int|np.ndarray] = None, res=20) -> pv.PolyData:
        """
        Returns a pv.PolyData object containing lines defining the edges of the mesh surface.
        """

        line_points = np.empty((0, 3))
        connectivity = np.empty((0, 3))
        blank_connectivity = np.column_stack((
            2 * np.ones(res - 1),
            np.arange(0, res - 1),
            np.arange(1, res)
        ))

        ele_iter  = [element_ids] if not isinstance(element_ids, list) else element_ids
        elements_to_iter = self.elements if element_ids is None else ele_iter #if we assume that all elements must be the same because it's easier.

        n_dim = self.elements[0].ndim
        residual_size = n_dim - 1 
        vals = [0, 1]
        combs = list(product(vals, repeat=residual_size)) #the combinations 
        all_xis = []

        total_ls = 0
        for i in range(n_dim):
            d = list(range(n_dim))
            d.pop(i)
            for comb in combs:
                xi_list = [0] * n_dim
                for cs, ind in zip(comb, d):
                    xi_list[ind] = cs * np.ones(res)
                xi_list[i] = np.linspace(0, 1, res)
                xis = np.column_stack(xi_list)
                all_xis.append(xis)

                l_pts = total_ls
                connectivity = np.concatenate((
                    connectivity,
                    blank_connectivity + [0, l_pts, l_pts],
                ))
                total_ls += xis.shape[0]

        flat_xis = np.array(all_xis).reshape(-1, n_dim)
                
        lc = flat_xis.shape[0]
        n_ele = len(self.elements) 
        ele_up = lc * np.arange(n_ele)[None, :, None] * [0, 1, 1]
        long_connectivity = (connectivity[:, None] + ele_up).reshape(-1, 3)
        line_points = np.asarray(self.evaluate_embeddings_in_every_element(flat_xis)) #.reshape(n_ele, -1 , 3)[:2].reshape(-1, 3)

        mesh = pv.PolyData(
            line_points, 
            lines=long_connectivity.astype(int),
        )

        return mesh

    def _explore_topology(self, rounding_res=10):
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



        pv.Plotter

        locs = np.round(locs, rounding_res)
        _, idx, inv, cnt = np.unique(
            locs, axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True
        )
        faces = []
        bmap = {}

        
        lookup_arr = np.ones((len(self.elements), self.fdim, 2), dtype=int) * -1

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


            elif cn > 2:
                raise ValueError("Mesh had multiple elements intersecting at a single point")
        lookup_arr = jnp.asarray(lookup_arr)
        #face if once, 
        # test_faces = self.get_faces()
        self.faces = faces
        self.bmap = bmap
        # print('.')
        # raise ValueError

        @jax.jit
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
            map_valid = jnp.sum(crossed.astype(jnp.int32)) == 1 # Only one bound transition allowed
            where_bound = jnp.argmax(crossed.astype(jnp.int32)).astype(jnp.int32)
            # jax.debug.print("elem {elem}, xi {xi}, maps {maps}, valid {valid}, where {where}", elem=ele, xi=xi, maps=crossed, valid=map_valid, where=where_bound)
            side = jnp.where(b_hi[where_bound], 1, 0).astype(jnp.int32)
            new_ele = lookup_arr[ele, where_bound, side]
            map_valid = map_valid & (new_ele != -1)
            xi_mapped = xi + b_lo.astype(xi.dtype) - b_hi.astype(xi.dtype)
            out_ele = jnp.where(map_valid, new_ele, ele)
            out_xi = jnp.where(map_valid, xi_mapped, xi_clipped)
            return out_ele, out_xi, map_valid

        self.topomap = topomap

    def get_faces(self, rounding_res = 10) -> list[tuple[int]]:
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

    def plot(self, scene:Optional[pv.Plotter] = None,
             node_colour='r', node_size=10,
             labels = False, tiling=(10, 6), 
             mesh_colour: str | np.ndarray ='gray', mesh_opacity=0.1, mesh_width = 2, mesh_col_scalar_name="Field",
             line_colour: str | np.ndarray ='black', line_opacity=1, line_width=2, line_col_scalar_name="Field",
             elem_labels=False,
             render_name:Optional[str] = None,
             ):
        """
        Draws the mesh as a pyvista scene.

        :param scene: A pyvista scene, if provided will not call .show().
        :param node_colour: The colour to draw the node values.
        :param node_size: The size of the node points.
        :param labels: Whether to label the node numbers.
        :param res: The resolution of the surface mesh.
        :param mesh_color: The mesh surface colour.
        :param mesh_opacity: The mesh surface opacity.
        :param elem_labels: Whether to label the mesh elements.

        """

        if labels:
            if not node_size == 10:
                logging.warning("Requested non-default node size, but setting node_size to 0 to allow labels to be visualised")
            node_size = 0

        is_tag = render_name is not None
        render_name = "" if render_name is None else render_name
        l_tag = render_name + "_lines" if is_tag else None 
        n_tag = render_name + "_nodes" if is_tag else None 
        h_tag = render_name + "_hexes" if is_tag else None 
        v_tag = render_name + "_nnums" if is_tag else None 
        e_tag = render_name + "_enums" if is_tag else None 
        render_name = None if is_tag else render_name


        #evaluate the mesh surface and evaluate all of the elements
        lines = self.get_lines()
        node_dots = np.array([node.loc for node in self.nodes])
        s=pv.Plotter() if scene is None else scene

        if isinstance(line_colour, np.ndarray):
            lines[line_col_scalar_name] = line_colour

        s.add_mesh(lines, line_width=line_width, color=line_colour if line_colour is not isinstance(line_colour, np.ndarray) else None, name=l_tag, opacity=line_opacity)
        node_dots_m = pv.PolyData(node_dots)
        s.add_mesh(node_dots, render_points_as_spheres=True, color=node_colour, point_size=node_size, name=n_tag)

        # tri_surf, tris = self.get_triangle_surface(res=res)
        hex_surf, lines = self.get_hex_surface(list(range(len(self.elements))), tiling)
        surf_mesh = pv.PolyData(hex_surf, lines)
        
        if isinstance(mesh_colour, np.ndarray):
            surf_mesh[mesh_col_scalar_name] = mesh_colour
        # surf_mesh.faces = np.concatenate((3 * np.ones((tris.shape[0], 1)), tris), axis=1).astype(int)
        s.add_mesh(surf_mesh, style='wireframe', color=None if isinstance(mesh_colour, np.ndarray) else mesh_colour, opacity=mesh_opacity, name=h_tag, line_width=mesh_width, render_lines_as_tubes=True)
        if labels:
            s.add_point_labels(points = node_dots, labels=[str(i) for i in range(node_dots.shape[0])], name=v_tag)
        if elem_labels:
            elem_locs= np.ones((1, self.elements[0].ndim)) * 0.5
            pts = np.array(self.evaluate_embeddings_in_every_element(elem_locs))
            elem_labels = [f"elem: {i}" if self.elements[0].id is None else f"elem: ind {i}, id {self.elements[i].id}" for i in range(pts.shape[0])] 
            s.add_point_labels(points = pts, labels=elem_labels, name=e_tag)

        if scene is not None:
            return
        s.show()

    def transform(self, tform):
        """
        Apply a 4x4 3D homogenous transform to the mesh.
        """
        for node in self.nodes:
            node.loc = h_tform(node.loc, tform, fill=1)
            for k,v in node.items():  
                node[k] = h_tform(v, tform, fill=0)
        self.generate_mesh()

    ################################## INTERNAL
    def _generate_elem_functions(self):
        """
            Creates the internal function evaluation structure.
        """
        self.elem_evals = make_eval(self.elements[0].basis_functions, self.elements[0].BasisProductInds)
        self.elem_xi_deriv = jax.jacfwd(self.elem_evals, argnums=1)
        self.elem_param_deriv = jax.jacfwd(self.elem_evals, argnums=0)

    def _generate_elem_deriv_functions(self):
        """
            Creates the internal function evaluation structure.
        """
        self.elem_deriv_evals = make_deriv_eval(self.elements[0].basis_functions, self.elements[0].BasisProductInds)

    def _generate_eval_function(self):
        """
            Generates the internal functions that evaluate embeddings.
            Code is structured so that the result can express custom derivatives
        """
        @wide_eval 
        def evaluate_embeddings(element_ids, xis, fit_params = self.optimisable_param_array, ele_map= self.ele_map):
            element_ids = jnp.atleast_1d(element_ids)
            xis = jnp.atleast_2d(xis)

            param_data = jnp.asarray(self.true_param_array)
            if not len(fit_params) == len(param_data):
                fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            params = jnp.asarray(fit_params)[map]
            p_array = params[jnp.asarray(element_ids).astype(int)]
            outputs = jax.vmap(lambda x: self.elem_evals(x, jnp.asarray(xis)).reshape(-1,self.fdim))
            res = outputs(p_array)
            return res.reshape(-1,self.fdim)
        
        self.evaluate_embeddings = evaluate_embeddings

    def _generate_deriv_function(self):
        """
            Generates the internal functions that evaluate the derivatives of embeddings
            Code is structured so that the result can express custom derivatives
        """
        @wide_eval
        def evaluate_deriv_embeddings(element_ids, xis, derivs, fit_params = self.optimisable_param_array, ele_map= self.ele_map):
            element_ids = jnp.atleast_1d(element_ids)
            xis = jnp.atleast_2d(xis)
            param_data = jnp.asarray(self.true_param_array)

            if not len(fit_params) == len(param_data):
                fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            params = jnp.asarray(fit_params)[map]
            p_array = params[jnp.asarray(element_ids).astype(int)]

            outputs = jax.vmap(lambda x: self.elem_deriv_evals(x, jnp.asarray(xis), derivs).reshape(-1,self.fdim))
            res = outputs(p_array)
            return res.reshape(-1,self.fdim)
        
        self.evaluate_deriv_embeddings = evaluate_deriv_embeddings

    def _generate_weight_function(self):
        """
        Creastes the weight matrix of the mesh. Useful for direct linear fitting with constant xi embeddings.
        """
        self.generate_weight_matrix = make_weight_eval(self.elements[0].basis_functions, self.elements[0].BasisProductInds)
    ################################# useful utils.

    def _solve_RHS(self, el, xi, r, stepsize, lbound, fit_params=None):
        J = self.evaluate_jacobians(el, xi, fit_params=fit_params)[0]  # (fdim, ndim)
        J_free = J * jnp.where(lbound, 0.0, 1.0)
        return jnp.where(lbound, jnp.zeros(xi.shape[0]), _pseudoinverse_matvec(J_free, r)) * stepsize

    # @partial(jax.jit, static_argnames=("self", "iterations"))
    def _xis_to_points(self, elem, xi0, x_target, init_err, lbound, iterations, fit_params=None):
        """
        RK4-like fixed-iteration update in xi-space, JAX-optimized for jit/vmap.
        Uses a basic descent check to garuntee convergence.
        Returns: ((elem, xi), residual)
        """

        def body(_, state):
            elem_prev, xi_prev, elem, xi, r_mag, stepsize = state
            # jax.debug.print("elem {elem}, xi {xi}, stepsize {stp}", elem=elem, xi=xi, stp=stepsize)
            current_x = self.evaluate_embeddings(elem, xi, fit_params=fit_params)[0]
            r = x_target - current_x
            r_dist = jnp.linalg.norm(r)
            # check if it was actually lowered, if not, just decrease stepsize
            lowered = r_dist < r_mag
            
            # if not lowered, switch on the bound
            
            elem = jnp.where(lowered, elem, elem_prev)
            xi = jnp.where(lowered, xi, xi_prev)
            stepsize = jnp.where(lowered, stepsize, stepsize/2)
            r_mag = jnp.where(lowered, r_dist, r_mag)

            # RK4 stages
            k1 = self._solve_RHS(elem, xi, r, stepsize, lbound, fit_params=fit_params)
            k2 = self._solve_RHS(elem, xi + 0.5 * k1, r, stepsize, lbound, fit_params=fit_params)
            k3 = self._solve_RHS(elem, xi + 0.5 * k2, r, stepsize, lbound, fit_params=fit_params)
            k4 = self._solve_RHS(elem, xi + k3, r, stepsize, lbound, fit_params=fit_params)

            xi_new = xi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            elem_new, xi_mapped, _ = self.topomap(elem, xi_new)


            return (elem, xi, elem_new, xi_mapped, r_mag, stepsize)

        _, _, elem_f, xi_f, _, _ = jax.lax.fori_loop(
            0, iterations, body, (elem.astype(int), xi0, elem.astype(int), xi0, init_err, 1)
        )

        residual = self.evaluate_embeddings(elem_f, xi_f, fit_params=fit_params) - x_target
        return (elem_f, xi_f), residual
    
    def embed_points(self, points, verbose=0, init_elexi=None, fit_params=None, return_residual=False, surface_embed=False, iterations=3):
        """Find the parametric coordinates (element, xi) for a set of physical-space points.

        Uses an approximate nearest-neighbour search on a coarse xi grid to
        obtain initial estimates, then refines with a JAX-accelerated RK4
        fixed-iteration descent (see :meth:`_xis_to_points`).  Topology
        mapping (:meth:`topomap`) is applied at each iteration so that points
        near element boundaries are correctly assigned to neighbouring
        elements.

        To cleanly handle the complex derivatives this generats,a mesh_embed_points helper is created to capture the mesh in a closure.

        Parameters
        ----------
        points:
            Physical-space query points, shape ``(n_pts, fdim)``.
        verbose:
            Verbosity level.  ``0`` → silent; ``2`` → print mean/max
            residual; ``3`` → also render an error visualisation with
            PyVista.
        init_elexi:
            Pre-computed initial ``(elem_num, xis)`` tuple.  When supplied,
            the coarse nearest-neighbour search is skipped.
        fit_params:
            Optional parameter override for the mesh geometry.
        return_residual:
            When ``True``, returns a ``((elem_num, embedded), residual)``
            tuple instead of just ``(elem_num, embedded)``.
        surface_embed:
            Restrict the coarse search to the surface faces of a 3-D mesh.
        iterations:
            Number of RK4 refinement iterations.

        Returns
        -------
        elem_num : jnp.ndarray
            Element index for each query point, shape ``(n_pts,)``.
        embedded : jnp.ndarray
            Parametric coordinates, shape ``(n_pts, ndim)``.
        residual : jnp.ndarray
            (Only when *return_residual* is ``True``) Embedding error
            vectors, shape ``(n_pts, fdim)``.
        """
        def mesh_embed_points_test(points, fit_params, init_elexi=None, surface_embed=False, iterations=3):
            """Find the parametric coordinates (element, xi) for a set of physical-space points.

            Uses an approximate nearest-neighbour search on a coarse xi grid to
            obtain initial estimates, then refines with a JAX-accelerated RK4
            fixed-iteration descent (see :meth:`_xis_to_points`).  Topology
            mapping (:meth:`topomap`) is applied at each iteration so that points
            near element boundaries are correctly assigned to neighbouring
            elements.

            Parameters
            ----------
            points:
                Physical-space query points, shape ``(n_pts, fdim)``.
            verbose:
                Verbosity level.  ``0`` → silent; ``2`` → print mean/max
                residual; ``3`` → also render an error visualisation with
                PyVista.
            init_elexi:
                Pre-computed initial ``(elem_num, xis)`` tuple.  When supplied,
                the coarse nearest-neighbour search is skipped.
            fit_params:
                Optional parameter override for the mesh geometry.
            return_residual:
                When ``True``, returns a ``((elem_num, embedded), residual)``
                tuple instead of just ``(elem_num, embedded)``.
            surface_embed:
                Restrict the coarse search to the surface faces of a 3-D mesh.
            iterations:
                Number of RK4 refinement iterations.

            Returns
            -------
            elem_num : jnp.ndarray
                Element index for each query point, shape ``(n_pts,)``.
            embedded : jnp.ndarray
                Parametric coordinates, shape ``(n_pts, ndim)``.
            residual : jnp.ndarray
                (Only when *return_residual* is ``True``) Embedding error
                vectors, shape ``(n_pts, fdim)``.
            """

            points = jnp.atleast_2d(points) #ensure correct shape and type
            
            if init_elexi is None: #do a coarse embedding
                if self.elements[0].ndim == 2:
                    res = 40
                    xis = jnp.asarray(self.xi_grid(res, 2, boundary_points=False))
                    ndim = 2
                    coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                    test_res, i_data = jax_aknn(points, coarse_pts, k=1)
                    i = i_data[:, 0]
                    elem_num = i // xis.shape[0]
                    init_xi = xis[i % xis.shape[0]]

                    #TODO 2D manifold embedding, it should maube be exactly the same
                    at_lo = init_xi < 1e-6
                    at_hi = init_xi > 1 - 1e-6
                    mf_pt = at_lo | at_hi
                else:
                    res = 40
                    ndim = 3
                    if not surface_embed:
                        # Build interior grid
                        xis = self.xi_grid(res, 3, boundary_points=True)
                        n_pts = xis.shape[0]          # grid points per element
                        coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                        test_res, i_data = jax_aknn(points, coarse_pts, k=1)

                        i = i_data[:, 0]
                        elem_num = i // xis.shape[0]
                        init_xi = xis[i % xis.shape[0]]
                        
                        init_ests = self.evaluate_embeddings_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                        J_init = self.eval_numeric_jac_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                        proj_dir = jnp.sum((points - init_ests)[:, :, None] * J_init, axis=1) > 0

                        # a further check is necessary here: does the mf edge actually sit on a boundary?

                        at_lo = init_xi < 1e-6
                        at_hi = init_xi > 1 - 1e-6
                        
                        mf_lo = at_lo & ~proj_dir #is the point on the manifold?
                        mf_hi = at_hi & proj_dir
                        init_xi += (~mf_lo & at_lo) * 2e-2 - (~mf_hi & at_hi)*2e-2

                        mf_pt = mf_lo | mf_hi

                    else:
                        # surface_embed=True: embed on self surface faces only (unchanged)
                        faces = self.faces
                        face_pts = []
                        elem_pts = []
                        xi_pts = []
                        xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)
                        for face in faces:
                            grid_def = xi3grid[face[1], face[2]]
                            elem_pts.append(np.ones(grid_def.shape[0]) * face[0])
                            xi_pts.append(grid_def)
                            face_pts.append(self.evaluate_embeddings(jnp.array([face[0]]),grid_def))
                        coarse_pts = jnp.concatenate(face_pts, axis=0)
                        elems = jnp.concatenate(elem_pts, axis=0)
                        xis = jnp.concatenate(xi_pts, axis=0)
                        test_res, i_data = jax_aknn(points, coarse_pts, k=1)
                        i = i_data[:, 0]
                        elem_num = elems[i]
                        init_xi  = xis[i]

                        at_lo = init_xi < 1e-6
                        at_hi = init_xi > 1 - 1e-6
                        mf_pt = at_lo | at_hi
            else:
                elem_num, init_xi = init_elexi
                elem_num = jnp.atleast_1d(elem_num)
                init_xi = jnp.atleast_2d(init_xi)
                ndim = self.elements[0].ndim

                test_res = points - self.evaluate_embeddings_ele_xi_pair(elem_num, init_xi)

                at_lo = init_xi < 1e-6
                at_hi = init_xi > 1 - 1e-6
                mf_pt = at_lo | at_hi

            (elem_num, embedded), res = jax.vmap(
                lambda elem, xi, target, rmag, lbound : self._xis_to_points(elem, xi, target, lbound, rmag, iterations=iterations, fit_params=fit_params)
            )(elem_num, init_xi, points, mf_pt, jnp.linalg.norm(test_res, axis=-1))

            # elem_num, embedded, res = elem_num, init_xi, test_res


            return (elem_num, embedded), res

        @jax.custom_jvp
        def mesh_embed_points(points, fit_params, init_elexi=None, surface_embed=False, iterations=3):
            """Find the parametric coordinates (element, xi) for a set of physical-space points.

            Uses an approximate nearest-neighbour search on a coarse xi grid to
            obtain initial estimates, then refines with a JAX-accelerated RK4
            fixed-iteration descent (see :meth:`_xis_to_points`).  Topology
            mapping (:meth:`topomap`) is applied at each iteration so that points
            near element boundaries are correctly assigned to neighbouring
            elements.

            Parameters
            ----------
            points:
                Physical-space query points, shape ``(n_pts, fdim)``.
            verbose:
                Verbosity level.  ``0`` → silent; ``2`` → print mean/max
                residual; ``3`` → also render an error visualisation with
                PyVista.
            init_elexi:
                Pre-computed initial ``(elem_num, xis)`` tuple.  When supplied,
                the coarse nearest-neighbour search is skipped.
            fit_params:
                Optional parameter override for the mesh geometry.
            return_residual:
                When ``True``, returns a ``((elem_num, embedded), residual)``
                tuple instead of just ``(elem_num, embedded)``.
            surface_embed:
                Restrict the coarse search to the surface faces of a 3-D mesh.
            iterations:
                Number of RK4 refinement iterations.

            Returns
            -------
            elem_num : jnp.ndarray
                Element index for each query point, shape ``(n_pts,)``.
            embedded : jnp.ndarray
                Parametric coordinates, shape ``(n_pts, ndim)``.
            residual : jnp.ndarray
                (Only when *return_residual* is ``True``) Embedding error
                vectors, shape ``(n_pts, fdim)``.
            """

            points = jnp.atleast_2d(points) #ensure correct shape and type
            
            if init_elexi is None: #do a coarse embedding
                if self.elements[0].ndim == 2:
                    res = 40
                    xis = jnp.asarray(self.xi_grid(res, 2, boundary_points=False))
                    ndim = 2
                    coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                    test_res, i_data = jax_aknn(points, coarse_pts, k=1)
                    i = i_data[:, 0]
                    elem_num = i // xis.shape[0]
                    init_xi = xis[i % xis.shape[0]]

                    #TODO 2D manifold embedding, it should maube be exactly the same
                    at_lo = init_xi < 1e-6
                    at_hi = init_xi > 1 - 1e-6
                    mf_pt = at_lo | at_hi
                else:
                    res = 40
                    ndim = 3
                    if not surface_embed:
                        # Build interior grid
                        xis = self.xi_grid(res, 3, boundary_points=True)
                        n_pts = xis.shape[0]          # grid points per element
                        coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                        test_res, i_data = jax_aknn(points, coarse_pts, k=1)

                        i = i_data[:, 0]
                        elem_num = i // xis.shape[0]
                        init_xi = xis[i % xis.shape[0]]
                        
                        init_ests = self.evaluate_embeddings_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                        J_init = self.eval_numeric_jac_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                        proj_dir = jnp.sum((points - init_ests)[:, :, None] * J_init, axis=1) > 0

                        # a further check is necessary here: does the mf edge actually sit on a boundary?

                        at_lo = init_xi < 1e-6
                        at_hi = init_xi > 1 - 1e-6
                        
                        mf_lo = at_lo & ~proj_dir #is the point on the manifold?
                        mf_hi = at_hi & proj_dir
                        init_xi += (~mf_lo & at_lo) * 2e-2 - (~mf_hi & at_hi)*2e-2

                        mf_pt = mf_lo | mf_hi

                    else:
                        # surface_embed=True: embed on self surface faces only (unchanged)
                        faces = self.faces
                        face_pts = []
                        elem_pts = []
                        xi_pts = []
                        xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)
                        for face in faces:
                            grid_def = xi3grid[face[1], face[2]]
                            elem_pts.append(np.ones(grid_def.shape[0]) * face[0])
                            xi_pts.append(grid_def)
                            face_pts.append(self.evaluate_embeddings(jnp.array([face[0]]),grid_def))
                        coarse_pts = jnp.concatenate(face_pts, axis=0)
                        elems = jnp.concatenate(elem_pts, axis=0)
                        xis = jnp.concatenate(xi_pts, axis=0)
                        test_res, i_data = jax_aknn(points, coarse_pts, k=1)
                        i = i_data[:, 0]
                        elem_num = elems[i]
                        init_xi  = xis[i]

                        at_lo = init_xi < 1e-6
                        at_hi = init_xi > 1 - 1e-6
                        mf_pt = at_lo | at_hi
            else:
                elem_num, init_xi = init_elexi
                elem_num = jnp.atleast_1d(elem_num)
                init_xi = jnp.atleast_2d(init_xi)
                ndim = self.elements[0].ndim

                test_res = points - self.evaluate_embeddings_ele_xi_pair(elem_num, init_xi)

                at_lo = init_xi < 1e-6
                at_hi = init_xi > 1 - 1e-6
                mf_pt = at_lo | at_hi

            (elem_num, embedded), res = jax.vmap(
                lambda elem, xi, target, rmag, lbound : self._xis_to_points(elem, xi, target, lbound, rmag, iterations=iterations, fit_params=fit_params)
            )(elem_num, init_xi, points, mf_pt, jnp.linalg.norm(test_res, axis=-1))

            # elem_num, embedded, res = elem_num, init_xi, test_res


            return (elem_num, embedded), res

        


        @mesh_embed_points.defjvp
        def embed_pts_ptderiv(primal, tangent):
            """
            Follows jax protocol to define the jax compatable derivatives.
            Gives the local derivatives of the point embedding with respect to the input points to embed.

            It calculates the linear derivatvies of elem_num, xi, and the residual of the mesh embed function.
            elem_num derivative is always zero.
            
            xi is the main point of interest.
            res is useful for fitting.

            """
            #testing value
            # correct_primal, correct_tangent = jax.jvp(mesh_embed_points_test, primals=primal, tangents=tangent)

            #primal computation.
            (ele, xi), res = primal_out = mesh_embed_points(*primal)
            res = res.squeeze()
            #local tangent spaces.
            point_dot, param_dot, _, _, _ = tangent
            params = primal[1] #the only really important thing from the primals

            res_mag = jnp.linalg.norm(res, axis=-1)
            approx_normals = res / jnp.maximum(res_mag, 1e-6)[:, None] #if the residual is tiny, then fully embedded.

            #given a delta_params, how will the output change.
            deriv_fn = jax.jacfwd(lambda e, x, f: self.evaluate_embeddings_ele_xi_pair(e, x, fit_params=f), argnums=2)
            param_shift = lambda e,x: deriv_fn(e,x, params) @ param_dot
            param_sft = jax.vmap(param_shift)(ele, xi)

            res_d_point = jnp.sum(approx_normals * point_dot) * approx_normals 
            res_d_param = jnp.sum(approx_normals * param_sft) * approx_normals #must have the same shape as the output! mx3
            
            # account for behaviour inside and outside the mesh by projecting the motion onto the residual if the residual is non-negligable.
            xi_rel_motion_point = jnp.where(res_mag[:, None] < 1e-4, point_dot, res_d_point)
            xi_rel_motion_param = jnp.where(res_mag[:, None] < 1e-4, param_sft, res_d_param)

            jacs = self.evaluate_jacobians_ele_xi_pair(ele, xi, fit_params=params)
            
            mapped_lsq = jax.vmap(lambda A,y: jnp.linalg.lstsq(A,y)[0])
            xi_d_point = mapped_lsq(jacs, xi_rel_motion_point)
            xi_d_param = mapped_lsq(jacs, xi_rel_motion_param)

            tangent_out = ((jnp.zeros_like(ele, dtype=jax.float0), xi_d_point + xi_d_param), (res_d_point + res_d_param)[:, None])

            return primal_out, tangent_out

        if fit_params is None:
            fit_params = self.optimisable_param_array

        points = jnp.atleast_2d(points) #ensure correct shape and type
        (elem_num, embedded), residual =  mesh_embed_points(points, fit_params, init_elexi, surface_embed, iterations)

        if verbose >= 2:
            final_mean_dist = np.mean(np.linalg.norm(np.asarray(residual), axis=-1))
            final_max_dist  = np.max(np.linalg.norm(np.asarray(residual), axis=-1))
            print(f"final mean error of {final_mean_dist} units, max error of {final_max_dist}")

        if verbose == 3:
            locs = self.evaluate_embeddings_ele_xi_pair(elem_num, embedded)
            vec_errors = points - locs
            errors = np.linalg.norm(vec_errors, axis=-1)

            line_segs = np.concatenate(
                (np.atleast_2d(locs)[:, None], np.atleast_2d(points)[:, None]), axis=1
            ).reshape(-1, self.fdim)
            s = pv.Plotter()
            self.plot(s)
            data = pv.PolyData(np.asarray(locs))
            data['err'] = errors
            s.add_mesh(pv.line_segments_from_points(line_segs), color='k')
            s.add_mesh(data, render_points_as_spheres=True, point_size=20)
            s.show()
        
        if return_residual:
            return (elem_num, embedded), residual

        return elem_num, embedded





    def evaluate_sobolev(self, weights=None, fit_params=None):
        """
        Works out and defines the Sobolev values associated with the derivatives of the input elements.
        Then calculates the appropriate gauss points, and returns the elements assessed with the appropriate weighting. 
        """

        n_derivs = [len(b.deriv) for b in self.elements[0].basis_functions]
        d_order = [b.order for b in self.elements[0].basis_functions]
        if fit_params is None:
            fit_params = self.true_param_array

        gp, w = self.gauss_grid(d_order)
        deriv_combos = list(product(*[range(d) for d in n_derivs]))[1:] # skip the no deriv case
        n_eles = len(self.elements)

        if weights is None:
            weights = np.ones(len(deriv_combos))
        else:
            if not len(weights) == len(deriv_combos):
                raise ValueError("The length of the provided weights did not match the number of sobolev terms associated with this element")

        out_data = []
        for d, sw in zip(deriv_combos, weights):
            data = self.evaluate_deriv_embeddings_in_every_element(gp, d, fit_params=fit_params)
            weighted = (data.reshape(n_eles, -1, 3) * w[None, :, None]).ravel() * sw
            out_data.append(weighted)

        return jnp.concatenate(out_data)

    def get_volume(self, fit_params = None):
        """
        Calculates the mesh volume using a gauss point integration scheme.

        :param fit_params: an overide of the standard mesh parameters to use for fitting.
        :returns vol: The estimated volume of the mesh.
        """
        gauss_points, weights = self.gauss_grid([e.order for e in self.elements[0].basis_functions])
        Jmats = self.evaluate_jacobians_in_every_element(gauss_points, fit_params=fit_params)
        dets = jnp.linalg.det(Jmats).reshape(len(self.elements), -1)
        vols = dets * weights[None]
        return jnp.sum(vols)
    
    @wide_eval
    def evaluate_strain(self, element_ids, xis, othr: "Mesh", coord_function: Optional[Callable] = None, return_F=False, fit_params=None):
        """Evaluate the Green-Lagrange strain tensor between two mesh states.

        Computes the deformation gradient **F** = J_ref⁻¹ · J_def where J_ref
        is the Jacobian of *self* (reference configuration) and J_def is the
        Jacobian of *othr* (deformed configuration), then returns the strain
        tensor **E** = (Fᵀ F − I) / 2.

        Parameters
        ----------
        element_ids:
            1-D integer array, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.
        othr:
            A :class:`MeshField` representing the *deformed* configuration of
            the same topology.
        coord_function:
            Optional callable ``(mesh, eles, xis, Jmats) → Jmats`` that
            re-maps the Jacobian into a local coordinate frame (required for
            2-D manifold meshes).
        return_F:
            When ``True``, return the deformation gradient **F** instead of
            the strain tensor **E**.
        fit_params:
            Optional parameter override for *self*.

        Returns
        -------
        jnp.ndarray
            Green-Lagrange strain tensor ``E``, shape
            ``(n_pts, ndim, ndim)``, or the deformation gradient **F** if
            *return_F* is ``True``.

        Raises
        ------
        ValueError
            If called on a 2-D manifold mesh without supplying *coord_function*.
        """

        if self.ndim == 2 and coord_function is None:
            raise ValueError("Strain tensor on manifold mesh requires a coord function to provide a meaninful basis")

        deriv_self = self.evaluate_jacobians(element_ids, xis, fit_params=fit_params)
        deriv_othr = othr.evaluate_jacobians(element_ids, xis, fit_params=None)

        if coord_function is not None:
            deriv_self = coord_function(self, element_ids, xis, deriv_self)
            deriv_othr = coord_function(othr, element_ids, xis, deriv_othr)
        
        # str_tensor = deriv_othr @ jnp.linalg.inv(deriv_self)
        # F = str_tensor.reshape(-1, self.ndim, self.ndim)

        F = jnp.linalg.solve(
            deriv_self.transpose(0,2,1),
            deriv_othr.transpose(0,2,1),
        ).transpose(0,2,1)
        if return_F:
            return F
  
        strain = (F.transpose(0,2,1) @ F - np.eye(self.ndim)[None])/2
        return strain.reshape(-1, self.ndim, self.ndim)


    def plot_strains(self, eles, xis, strains, scene:Optional[pv.Plotter]=None, cmap='coolwarm'):
        """
        Given ele, xi locations, and the strain tensors evaluated at those locations, evaluates local strain ellipsoids, and plots them.
        """
        def get_batch_stretch_tensors(strains):
            m = strains.shape[0]
            I_batch = jnp.tile(jnp.eye(3), (m, 1, 1))
            C = 2 * strains + I_batch
            evals, evecs = jnp.linalg.eigh(C)
            safe_evals = jnp.maximum(evals, 0.0)
            sqrt_lambdas = jnp.sqrt(safe_evals)
            def reconstruct_single_U(v, s_lambdas):
                return v @ jnp.diag(s_lambdas) @ v.T
            U = jax.vmap(reconstruct_single_U)(evecs, sqrt_lambdas)
            return U
        locs = self.evaluate_embeddings_ele_xi_pair(eles, xis)
        sphere_base = pv.Sphere(1, theta_resolution=15, phi_resolution=15)

        test_pts = sphere_base.points[None, ..., None]
        U = get_batch_stretch_tensors(strains)
        
        def_pts = U[:, None] @ test_pts
        r_mag = np.linalg.norm(def_pts[..., 0], axis=-1) - 1

        pts = jax_aknn(locs, locs, k=2)[0]
        scale_to_use = np.median(pts[:, 1])/4

        sphere_pts = def_pts[..., 0] * scale_to_use + locs[:, None]

        sphere_arr = spheres_to_polydata(np.asarray(sphere_pts), sphere_base.faces)
        sphere_arr['relative length change'] = r_mag.flatten()

        max_c = np.nanmax(np.abs(r_mag))
        
        draw_flag = False
        if scene is None:
            scene = pv.Plotter()
            self.plot(scene)
            draw_flag= True
        scene.add_mesh(sphere_arr, smooth_shading=True, cmap=cmap, clim=[-max_c, max_c])
        if draw_flag:
            scene.show()


    ################################# FASTFITTING

    def get_xi_weight_mat(self, eles, xis):
        """Build the linear weight matrix for least-squares fitting.

        For each query point ``(eles[i], xis[i])``, evaluates the basis
        function values and places them in the appropriate column positions of
        a global weight matrix **W**, where ``W[i, j]`` is the contribution
        of the *j*-th nodal degree of freedom to the *i*-th query point.

        This matrix is used by :meth:`linear_fit`::

            W * node_params = target_values   (solved in a least-squares sense)

        Parameters
        ----------
        eles:
            1-D integer array of element indices, shape ``(n_pts,)``.
        xis:
            Parametric coordinates, shape ``(n_pts, ndim)``.

        Returns
        -------
        numpy.ndarray
            Weight matrix, shape ``(n_pts, n_nodes)``.
        """
        out_weight = np.zeros((len(eles), len(self.true_param_array)//self.fdim)) #
        unique_elem, inv = jnp.unique_inverse(eles)
        for ide, e in enumerate(unique_elem):
            mask = ide == inv
            weight_mat = self.generate_weight_matrix(xis[mask]).T #weights associated with each of the parameters for the input matrix.
            relevant_weight_locs = (self.ele_map[e, ::self.fdim]//self.fdim).astype(int)
            out_weight[np.ix_(mask, relevant_weight_locs)] = weight_mat
        return out_weight

    def linear_fit(self, targets, weight_mat, target_empty=-1):
        """Fit nodal parameters by solving a linear least-squares problem.

        Solves ``weight_mat @ params ≈ targets`` via :func:`numpy.linalg.lstsq`
        and updates the mesh's nodal parameters with the solution.  This is
        the fastest fitting approach when the xi embeddings are fixed (i.e. the
        mesh topology does not change during fitting).

        Parameters
        ----------
        targets:
            Target field values, shape ``(n_pts,)`` or ``(n_pts, fdim)``.
            Rows equal to *target_empty* (default ``-1``) are excluded from
            the fit.
        weight_mat:
            Weight matrix from :meth:`get_xi_weight_mat`,
            shape ``(n_pts, n_nodes)``.
        target_empty:
            Sentinel value used to mask out unused target rows.

        Notes
        -----
        Fixed parameters (set via :meth:`MeshNode.fix_parameter`) are
        currently **not** respected by this method.  Use the nonlinear
        optimisation pathway (``fitting.point_cloud_fit``) if constraints
        are required.
        """
        if targets.ndim > 1:
            target_mask = np.any(targets != target_empty, axis=-1)
        else:
            target_mask = targets != target_empty
        A = weight_mat[target_mask]
        b = targets[target_mask]

        assert A.shape[0] > A.shape[1], "Attempted to solve an undertederimined system, more datapoints are needed"
        new_params, residual, rank, s = np.linalg.lstsq(np.asarray(A).astype(np.double), np.asarray(b).astype(np.double))

        if rank < A.shape[1]:
            logging.warning("Problem matrix was rank deficient. Try fitting (i) more datapoints, or (ii) a lower order field")

        print('residual error:', residual)
        self.true_param_array = new_params.flatten()
        self.optimisable_param_array = self.true_param_array[self.optimisable_param_bool]
        self.update_from_params(new_params.flatten(), generate=False)
        self.generate_mesh()


    ################################# REFINEMENT
    def refine(self, refinement_factor: Optional[int]=None, by_xi_refinement: Optional[tuple[np.ndarray]] =  None,
               clean_nodes = True):
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

        Raises
        ------
        AssertionError
            If both *refinement_factor* and *by_xi_refinement* are given, or
            if *refinement_factor* < 2.
        """
        assert not(refinement_factor is not None and by_xi_refinement is not None), "Refinement factor and refining by defined xi are mutually exclusive."

        new_elements = []
        spatial_hash = {tuple(np.round(node.loc, 6).tolist()):idn for idn, node in enumerate(self.nodes)}
        
        for ide, e in enumerate(self.elements): #MAKE THE POINTS TO EVAL AT
            intermediate_node_number = np.array(e.n_in_dim) - 2
            if refinement_factor is not None:
                if refinement_factor < 2:
                    raise ValueError("Refining by less than 2 will not change the mesh")
                lr = (intermediate_node_number + 1) * refinement_factor + 1 
                if e.ndim==2:
                    eval_pts = np.column_stack([x.flatten() for x in np.array(np.mgrid[:lr[0], :lr[1]])/(lr[:, None, None]-1)])
                elif e.ndim==3:
                    eval_pts = np.column_stack([x.flatten() for x in np.array(np.mgrid[:lr[0], :lr[1], :lr[2]])/(lr[:, None, None, None]-1)])
                ref_array = np.ones(e.ndim) * refinement_factor
            if by_xi_refinement is not None:
                for b in by_xi_refinement:
                    assert b[0]==0 and b[-1] == 1, "Provided b arrays must start with 0 and and with 1"
                new_xi_refinement = []
                lr = []
                for idb, b in enumerate(by_xi_refinement):
                    n_points = (len(b)-1) * (intermediate_node_number[idb] + 1) + 1 
                    lr.append(n_points)
                    new_xi_refinement.append(np.interp(np.linspace(0,1, n_points), np.linspace(0,1, len(b)), b))
                eval_pts = np.column_stack([x.flatten() for x in np.meshgrid(*by_xi_refinement, indexing='ij')])
                if self.ndim == 2:
                    ref_array = np.array([len(by_xi_refinement[i]) for i in [0,1]]) - 1
                else:
                    ref_array = np.array([len(by_xi_refinement[i]) for i in [0,1, 2]]) - 1
        
            pts = self.evaluate_embeddings(np.array([ide]), eval_pts)
            additional_pts = []
            deriv_bound = np.where([np.any([st[:2] == 'dx' for st in b.weights]) for b in e.basis_functions] )[0]
            for d_val in EVAL_PATTERN[len(e.used_node_fields)]:
                #calculate the additional derivatives in the directions that need them
                derivs = [0,0,0]
                for dl, di in zip(deriv_bound, d_val): 
                    derivs[dl] = di
                d_scale = np.mean(ref_array[np.where(np.array(d_val))])
                additional_pts.append(self.evaluate_deriv_embeddings(np.array([ide]), eval_pts, derivs=derivs)/d_scale)

            #check the generated points against the element hashmap.
            pt_index_array = [] 
            for idpt, pt in enumerate(pts):
                ind = spatial_hash.get(hashp:=tuple(np.round(np.asarray(pt), 6)), None) 
                new_vals = {k:np.array(v) for k, v in zip(e.used_node_fields, [a[idpt] for a in additional_pts])}
                if ind is None:
                    node = MeshNode(pt, **new_vals)
                    self.add_node(node)
                    pt_index_array.append(len(self.nodes)-1)
                    spatial_hash[hashp] = len(self.nodes)-1
                else:
                    pt_index_array.append(ind)
                    self.nodes[ind].update(new_vals)

            if refinement_factor is not None:
                if e.ndim==2:
                    n_new = [refinement_factor, refinement_factor]
                elif e.ndim==3:
                    n_new = [refinement_factor] * 3
            elif by_xi_refinement is not None:
                n_new = [len(b) - 1 for b in by_xi_refinement]
            
            #lr now defines the total number of points per dimension of this elementlet
            pt_inds = np.array(pt_index_array).reshape(lr)


            for i in range(n_new[0]):
                i_loc = i * (e.n_in_dim[0] -1) 
                for j in range(n_new[1]):
                    j_loc = j * (e.n_in_dim[1] -1)
                    if e.ndim == 2:
                        points = pt_inds[i_loc:i_loc + e.n_in_dim[0], j_loc:j_loc + e.n_in_dim[1]]

                        elem_id = None
                        if e.id is not None:
                            elem_id = str(e.id) + f"_subelem_{i}_{j}"

                        new_e = MeshElement(node_indexes=points.T.flatten().tolist(), basis_functions=e.basis_functions, id=elem_id)
                        new_elements.append(new_e)
                        continue
                    for k in range(n_new[2]): 
                        k_loc = k * (e.n_in_dim[2] - 1)
                        points = pt_inds[
                            i_loc:i_loc + e.n_in_dim[0],
                            j_loc:j_loc + e.n_in_dim[1],
                            k_loc:k_loc + e.n_in_dim[2],
                        ]
                        # points = points.transpose((0,2,1))
                        points = points.flatten(order="F").astype(int)

                        elem_id = None
                        if e.id is not None:
                            elem_id = str(e.id) + f"_subelem_{i}_{j}_{k}"

                        new_e = MeshElement(node_indexes=points.tolist(), basis_functions=e.basis_functions, BP_inds=e.BasisProductInds, id=elem_id)
                        new_elements.append(new_e)
          

        self.elements = new_elements
        if clean_nodes:
            self._clean_pts()
        self.generate_mesh()
            
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


    def save(self, loc: PathLike):
        """
        Saves the mesh to a .json formated file in the given location
        """
        from HOMER.io import save_mesh #avoid the circular import here
        save_mesh(self, loc)

    def dump_to_dict(self):
        """
        Returns a dict structure representing the mesh object, for ease of saving
        """

        from HOMER.io import dump_mesh_to_dict
        return dump_mesh_to_dict(self)

    def __deepcopy__(self, memo):
        """
        Dumps the mesh to a dictionairy then rebuilds it to ensure that there is no shared memory between a mesh and it's deepcopy.
        """

        from HOMER.io import dump_mesh_to_dict, parse_mesh_from_dict
        dict_rep = deepcopy(dump_mesh_to_dict(self))
        return parse_mesh_from_dict(dict_rep) 


    def rebase(self, new_basis: BasisGroup, in_place=False, res=10) -> 'MeshField':
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

        Parameters
        ----------
        new_basis:
            Sequence of new 1-D basis classes, one per parametric direction.
        in_place:
            Currently unused (future: modify *self* rather than returning a
            new object).
        res:
            Number of xi grid points per direction used for the linear fit.

        Returns
        -------
        MeshField
            New mesh with the requested basis functions.
        """
        new_mesh = deepcopy(self)

        s_hash = {}
        def all_pairings(*lists):
            return [t[::-1] for t in itertools.product(*reversed(copy(lists)))]

        list_locs = [b.node_locs for b in new_basis]
        eval_pts = np.array(all_pairings(*list_locs))
        new_elements = []
        new_pts = []

        used_fields = MeshElement(node_ids=[np.arange(eval_pts.shape[0])], basis_functions=new_basis).used_node_fields

        for ide, elem in enumerate(self.elements):
            node_locs = self.evaluate_embeddings([ide], eval_pts)
            node_ids = []
            for node in node_locs:
                ind = s_hash.get(hashp:=tuple(np.round(np.asarray(node), 6)), None) 
                if ind is None:
                    ind = len(new_pts)
                    new_pts.append(MeshNode(loc=node, **{uf:np.zeros(self.fdim) for uf in used_fields}))
                    s_hash[hashp] = ind
                node_ids.append(ind)
            element = MeshElement(node_indexes=node_ids, basis_functions=new_basis)
            new_elements.append(element)

        new_mesh = MeshField(nodes=new_pts, elements=new_elements)
        egrid = self.xi_grid(res=res, boundary_points=False)
        el = (np.ones((1, res**self.ndim)) * np.arange(len(self.elements))[:, None]).flatten().astype(int)
        xi = np.tile(egrid.reshape(-1, self.ndim), (len(self.elements), 1))
        w_mat = new_mesh.get_xi_weight_mat(el, xi)
        locs = self.evaluate_embeddings_ele_xi_pair(el, xi)
        new_mesh.linear_fit(weight_mat=w_mat, targets=locs)
        new_mesh.generate_mesh()

        if in_place:
            self = new_mesh
        else:
            return new_mesh

class Mesh(MeshField):
    """A coordinate mesh that can also carry named secondary fields.

    :class:`Mesh` extends :class:`MeshField` by adding a dictionary
    ``fields`` that stores secondary :class:`MeshField` objects.  The
    primary geometry (XYZ world-space coordinates) is stored in the parent
    :class:`MeshField`, while any secondary quantities such as fibre
    directions, velocities, or material properties are stored as named
    entries in ``fields``.

    Secondary fields are created with :meth:`new_field` and accessed via
    dictionary-style indexing::

        mesh = Mesh(nodes=[...], elements=[...])
        mesh.new_field('fibre', field_dimension=3,
                       field_locs=data_pts, field_values=fibre_vectors,
                       new_basis=[H3Basis]*3)
        fibre_field = mesh['fibre']   # MeshField

    Parameters
    ----------
    nodes:
        Nodes defining the primary geometry.
    elements:
        Elements of the mesh.
    jax_compile:
        Pre-compile JAX functions at construction time.

    Attributes
    ----------
    fields : dict[str, MeshField]
        Named secondary fields.
    """

    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        """Initialise a :class:`Mesh`.

        Parameters
        ----------
        nodes:
            Node list (may be ``None`` for incremental construction).
        elements:
            Element or element list.
        jax_compile:
            If ``True``, JIT-compile internal functions at construction.
        """
        super().__init__(nodes, elements, jax_compile)
        self.fields = {}

    def __getitem__(self, input: str) -> MeshField:
        return self.fields[input]

    def __setitem__(self, key: str, value: MeshField):
        assert len(self.elements) == len(value.elements), 'Fields must have the same number of elements'
        assert self.elements[0].ndim == value.elements[0].ndim, 'Feilds must share the same dimensionality of basis components'
        self.fields[key] = value

    def refine(self, refinement_factor: Optional[int] = None, by_xi_refinement: Optional[tuple[np.ndarray]] = None, clean_nodes=True):
        """Refine the primary geometry *and* all secondary fields simultaneously.

        Calls :meth:`MeshField.refine` on the coordinate mesh and on every
        field in :attr:`fields`.

        Parameters
        ----------
        refinement_factor:
            Uniform refinement multiplier (≥ 2).
        by_xi_refinement:
            Per-direction xi breakpoint arrays.
        clean_nodes:
            Remove unreferenced nodes after refinement.
        """
        super().refine(refinement_factor, by_xi_refinement, clean_nodes)
        for field in self.fields.values():
            field.refine(refinement_factor, by_xi_refinement, clean_nodes)

    def plot(self, scene: Optional[pv.Plotter] = None, node_colour='r', node_size=10, labels=False, tiling=(10, 6), 
             mesh_colour: str | np.ndarray = 'gray', mesh_opacity=0.1, mesh_width=2, mesh_col_scalar_name="Field", 
             line_colour: str | np.ndarray = 'black', line_opacity=1, line_width=2, line_col_scalar_name="Field",
             elem_labels=False, render_name: Optional[str] = None, 
             field_to_draw = None, field_xi = None, draw_xyz_field = True, field_artist: Optional[Callable[[pv.Plotter, np.ndarray, np.ndarray], None]] = None,
             default_field_point_size=25, default_xi_res=4):
        """Draw the mesh and optionally overlay a secondary field.

        Parameters
        ----------
        scene:
            Existing :class:`pyvista.Plotter`.  When ``None``, a new plotter
            is created and shown.
        node_colour:
            Colour for node spheres.
        node_size:
            Node sphere size.
        labels:
            When ``True``, add node index labels (forces *node_size* = 0).
        tiling:
            ``(xn, yn)`` tiling for the hexagonal surface overlay.
        mesh_colour:
            Surface mesh colour.  Pass a :class:`numpy.ndarray` to colour-map
            by scalar values.
        mesh_opacity:
            Surface opacity (0–1).
        mesh_width:
            Line width for the hex wireframe.
        mesh_col_scalar_name:
            Scalar array name used when *mesh_colour* is an array.
        line_colour:
            Colour for the structural edge lines.
        line_opacity:
            Edge line opacity.
        line_width:
            Edge line width.
        line_col_scalar_name:
            Scalar name for colour-mapped edges.
        elem_labels:
            When ``True``, label element centres.
        render_name:
            Prefix for named actors (allows individual actor replacement in
            an interactive scene).
        field_to_draw:
            Name of a secondary field to visualise.  When ``None`` only the
            geometry is drawn.
        field_xi:
            Custom xi grid at which to evaluate the secondary field.
            Defaults to a uniform grid at *default_xi_res*.
        draw_xyz_field:
            When ``False``, suppress drawing of the primary geometry.
        field_artist:
            Custom callable ``(plotter, locs, values) → None`` for rendering
            the secondary field.  Defaults to line segments for 3-D fields
            and coloured spheres for 1-D scalar fields.
        default_field_point_size:
            Point size used by the default scalar field artist.
        default_xi_res:
            Xi grid resolution for the secondary field visualisation.
        """
        s_flag = False
        if scene is None:
            scene = pv.Plotter()
            s_flag = True
    
        #then you evaluate the field with the surface values throughout the mesh.
        if draw_xyz_field:
            super().plot(scene, node_colour, node_size, labels, tiling, 
                         mesh_colour, mesh_opacity, mesh_width, mesh_col_scalar_name,
                         line_colour, line_opacity, line_width, line_col_scalar_name,
                         elem_labels, render_name)

        if field_to_draw == None:
            if s_flag:
                scene.show()
            return
        
        if field_xi is None:
            field_xi = self.xi_grid(res=default_xi_res, boundary_points=False)

        f_locs = self.evaluate_embeddings_in_every_element(field_xi)

        f_values = self[field_to_draw].evaluate_embeddings_in_every_element(field_xi)

        if field_artist is None:
            def field_artist(lscene, locs, values):
                if self[field_to_draw].fdim == 3:
                    #rather than arrows, create a line object.
                    ldata = np.concatenate((locs[:, None], (locs + values)[:, None]), axis=1).reshape(-1, 3)
                    lines = pv.line_segments_from_points(ldata)
                    lines[field_to_draw] = np.linalg.norm(values, axis=-1)
                    lscene.add_mesh(lines, render_lines_as_tubes=True, line_width=5)
                elif self[field_to_draw].fdim == 1:
                    f = pv.PolyData(np.asarray(locs))
                    f[field_to_draw] = np.asarray(values)
                    lscene.add_mesh(f, render_points_as_spheres=True, point_size=default_field_point_size)
                else:
                    raise ValueError(f"Default field artist doesn't support {self[field_to_draw].fdim} dimension fields, create a custom artist")
                    
        field_artist(scene, f_locs, f_values)

        if s_flag:
            scene.show()
        return

    def new_field(self, field_name: str, field_dimension: int, new_basis: BasisGroup, field_locs: Optional[np.ndarray]=None, field_values: Optional[np.ndarray]=None, res=10) -> None:
        """Create a secondary field and optionally fit it to sample data.

        A secondary field is a :class:`MeshField` with its own basis
        functions and node topology that is *co-located* with the primary
        coordinate mesh.  It can represent any spatially varying quantity
        – fibre directions, velocity vectors, pressures, stresses, etc.

        The three-step construction algorithm is:

        1. Determine the new field node locations by evaluating the primary
           mesh at the node positions of *new_basis*.
        2. If *field_locs* and *field_values* are provided, embed the sample
           points into the mesh with :meth:`embed_points`.
        3. Build the linear weight matrix and solve for nodal parameters with
           :meth:`linear_fit`.

        After this call, the field is accessible as ``mesh[field_name]``.

        Parameters
        ----------
        field_name:
            Key used to store and retrieve the new field, e.g.
            ``'fibre_direction'``.
        field_dimension:
            Dimensionality of the field values:

            * ``1`` – scalar field (e.g. pressure, temperature, Z-coordinate)
            * ``3`` – 3-D vector field (e.g. fibre direction, velocity)
        new_basis:
            Sequence of 1-D basis classes for the new field, one per
            parametric direction.  May differ from the primary mesh basis.
            For example, use ``[H3Basis]*3`` for a smooth vector field or
            ``[L1Basis]*3`` for a piecewise-linear scalar field.
        field_locs:
            Physical-space sample locations where field values are known,
            shape ``(n_samples, fdim)``.  When ``None``, an empty field is
            created without fitting.
        field_values:
            Target field values at *field_locs*, shape
            ``(n_samples,)`` for scalars or ``(n_samples, field_dimension)``
            for vectors.  Required if *field_locs* is provided.
        res:
            Unused (reserved for future use).

        Examples
        --------
        Fit a unit-normal vector field and a scalar height field::

            mesh.new_field(
                'normals',
                field_dimension=3,
                field_locs=sample_pts,       # shape (N, 3)
                field_values=normal_vectors, # shape (N, 3)
                new_basis=[H3Basis, H3Basis, H3Basis],
            )
            mesh.new_field(
                'height',
                field_dimension=1,
                field_locs=sample_pts,       # shape (N, 3)
                field_values=sample_pts[:, 2],  # scalar Z values
                new_basis=[L1Basis, L1Basis, L1Basis],
            )

            # Retrieve and evaluate
            normal_field = mesh['normals']
            values_at_xis = normal_field.evaluate_embeddings(elem_ids, xis)
        """

        list_locs = [b.node_locs for b in new_basis]
        eval_pts = np.array(all_pairings(*list_locs))
        used_fields = MeshElement(node_ids=[np.arange(eval_pts.shape[0])], basis_functions=new_basis).used_node_fields

        s_hash = {}

        new_elements = []
        new_pts = []

        for ide, elem in enumerate(self.elements):
            node_locs = self.evaluate_embeddings([ide], eval_pts)
            node_ids = []
            for node in node_locs:
                ind = s_hash.get(hashp:=tuple(np.round(np.asarray(node), 6)), None) 
                if ind is None:
                    ind = len(new_pts)
                    new_pts.append(MeshNode(loc=[0] * field_dimension, **{uf:np.zeros(field_dimension) for uf in used_fields}))
                    s_hash[hashp] = ind
                node_ids.append(ind)
            element = MeshElement(node_indexes=node_ids, basis_functions=new_basis)
            new_elements.append(element)

        self[field_name] = MeshField(nodes=new_pts, elements=new_elements)

        if field_locs is None or field_values is None:
            return

        locs = self.embed_points(field_locs)
        w_mat = self[field_name].get_xi_weight_mat(*locs)
        self[field_name].linear_fit(weight_mat=w_mat, targets=field_values)
        return

    
    def save(self, loc: PathLike):
        """
        Saves the mesh to a .json formated file in the given location
        """
        from HOMER.io import save_mesh #avoid the circular import here
        save_mesh(self, loc)

    def dump_to_dict(self):
        """
        Returns a dict structure representing the mesh object, for ease of saving
        """
        all_dicts = {f:self[f].dump_to_dict() for f in self.fields.keys()}
        all_dicts['main'] = self.super().dump_to_dict()
        return all_dicts



def make_eval(basis_funcs: BasisGroup, bp_inds:list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a 
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval



def make_deriv_eval(basis_funcs, bp_inds):
    """
    Returns a JAX-compliant evaluator function.
    - basis_funcs length must be 2 or 3
    - bp_inds should be static for best compilation behavior
    """
    bp_inds = jnp.asarray(bp_inds, dtype=jnp.int32)
    ndim = len(basis_funcs)

    if ndim == 2:
        def xi_eval(elem_params, xis, d_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])  
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1]) 
            weights = N2_weights(w0, w1, bp_inds)            
            params2 = elem_params.reshape(weights.shape[0], -1)  
            out = jnp.einsum("bo,bp->po", params2, weights)     
            return out.reshape(-1)

    elif ndim == 3:
        def xi_eval(elem_params, xis, d_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            w2 = basis_funcs[2].deriv[d_inds[2]](xis[:, 2])
            weights = N3_weights(w0, w1, w2, bp_inds)
            params2 = elem_params.reshape(weights.shape[0], -1)
            out = jnp.einsum("bo,bp->po", params2, weights)
            return out.reshape(-1)

    else:
        raise ValueError("Currently, meshes must be 2D or 3D")

    return xi_eval

def make_weight_eval(basis_funcs: BasisGroup, bp_inds):
    if len(basis_funcs) == 2:
        def xi_eval(xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            return weights
    elif len(basis_funcs) == 3:
        def xi_eval(xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            return weights
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval

def _pseudoinverse_matvec(J: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    Jt_v = J.T @ v          # (d,) — project v onto tangent space
    JtJ = J.T @ J           # (d, d) Gram matrix
    dxi, _, _, _ = jnp.linalg.lstsq(JtJ, Jt_v, rcond=None)
    return dxi

GAUSS = { 
        1:[np.array([[0.5]]),
           np.array([1])],
        2:[np.array([[0.21132486540518708], [0.78867513459481287]]),
           np.array([0.5, 0.5])],
        3:[np.array([[0.1127016653792583], [0.5], [0.8872983346207417]]), 
           np.array([5./18., 4./9., 5./18])],
        4:[np.array([[0.33000947820757187, 0.6699905217924281, 0.06943184420297371, 0.9305681557970262]]).T,
           np.array([0.32607257743127305, 0.32607257743127305, 0.1739274225687269, 0.1739274225687269])],
        5:[np.array([[0.5, 0.230765344947, 0.769234655053, 0.0469100770307, 0.953089922969]]).T,
           np.array([0.284444444444, 0.23931433525, 0.23931433525, 0.118463442528, 0.118463442528])],
        6:[np.array([[0.8306046932331322, 0.1693953067668678, 0.3806904069584016, 0.6193095930415985, 0.0337652428984240, 0.9662347571015760]]).T,
           np.array([0.1803807865240693, 0.1803807865240693, 0.2339569672863455, 0.2339569672863455, 0.0856622461895852, 0.0856622461895852])],
}

@jax.custom_jvp
def mesh_embed_points(points, verbose=0, init_elexi=None, fit_params=None, surface_embed=False, iterations=3):
    """Find the parametric coordinates (element, xi) for a set of physical-space points.

    Uses an approximate nearest-neighbour search on a coarse xi grid to
    obtain initial estimates, then refines with a JAX-accelerated RK4
    fixed-iteration descent (see :meth:`_xis_to_points`).  Topology
    mapping (:meth:`topomap`) is applied at each iteration so that points
    near element boundaries are correctly assigned to neighbouring
    elements.

    Parameters
    ----------
    points:
        Physical-space query points, shape ``(n_pts, fdim)``.
    verbose:
        Verbosity level.  ``0`` → silent; ``2`` → print mean/max
        residual; ``3`` → also render an error visualisation with
        PyVista.
    init_elexi:
        Pre-computed initial ``(elem_num, xis)`` tuple.  When supplied,
        the coarse nearest-neighbour search is skipped.
    fit_params:
        Optional parameter override for the mesh geometry.
    return_residual:
        When ``True``, returns a ``((elem_num, embedded), residual)``
        tuple instead of just ``(elem_num, embedded)``.
    surface_embed:
        Restrict the coarse search to the surface faces of a 3-D mesh.
    iterations:
        Number of RK4 refinement iterations.

    Returns
    -------
    elem_num : jnp.ndarray
        Element index for each query point, shape ``(n_pts,)``.
    embedded : jnp.ndarray
        Parametric coordinates, shape ``(n_pts, ndim)``.
    residual : jnp.ndarray
        (Only when *return_residual* is ``True``) Embedding error
        vectors, shape ``(n_pts, fdim)``.
    """

    points = jnp.atleast_2d(points) #ensure correct shape and type
    
    if init_elexi is None: #do a coarse embedding
        if mesh.elements[0].ndim == 2:
            res = 40
            xis = jnp.asarray(mesh.xi_grid(res, 2, boundary_points=False))
            ndim = 2
            coarse_pts = mesh.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
            test_res, i_data = jax_aknn(points, coarse_pts, k=1)
            i = i_data[:, 0]
            elem_num = i // xis.shape[0]
            init_xi = xis[i % xis.shape[0]]

            #TODO 2D manifold embedding, it should maube be exactly the same
            at_lo = init_xi < 1e-6
            at_hi = init_xi > 1 - 1e-6
            mf_pt = at_lo | at_hi
        else:
            res = 40
            ndim = 3
            if not surface_embed:
                # Build interior grid
                xis = mesh.xi_grid(res, 3, boundary_points=True)
                n_pts = xis.shape[0]          # grid points per element
                coarse_pts = mesh.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                test_res, i_data = jax_aknn(points, coarse_pts, k=1)

                i = i_data[:, 0]
                elem_num = i // xis.shape[0]
                init_xi = xis[i % xis.shape[0]]
                
                init_ests = mesh.evaluate_embeddings_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                J_init = mesh.eval_numeric_jac_ele_xi_pair(elem_num, init_xi, fit_params=fit_params)
                proj_dir = jnp.sum((points - init_ests)[:, :, None] * J_init, axis=1) > 0

                # a further check is necessary here: does the mf edge actually sit on a boundary?

                at_lo = init_xi < 1e-6
                at_hi = init_xi > 1 - 1e-6
                
                mf_lo = at_lo & ~proj_dir #is the point on the manifold?
                mf_hi = at_hi & proj_dir
                init_xi += (~mf_lo & at_lo) * 2e-2 - (~mf_hi & at_hi)*2e-2

                mf_pt = mf_lo | mf_hi

            else:
                # surface_embed=True: embed on mesh surface faces only (unchanged)
                faces = mesh.faces
                face_pts = []
                elem_pts = []
                xi_pts = []
                xi3grid = mesh.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)
                for face in faces:
                    grid_def = xi3grid[face[1], face[2]]
                    elem_pts.append(np.ones(grid_def.shape[0]) * face[0])
                    xi_pts.append(grid_def)
                    face_pts.append(mesh.evaluate_embeddings(jnp.array([face[0]]),grid_def))
                coarse_pts = jnp.concatenate(face_pts, axis=0)
                elems = jnp.concatenate(elem_pts, axis=0)
                xis = jnp.concatenate(xi_pts, axis=0)
                test_res, i_data = jax_aknn(points, coarse_pts, k=1)
                i = i_data[:, 0]
                elem_num = elems[i]
                init_xi  = xis[i]

                at_lo = init_xi < 1e-6
                at_hi = init_xi > 1 - 1e-6
                mf_pt = at_lo | at_hi
    else:
        elem_num, init_xi = init_elexi
        elem_num = jnp.atleast_1d(elem_num)
        init_xi = jnp.atleast_2d(init_xi)
        ndim = mesh.elements[0].ndim

        test_res = points - mesh.evaluate_embeddings_ele_xi_pair(elem_num, init_xi)

        at_lo = init_xi < 1e-6
        at_hi = init_xi > 1 - 1e-6
        mf_pt = at_lo | at_hi

    (elem_num, embedded), res = jax.vmap(
        lambda elem, xi, target, rmag, lbound : mesh._xis_to_points(elem, xi, target, lbound, rmag, iterations=iterations, fit_params=fit_params)
    )(elem_num, init_xi, points, mf_pt, jnp.linalg.norm(test_res, axis=-1))

    # elem_num, embedded, res = elem_num, init_xi, test_res

    if verbose >= 2:
        final_mean_dist = np.mean(np.linalg.norm(np.asarray(res), axis=-1))
        final_max_dist  = np.max(np.linalg.norm(np.asarray(res), axis=-1))
        print(f"final mean error of {final_mean_dist} units, max error of {final_max_dist}")

    if verbose == 3:
        locs = mesh.evaluate_embeddings_ele_xi_pair(elem_num, embedded)
        vec_errors = points - locs
        errors = np.linalg.norm(vec_errors, axis=-1)

        line_segs = np.concatenate(
            (np.atleast_2d(locs)[:, None], np.atleast_2d(points)[:, None]), axis=1
        ).reshape(-1, mesh.fdim)
        s = pv.Plotter()
        mesh.plot(s)
        data = pv.PolyData(np.asarray(locs))
        data['err'] = errors
        s.add_mesh(pv.line_segments_from_points(line_segs), color='k')
        s.add_mesh(data, render_points_as_spheres=True, point_size=20)
        s.show()

    return (elem_num, embedded), res


@mesh_embed_points.defjvp
def embed_pts_ptderiv(primal, cotangent):
    """
    Follows jax protocol to define the jax compatable derivatives.
    Gives the local derivatives of the point embedding with respect to the input points to embed.

    It calculates the linear derivatvies of elem_num, xi, and the residual of the mesh embed function.
    elem_num derivative is always zero.
    
    xi is the main point of interest.
    res is useful for fitting.

    """
    mesh = primal[0]
    x = primal[1]
    breakpoint()
    (ele, xi), res = primal_out = mesh_embed_points(*primal)

    """
    define the derivatives with respect to the second argument, "points"
    Describes how the ele xi positions change with response to the locations of the points. 
    """
    tval = None if len(primal) == 1 else primal[1]
    jacs = mesh.evaluate_jacobians_ele_xi_pair(ele, xi, fit_params=tval) 
    a0_local_xi_deriv = jnp.linalg.solve(jacs, x) #get the results over this space.
    #this is then a block diagonal value
    a0_local_el_deriv = jnp.zeros(ele.shape, x.shape)

    #the residual of the res vector is nice and simple - it changes with the input points.
    a0_local_res_deriv = cotangent[0]

    """
    Following section implements the derivatives with respect to the mesh embedding.
    gives the local deriates of the point embedding with respect to the mesh parameters.
    """

    deriv_fn = jax.jacfwd(mesh.evaluate_embeddings_in_ele_xi, argnums=3)
    local_param_derivs = deriv_fn(ele, xi, fit_params=primal[4])

    #can then just do the same tangenting
    a1_local_el_deriv = jnp.zeros(ele.shape, x.shape)

    a1_local_xi_deriv = local_param_derivs @ cotangent_out

    a1_loacl_res_deriv





    #formatting of the cotangent out is wrong, please fix.
    cotangent_out = (local_el_deriv, tangented_xi)
    

    return primal_out, cotangent_out

    
    

