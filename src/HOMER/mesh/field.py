"""
field.py - :class:`MeshField`, a vector-valued field over a mesh topology.

The class owns the mesh state - its nodes, elements, parameter vector and the
JAX closures compiled from them - and the lifecycle that keeps those in step
(:meth:`MeshField.generate_mesh`).  Everything a field *does* with that state
lives in a sibling module and is bound into the class body below:

* :mod:`HOMER.mesh.evaluation` - evaluating the field and its derivatives
* :mod:`HOMER.mesh.parameters` - the parameter vector, and fitting it
* :mod:`HOMER.mesh.topology`   - connectivity, faces, surfaces, colouring
* :mod:`HOMER.mesh.refinement` - refine and rebase
* :mod:`HOMER.mesh.plotting`   - drawing

Binding rather than inheriting is deliberate: ``@expand_wide_evals`` reads
``vars(cls)``, so a method reached through a base class would be invisible to
it and the generated ``*_in_every_element`` / ``*_ele_xi_pair`` variants would
silently disappear.
"""

from copy import deepcopy
from os import PathLike
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from HOMER.embedding import build_embedding_fn
from HOMER.mesh import plotting, evaluation, parameters, topology, refinement
from HOMER.mesh.node import MeshNode
from HOMER.mesh.element import MeshElement
from HOMER.mesh.element_eval import make_eval, make_deriv_eval, make_weight_eval
from HOMER.mesh_decorators import (expand_wide_evals, wide_eval,
                                   DEFAULT_EVAL_CHUNK_SIZE, DEFAULT_EVAL_REMAT)
from HOMER.utils import h_tform

if TYPE_CHECKING:
    from HOMER.mesh.mesh import Mesh


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

    #--- methods defined in sibling modules
    get_surface = plotting.get_surface
    get_hex_surface = plotting.get_hex_surface
    get_triangle_surface = plotting.get_triangle_surface
    get_lines = plotting.get_lines
    plot = plotting.plot
    plot_strains = plotting.plot_strains
    evaluate_embeddings = evaluation.evaluate_embeddings
    evaluate_deriv_embeddings = evaluation.evaluate_deriv_embeddings
    evaluate_element_embeddings = evaluation.evaluate_element_embeddings
    evaluate_normals = evaluation.evaluate_normals
    eval_numeric_jac = evaluation.eval_numeric_jac
    evaluate_jacobians = evaluation.evaluate_jacobians
    xi_grid = evaluation.xi_grid
    gauss_grid = evaluation.gauss_grid
    eval_surface = evaluation.eval_surface
    embed_points = evaluation.embed_points
    evaluate_sobolev = evaluation.evaluate_sobolev
    get_volume = evaluation.get_volume
    evaluate_strain = evaluation.evaluate_strain
    get_element_params = parameters.get_element_params
    update_from_params = parameters.update_from_params
    unfix_mesh = parameters.unfix_mesh
    get_xi_weight_mat = parameters.get_xi_weight_mat
    linear_fit = parameters.linear_fit
    associated_node_index = topology.associated_node_index
    _explore_topology = topology._explore_topology
    get_xi_surface_nodes = topology.get_xi_surface_nodes
    get_faces = topology.get_faces
    topo_chain_check = topology.topo_chain_check
    _update_id_mappings = topology._update_id_mappings
    _clean_pts = topology._clean_pts
    get_colouring_dict = topology.get_colouring_dict
    refine = refinement.refine
    rebase = refinement.rebase

    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False, skip_generate=False) -> None:
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
        self.generate_weight_matrix_T: Optional[Callable] = None

        self.faces = None

        ######### optimisation
        self.true_param_array: Optional[np.ndarray] = None
        self.optimisable_param_array: Optional[np.ndarray] = None
        self.optimisable_param_bool: Optional[np.ndarray] = None
        self.ele_map: Optional[np.ndarray] = None
        
        ######### field stuff
        self.fdim = None

        ######### support for element scales
        self.ele_scales = None

        ######### Compilation flags
        self.compile = jax_compile
        #: max point evaluations per chunk in the generated wide-eval wrappers
        self.eval_chunk_size: int = DEFAULT_EVAL_CHUNK_SIZE
        #: rematerialise chunk intermediates when differentiating a wide eval
        self.eval_remat: bool = DEFAULT_EVAL_REMAT
        if not len(self.nodes) == 0 and not len(self.elements) == 0 and not skip_generate:
            self.generate_mesh()
    
        
    ################################## MAIN FUNCTIONS
    ################################## CONVENIENCE
    ################################## MORPHIC INTERFACE COMPATIBILITY
    def generate_mesh(self) -> None:
        """
        Builds the mesh representation on call.

        This code is responsible for handling on-the-fly functions, and the generation of the
        'fast' pathway jax.numpy array representation.

        """

        self.fdim = self.nodes[0].loc.shape[0] if self.fdim is None else self.fdim
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

        self.ele_scales = None #get this done so it's captured in the closure
        if self.elements[0].scale_factors is not None:
            self.ele_scales = np.array([e.scale_factors for e in self.elements])


        self._generate_elem_functions()
        self._generate_elem_deriv_functions()
        self._generate_eval_function()
        self._generate_deriv_function()
        self._generate_weight_function()
        self._explore_topology()
        self._generate_embedding_function()

        #adding scalar mapping support.

    def add_node(self, node:MeshNode) -> None:
        """
        Add a node to the node list.
        """
        self.nodes.append(node)
        # self.generate_mesh()

    def add_element(self, element:MeshElement, generate_mesh=True) -> None:
        """
        Adds an element to the element list.
        """
        self.elements.append(element)
        if generate_mesh:
            self.generate_mesh()

    def drop_elements(self, inds_to_drop, generate_mesh=True, clean_points=True) -> None:
        """
        Drops the specified elements.
        """
        if not isinstance(inds_to_drop, list):
            inds_to_drop = [inds_to_drop]
        self.elements = [val for i, val in enumerate(self.elements) if i not in inds_to_drop]

        if clean_points:
            self._clean_pts()
        if generate_mesh:
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

    ################################## PLOTTING
        # self.topomap = topomap

    def _generate_embedding_function(self):
        """Build the JIT-compiled embedding function via :mod:`HOMER.embedding`.

        Creates ``self._mesh_embed_points``, a ``@jax.custom_jvp``
        function reused by every :meth:`embed_points` call, avoiding
        redundant XLA retracing.
        """
        self._mesh_embed_points = build_embedding_fn(self)

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
        def evaluate_embeddings(element_ids, xis, fit_params = self.optimisable_param_array, ele_map = self.ele_map, scalars = self.ele_scales):
            element_ids = jnp.atleast_1d(jnp.array(element_ids))
            xis = jnp.atleast_2d(jnp.array(xis))

            param_data = jnp.asarray(self.true_param_array)
            if fit_params is not None:
                if not len(fit_params) == len(param_data):
                    fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)
            else:
                fit_params = param_data

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            if scalars is not None:
                scalar_factor = jnp.array(scalars)[jnp.asarray(element_ids)]
            else:
                scalar_factor = 1
            params = jnp.asarray(fit_params)[map] * scalar_factor
            outputs = jax.vmap(lambda x: self.elem_evals(x, jnp.asarray(xis)).reshape(-1,self.fdim))
            res = outputs(
                params
            )
            return res.reshape(-1,self.fdim)
        
        self.evaluate_embeddings = evaluate_embeddings

    def _generate_deriv_function(self):
        """
            Generates the internal functions that evaluate the derivatives of embeddings
            Code is structured so that the result can express custom derivatives
        """
        @wide_eval
        def evaluate_deriv_embeddings(element_ids, xis, derivs, fit_params = self.optimisable_param_array, ele_map= self.ele_map, scalars = self.ele_scales):
            element_ids = jnp.atleast_1d(jnp.array(element_ids))
            xis = jnp.atleast_2d(jnp.array(xis))

            param_data = jnp.asarray(self.true_param_array)
            if fit_params is not None:
                if not len(fit_params) == len(param_data):
                    fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)
            else:
                fit_params = param_data
                    
            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            if scalars is not None:
                scalar_factor = jnp.array(scalars)[jnp.asarray(element_ids)]
            else:
                scalar_factor = 1
            params = jnp.asarray(fit_params)[map] * scalar_factor

            outputs = jax.vmap(lambda x: self.elem_deriv_evals(x, jnp.asarray(xis), derivs).reshape(-1,self.fdim))
            res = outputs(params)
            return res.reshape(-1,self.fdim)
        
        self.evaluate_deriv_embeddings = evaluate_deriv_embeddings

    def _generate_weight_function(self):
        """
        Creastes the weight matrix of the mesh. Useful for direct linear fitting with constant xi embeddings.
        """
        self.generate_weight_matrix = make_weight_eval(self.elements[0].basis_functions, self.elements[0].BasisProductInds)
        #for large functions, it can be desirable to generate the transpose directly.
        self.generate_weight_matrix_T = jax.vmap(lambda xu:self.generate_weight_matrix(jnp.atleast_2d(xu)))



    ################################# useful utils.

    ################################# FASTFITTING

    ################################# REFINEMENT
    def save(self, loc: PathLike):
        """
        Saves the field to a .json formated file in the given location
        """
        from HOMER.io import save_mesh #avoid the circular import here
        save_mesh(self, loc)

    def dump_to_dict(self):
        """
        Returns a dict structure representing the field object, for ease of saving
        """

        from HOMER.io import dump_meshfield_to_dict
        return dump_meshfield_to_dict(self)

    def __deepcopy__(self, memo):
        """
        Dumps the field to a dictionairy then rebuilds it to ensure that there is no shared memory between a field and it's deepcopy.
        """

        from HOMER.io import dump_meshfield_to_dict, parse_meshfield_from_dict
        dict_rep = deepcopy(dump_meshfield_to_dict(self))
        return parse_meshfield_from_dict(dict_rep)


    def __add__(self, othr:'MeshField'): #TODO, robustify this function, as it has only basic functionality.
        """
        A quick function to enable adding meshes together by fusing the element and basis definititions.
        """
        shared_nodes = self.nodes + othr.nodes
        len_self_nodes = len(self.nodes)
        updated_other_elements = [MeshElement(basis_functions=e.basis_functions, node_indexes=[n+len_self_nodes for n in e.nodes]) for e in othr.elements] 
        shared_elements = self.elements + updated_other_elements
        from HOMER.mesh.mesh import Mesh  #deferred: Mesh subclasses this class
        return Mesh(nodes=shared_nodes, elements=shared_elements)
