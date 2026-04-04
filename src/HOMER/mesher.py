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
from HOMER.utils import vol_hexahedron, make_tiling, h_tform, all_pairings, block_diagonal_jacobian, jax_aknn
from HOMER.mesh_decorators import expand_wide_evals, wide_eval

class MeshNode(dict):
    def __init__(self, loc, id=None, **kwargs):
        """
        The base node class, handling arbitrary properties over the mesh surface.
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
        """
        Given the node parameter strings, identifies the nodes as fixed nodes, which are not part of the default optimisable parameters of a mesh.

        :param param_names: The parameter name to fix
        :param values: The optional value of the parameter to be fixed too.
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
    def __init__(self, basis_functions: BasisGroup, node_indexes: Optional[list[int]] = None, 
                 node_ids: Optional[list] = None, BP_inds: Optional = None, id=None):
        """
            A high order mesh element. This element is constructed from a series of basis functions.
            These can be 2D, manifold meshes, or 3D, volume meshes.

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
    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        """
        Defines a collection of homogenous elements, which link a collection of nodes.
        A mesh fits a collection of vector valued fields. The fundamental field is the world space embedding.
        It is this value that is used to initially define the mesh. 
        The goal of this implementation is to allow the fit field to be optimised at the same time as the mesh topology.

        :param nodes: The nodes of the Mesh
        :param elements: The elements of the Mesh.
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
        if not typing:
            raise RuntimeError('Called evaluate_embeddings before initialisation')
        return

    @wide_eval
    def evaluate_deriv_embeddings(self, *a, **kw): #placeholder for later func definition
        if not typing:
            raise RuntimeError('Called evaluate_deriv_embeddings before initialisation')
        return

    def evaluate_element_embeddings(self, element_id, xis, fit_params=None):
        """
        Given an element id, evaluates the embedding.
        
        :param element_id: The element id to evaluate the xi location in
        :param xis: The xis to evaluate.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        return self.evaluate_embeddings([self.element_id_to_ind[element_id]], xis, fit_params=fit_params)
     
    @wide_eval
    def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray, fit_params=None) -> np.ndarray:
        """
        Returns the normal at the element surface.
        Only valid for manifold meshes.
        :params elemend_ids: The elements to evaluate
        :params xis: the locations to evaluate.
        :returns normals: unit vector directions associated with the mesh surface.
        """

        if self.ndim == 3: 
            raise ValueError("Normals aren't defined on a volume mesh")
        if fit_params is None:
            fit_params = self.optimisable_param_array

        d0 = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1], fit_params) 
        d1 = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0], fit_params)
        return jnp.cross(d0, d1)

    @wide_eval
    def evaluate_jacobians(self, element_ids, xis, fit_params=None):
        """ 
        Evaluates the jacobian at a set of xis within an element
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

        """
        Convinience function for defining grids of xi points over elements on the surface of the mesh.
        Returns a grid of xi points at the requested resolution.

        :param res: the mxm resolution.
        :param dim: the dimension of the grid, i.e. 2 or 3.
        :param surface: for a 3D grid, only return grid points on the element boundary.
        :param boundary_points: whether to include 0, 1 in the grid of points. Prevents doubly selecting points on mesh boundarys.

        :returns xi_grid: a np array containing grid points.

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
        """
        returns a grid of gauss points
        :params ng: the number of gauss points to evaluate, either in 1D or a grid of gauss points.
        :returns X, W: The location and weighting of the gauss points for the function order.
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

        locs = np.round(locs, rounding_res)
        _, idx, inv, cnt = np.unique(
            locs, axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True
        )
        faces = []
        bmap = {}

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
            elif cn > 2:
                raise ValueError("Mesh had multiple elements intersecting at a single point")
        #face if once, 
        # test_faces = self.get_faces()
        self.faces = faces
        self.bmap = bmap

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

    def embed_points(self, points, verbose=0, init_elexi=None, fit_params=None, return_residual=False, surface_embed=False):
        """
        Given an mx3 array of points, returns the embedded xi locations which best match these points.
        Minimises the squared distance between the embedded locations and the given points, as a non linear least squares.

        :param points: The points to embed
        :param verbose: Level of information printed by the least squares fitting process
        :param init_elexi: Initial locations of the points, just so skip straigh to optimising
        :param surface_embed: Whether to specifically embed the points into the surface
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        points = jnp.atleast_2d(points) #ensure correct shape and type
        
        if init_elexi is None: #do a coarse embedding
            if self.elements[0].ndim == 2:
                res = 40
                xis = jnp.asarray(self.xi_grid(res, 2, boundary_points=False))
                ndim = 2
                coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
            else:
                res = 20
                if not surface_embed:
                    #evaluate surface points,

                    #evaluate shared boundaries.

                    #evaluate an internal grid.

                    #find the closest point
                    #if its a surface point, confirm positive residual along surface xi coodinate
                    #and if not, bump inwards a fraction.
                    #if its a shared boundary, note the embedding ambiguity (optimise both possible positions.
                    #otherwise, it's a normal object.


                    xis = jnp.asarray(self.xi_grid(res, 3, boundary_points=False))
                    ndim = 3
                    coarse_pts = self.evaluate_embeddings_in_every_element(xis, fit_params=fit_params)
                else:
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
            if not surface_embed:
                elem_num = i // xis.shape[0]
            else:
                elem_num = elems[i]

            xi_ind = i % xis.shape[0]
            if not surface_embed:
                init_xi = xis[xi_ind]
            else:
                init_xi = xis[i]
        else:
            elem_num, init_xi = init_elexi
            elem_num = jnp.atleast_1d(elem_num)
            init_xi = jnp.atleast_2d(init_xi)
            ndim = self.elements[0].ndim

        def _pseudoinverse_matvec(J: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            Jt_v = J.T @ v          # (d,) — project v onto tangent space
            JtJ = J.T @ J        # (d, d) Gram matrix 
            dxi, _, _, _ = jnp.linalg.lstsq(JtJ, Jt_v, rcond=None)
            return dxi
        
        n_steps = 5
        @jax.jit
        def xis_to_points(elem, xi0, x_target):
            xi = xi0.copy()
            d = xi0.shape[0]
            lo = jnp.zeros(d) 
            hi = jnp.ones(d) 
            
            # Step size / relaxation factor
            dt = 1.5 / n_steps 
            for _ in range(n_steps):
                current_x = self.evaluate_embeddings(elem, xi, fit_params=fit_params)[0]
                r0 = x_target - current_x
                J = self.evaluate_jacobians(elem, xi, fit_params=fit_params)[0] # (n, d)
                on_lo = xi <= lo + 1e-12
                on_hi = xi >= hi - 1e-12
                bound = on_lo | on_hi                        # (d,)
                J_free = J * jnp.where(bound, 0.0, 1.0)
                dxi = jnp.where(jnp.all(bound), 0, _pseudoinverse_matvec(J_free, r0))
                xi = xi + dxi* jnp.where(jnp.any(bound),dt, 1) #step sizing should be adaptive for interior and exterior points.
                xi = jnp.clip(xi, lo, hi)

            residual = self.evaluate_embeddings(elem, xi, fit_params=fit_params) - x_target
            return xi, residual

        # for e, x, p in zip(elem_num, init_xi, points):
        #     embedded, res = xis_to_points(e, x, p)
        embedded, res = jax.vmap(xis_to_points)(elem_num, init_xi, points) 
        # embedded, res = init_xi, test_res 

        # if verbose >= 2:
        #     final_mean_dist = np.mean(np.linalg.norm(res, axis=-1))
        #     final_max_dist = np.max(np.linalg.norm(res, axis=-1))
        #     print(f"final mean error of {final_mean_dist} units, max error of {final_max_dist}")

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
            return (elem_num, embedded), res 

        return elem_num, embedded

        def optim_embed(params):
            xi_data = params.reshape(-1,ndim)
            return (self.evaluate_embeddings_ele_xi_pair(elem_num, xi_data) - points).ravel()

        spm = block_diagonal_jacobian(self.fdim, self.ndim, points.shape[0])
        def jac(params):
            xi_data = params.reshape(-1,ndim)
            data = jnp.swapaxes(self.evaluate_jacobians_ele_xi_pair(elem_num, xi_data), -1,-2)
            spm.data = np.asarray(data.ravel())
            return spm

        bounds = (np.zeros_like(init_xi), np.ones_like(init_xi))
        if verbose == 2: 
            print("Beginning embedding")
        result = least_squares(optim_embed, init_xi, jac=jac, bounds=bounds, verbose=min(verbose,2))

        final_mean_dist = np.mean(np.linalg.norm(result.fun.reshape(-1, self.fdim), axis=-1))
        final_max_dist = np.max(np.linalg.norm(result.fun.reshape(-1, self.fdim), axis=-1))
        if verbose == 2:
            print(f"final mean error of {final_mean_dist} units, max error of {final_max_dist}")

        if verbose == 3:
            locs = self.evaluate_ele_xi_pair_embeddings(elem_num, result.x.reshape(-1, ndim))
            vec_errors = points - locs
            errors = np.linalg.norm(vec_errors, axis=-1)
            s = pv.Plotter()
            self.plot(s)
            data = pv.PolyData(points)
            data['err'] = np.log(errors + 1e-16)
            s.add_mesh(data, render_points_as_spheres=True, point_size=20)
            s.show()

        return elem_num, result.x.reshape(-1,ndim)


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
        """
        Assesses the strain in a deformed state at a set of given locations.
        :param element_ids: The elements to asses strain in.
        :param xis: The xi locations to evaluate strain in.
        :param othr: A second mesh object with the same topology to assess strain against.
        :param coord_function: A function with input Mesh, eles, xis, tensors -> remapped_tensors. Used to evaluate strains in relevant coordinate schemes.
        """

        if self.ndim == 2 and coord_function is None:
            raise ValueError("Strain tensor on manifold mesh requires a coord function to provide a meaninful basis")

        deriv_self = self.evaluate_jacobians(element_ids, xis, fit_params=fit_params)
        deriv_othr = othr.evaluate_jacobians(element_ids, xis, fit_params=None)

        if coord_function is not None:
            deriv_self = coord_function(self, element_ids, xis, deriv_self)
            deriv_othr = coord_function(othr, element_ids, xis, deriv_othr)
        
        # str_tensor = jnp.linalg.inv(deriv_self) @ deriv_othr
        F = jnp.linalg.solve(deriv_self, deriv_othr)
        # F = str_tensor.reshape(-1, self.ndim, self.ndim)
        if return_F:
            return F
  
        strain = (F.transpose(0,2,1) @ F - np.eye(self.ndim)[None])/2
        return strain.reshape(-1, self.ndim, self.ndim)

    ################################# FASTFITTING

    def get_xi_weight_mat(self, eles, xis):
        """
        Given an input set of poinst ele, yi, evaluates the meta-weight matrix that can be inverted to solve the least squares shape update.
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
        """
        Performs a linear fit between the target using the current weight matrix.
        Does not currently respect fixed parameters (maybe an Ax + c = b situation)
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
        """
            This code refines a mesh, increasing the resolution of every element.
            A refinement factor can be provided, or alternatively a tuple of xi_locations to refine the nodes at can be provided.
        :param refinement_factor: an integer factor to increase the resolution by.
        :param by_xi_refinement: an ndtuple of monotonically increasing points to sample the new nodes at.
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
        """
        Rebases a mesh by (1) determining any new node locations, (2) sampling a xi grid in the old basis, and then
        (3) linearly fitting the parameters from the xi embedding.
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
    """ 
    A mesh is a collection of fields, with a shared topology and coordinate system.
    By default, a mesh has one spatial feild, but this is not necessary.

    """

    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        #first, define the XYZ mesh_field
        super().__init__(nodes, elements, jax_compile)
        self.fields = {}

    def __getitem__(self, input: str) -> MeshField:
        return self.fields[input]

    def __setitem__(self, key: str, value: MeshField):
        assert len(self.elements) == len(value.elements), 'Fields must have the same number of elements'
        assert self.elements[0].ndim == value.elements[0].ndim, 'Feilds must share the same dimensionality of basis components'
        self.fields[key] = value

    def refine(self, refinement_factor: Optional[int] = None, by_xi_refinement: Optional[tuple[np.ndarray]] = None, clean_nodes=True):
        super().refine(refinement_factor, by_xi_refinement, clean_nodes)
        for field in self.fields.values():
            field.refine(refinement_factor, by_xi_refinement, clean_nodes)

    def plot(self, scene: Optional[pv.Plotter] = None, node_colour='r', node_size=10, labels=False, tiling=(10, 6), 
             mesh_colour: str | np.ndarray = 'gray', mesh_opacity=0.1, mesh_width=2, mesh_col_scalar_name="Field", 
             line_colour: str | np.ndarray = 'black', line_opacity=1, line_width=2, line_col_scalar_name="Field",
             elem_labels=False, render_name: Optional[str] = None, 
             field_to_draw = None, field_xi = None, draw_xyz_field = True, field_artist: Optional[Callable[[pv.Plotter, np.ndarray, np.ndarray], None]] = None,
             default_field_point_size=25, default_xi_res=4):

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
        """
        Defines a new field over the previous mesh topology using the same methodology used for rebasing.
        Creates a field by (1) determining any new node locations and topologies.
        If field locations and values are fo(2) 
        (3) linearly fitting the parameters from the xi embedding.
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

def make_deriv_eval(basis_funcs: BasisGroup, bp_inds:list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a 
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, d_inds, b_inds = bp_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])  
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, d_inds, b_inds = bp_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])  
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            w2 = basis_funcs[2].deriv[d_inds[2]](xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(weights.shape[0],-1)[:, None] * weights[..., None], axis=0).flatten()
            return output
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

