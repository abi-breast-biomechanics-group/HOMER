import logging
from typing import Optional, Callable
import numpy as np
import jax.numpy as jnp
import jax
import pyvista as pv
from matplotlib import pyplot as plt

from functools import reduce, partial
from itertools import groupby, combinations_with_replacement, product

from scipy.spatial import KDTree
from scipy.optimize import least_squares

from HOMER.basis_definitions import N2_weights, N3_weights, AbstractBasis, BasisGroup, DERIV_ORDER, EVAL_PATTERN
from HOMER.jacobian_evaluator import jacobian
from HOMER.utils import vol_hexahedron, make_tiling



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
        if inds is not None:
            inds = np.array(inds).astype(int)
        if isinstance(param_names, str):
            param_names = [param_names]
        if not isinstance(values, list):
            values = [values] * len(param_names)

        for idp, param in enumerate(param_names):
            if inds is None:
                inds = [0,1,2]
            if param in self.fixed_params:
                total = list(set(self.fixed_params[params]) + set(inds))
                self.fixed_params[params] = total
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
        free_loc = np.ones(3)
        free_loc[self.fixed_params.get('loc', [])] = 0
        list_data = [free_loc]
        for key in self.keys():
            free_key = np.ones(3)
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





class Mesh:
    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        """
        Defines a collection of homogenous elements, which link a collection of nodes.
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
        self.evaluate_embeddings = lambda element_ids, xis, params: None #def the function signature here
        self.evaluate_deriv_embeddings = lambda element_ids, xis, derivs, params: None
        self.elem_deriv_evals: Optional[Callable] = None
        self.fit_param: Optional[np.ndarray] = None

        ######### optimisation
        self.true_param_array: Optional[np.ndarray] = None
        self.optimisable_param_array: Optional[np.ndarray] = None
        self.optimisable_param_bool: Optional[np.ndarray] = None
        self.ele_map: Optional[np.ndarray] = None

        ######### Compilation flags
        self.compile = jax_compile
        if not len(self.nodes) == 0 and not len(self.elements) == 0:
            self.generate_mesh()

    ################################## MAIN FUNCTIONS

    def evaluate_element_embeddings(self, element_id, xis):
        """
        Given an element id, evaluates the embedding.
        
        :param element_id: The element id to evaluate the xi location in
        :param xis: The xis to evaluate.
        """
        return self.evaluate_embeddings([self.element_id_to_ind[element_id]], xis)
    
    def evaluate_embeddings_in_every_element(self, xis, fit_params=None):
        """
        Wrapper around evaluate embeddings that evaluates the embeddings in every element.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        return self.evaluate_embeddings(jnp.array(list(range(len(self.elements)))).astype(int), xis, fit_params=fit_params)

    def evaluate_deriv_embeddings_in_every_element(self, xis, derivs, fit_params=None):
        """
        Wrapper around evaluate deriv embeddings that evaluates the embeddings in every element.
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        return self.evaluate_deriv_embeddings(jnp.array(list(range(len(self.elements)))).astype(int), xis, derivs, fit_params=fit_params)

    def evaluate_ele_xi_pair_embeddings(self, eles, xis, fit_params=None):
        """
        Wrapper around evaluate embedings that evaluates the embaeddings at pairs of ele xi coordinates.
        """
        if fit_params is None:
            fit_params = self.true_param_array

        unique_elem, inv = np.unique_inverse(eles)
        out_array = jnp.zeros((xis.shape[0], 3))
        for ide, e in enumerate(unique_elem):
            mask = ide == inv
            out_array = out_array.at[mask].set(self.evaluate_embeddings(element_ids=[e], xis=xis[mask], fit_params=fit_params))
        return out_array

    def evaluate_ele_xi_pair_deriv_embeddings(self, eles, xis, derivs, fit_params=None):
        """
        Wrapper around evaluate deriv embeddings that evaluates the embeddings at pairs of ele and xi coordinates.
        """
        if fit_params is None:
            fit_params = self.true_param_array

        unique_elem, inv = np.unique_inverse(eles)
        out_array = jnp.zeros((xis.shape[0], 3))
        for ide, e in enumerate(unique_elem):
            mask = ide == inv
            out_array = out_array.at[mask].set(
                self.evaluate_deriv_embeddings([e], xis[mask], derivs, fit_params=fit_params))
        return out_array

    def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray) -> np.ndarray:
        """
        Returns the normal at the element surface.
        Only valid for manifold meshes.
        :params elemend_ids: The elements to evaluate
        :params xis: the locations to evaluate.
        :returns normals: unit vector directions associated with the mesh surface.
        """
        raise NotImplementedError()

    ################################## CONVENIENCE
    def xi_grid(self, res: int, dim=2, surface=False) -> np.ndarray:
        if dim == 2:
            X, Y = np.mgrid[:res, :res] / (res - 1)
            return np.column_stack((X.flatten(), Y.flatten()))
        else:
            if not surface:
                X, Y, Z = np.mgrid[:res, :res, :res] / (res - 1)
                return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            else:
                raw_x = np.array([x.flatten() for x in np.mgrid[:res, :res] / (res - 1)])
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
                return np.concatenate(arrays)

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
                gindex = np.array([gindex[n].flatten() for n in [0, 1, 2]]).T  # doesn't seem to work as default
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
        """

        if len(inp_params) == len(self.optimisable_param_array):
            params = self.true_param_array.copy()
            params[self.optimisable_param_bool] = inp_params
        elif len(inp_params) == len(self.true_param_array):
            params = inp_params
        else:
            raise ValueError(
                "Input param array was provided that did not match either that set of parameters, or the optimisable subset of parameters")

        for node in self.nodes:
            node.loc, params = params[:3], params[3:]
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
        'fast' pathway numpy array representation.

        """

        self.true_param_array = np.concatenate(
            [np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()
        self.optimisable_param_bool = np.concatenate([node.get_optimisability_arr() for node in self.nodes],
                                                     axis=0).astype(bool)
        self.optimisable_param_array = self.true_param_array[self.optimisable_param_bool]

        self.update_from_params(np.arange(self.true_param_array.shape[-1]), generate=False)

        ########## build the lookup from the input values.
        self.node_id_to_ind = {}
        self.element_id_to_ind = {}

        for e, n in [(e, n) for e, n in enumerate(self.nodes) if n.id is not None]:
            key_in = self.node_id_to_ind.get(n.id, None)
            if key_in is not None:
                raise ValueError(f"Duplicate nodes with the id: {n.id} were added to the mesh")
            self.node_id_to_ind[n.id] = e

        for e, el in [(e, el) for e, el in enumerate(self.elements) if el.id is not None]:
            key_in = self.element_id_to_ind.get(el.id, None)
            if key_in is not None:
                raise ValueError(f"Duplicate nodes with the id: {el.id} were added to the mesh")
            self.element_id_to_ind[el.id] = e

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

    def add_node(self, node: MeshNode) -> None:
        self.nodes.append(node)
        # self.generate_mesh()

    def add_element(self, element: MeshElement) -> None:
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

    def associated_node_index(self, index_list: list, nodes_to_gather: Optional[list] = None, node_by_id=False):
        """
        Given an index list, returns the associated indexes of features in that index in the input param array.
        Used to perform manipulations
        """
        true_param_array = np.concatenate(
            [np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()
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

    ################################## PLOTTING
    def get_surface(self, element_ids: Optional[np.ndarray] = None, res: int = 20, just_faces=False) -> np.ndarray:
        """
        Returns a set of points evaluated over the mesh surface.
        """
        ele_iter = [element_ids] if not isinstance(element_ids, list) else element_ids
        elements_to_iter = self.elements if element_ids is None else ele_iter
        if not just_faces:
            grid = self.xi_grid(res=res, ndim=self.elements[0].ndim, surface=True)
            if element_ids is not None:
                all_points = []
                for ne, e in enumerate(elements_to_iter):
                    all_points.append(self.evaluate_embeddings(np.array([ne]), grid))
                return np.concatenate(all_points, axis=0)
            else:
                return self.evaluate_embeddings_in_every_element(grid)
        else:
            face_pts = []

            if self.elements[0].ndim == 3:
                faces = self.get_faces()
                xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3, 2, -1, 3)
                for face in faces:
                    element = self.elements[face[0]]
                    grid_def = xi3grid[face[1], face[2]]
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]), grid_def))
                return np.concatenate(face_pts, axis=0)
            else:
                xi2grid = self.xi_grid(res=res, dim=2)
                return np.asarray(self.evaluate_embeddings_in_every_element(xi2grid))

            faces = self.get_faces()
            for face in faces:
                xi2grid = self.xi_grid(res=res, dim=2)
                xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3, 2, -1, 3)

                element = self.elements[face[0]]
                if element.ndim == 2:
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]), xi2grid))
                elif element.ndim == 3:
                    grid_def = xi3grid[face[1], face[2]]
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]), grid_def))
            return np.concatenate(face_pts, axis=0)

    def get_triangle_surface(self, element_ids: Optional[np.ndarray] = None, res: int = 20) -> tuple[
        np.ndarray, np.ndarray]:
        """
        Returns a set of points evaluated over the mesh surface, and triangles to create the surface.

        :returns surface pts: Surface points evaluated over the mesh.
        :returns tris: the triangles creatign the mesh surface.
        """
        base_0 = np.array([0, 1, res])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(res - 1) * res)[:,
                                                                                         None, None]
        base_1 = np.array([res, 1, res + 1])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(
            res - 1) * res)[:, None, None]
        surface_pts = self.get_surface(element_ids, just_faces=True, res=res)
        n_surfaces = surface_pts.shape[0] / (res ** 2)
        tris = (np.concatenate((base_0.reshape((-1, 3)), base_1.reshape((-1, 3))))[None] + np.arange(n_surfaces)[:,
                                                                                           None,
                                                                                           None] * res ** 2).reshape(-1,
                                                                                                                     3)

        return surface_pts, tris

    def get_lines(self, element_ids: Optional[list[int] | int | np.ndarray] = None, res=10) -> pv.PolyData:
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

        ele_iter = [element_ids] if not isinstance(element_ids, list) else element_ids
        elements_to_iter = self.elements if element_ids is None else ele_iter  # if we assume that all elements must be the same because it's easier.

        n_dim = self.elements[0].ndim
        residual_size = n_dim - 1
        vals = [0, 1]
        combs = list(product(vals, repeat=residual_size))  # the combinations
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
        line_points = np.asarray(
            self.evaluate_embeddings_in_every_element(flat_xis))  # .reshape(n_ele, -1 , 3)[:2].reshape(-1, 3)

        # for ne, e in enumerate(elements_to_iter):
        #     n_dim = e.ndim
        #     residual_size = n_dim - 1
        #     vals = [0, 1]
        #     combs = list(product(vals, repeat=residual_size)) #the combinations
        #     for i in range(n_dim):
        #         d = list(range(n_dim))
        #         d.pop(i)
        #         for comb in combs:
        #             xi_list = [0] * n_dim
        #             for cs, ind in zip(comb, d):
        #                 xi_list[ind] = cs * np.ones(res)
        #             xi_list[i] = np.linspace(0, 1, res)
        #             xis = np.column_stack(xi_list)
        #             comb_pts = self.evaluate_embeddings(np.array([ne]), xis)
        #
        #             l_pts = line_points.shape[0]
        #             line_points = np.concatenate((line_points, comb_pts))
        #             connectivity = np.concatenate((
        #                 connectivity,
        #                 blank_connectivity + [0, l_pts, l_pts],
        #             ))
        mesh = pv.PolyData(
            line_points,
            # lines=connectivity.astype(int)
            lines=long_connectivity.astype(int),
        )
        # mesh.plot(color='k', render_points_as_spheres=True)
        # raise ValueError
        return mesh

    def get_faces(self, rounding_res=10) -> list[tuple[int]]:
        """
        Returns all external faces of the current mesh.
        Faces are indicated as tuples (elem_id, dim, {0,1}).
        By definition, A manifold is a face, indicated as (elem_id, -1, -1).
        Faces are determined by spatial hashing of the face center i.e (0.5,0.5, {0,1})
        """
        hash_space = {}

        elem_eval = np.array([
            [0, 0.5, 0.5], [1, 0.5, 0.5],
            [0.5, 0, 0.5], [0.5, 1, 0.5],
            [0.5, 0.5, 0], [0.5, 0.5, 1],
        ])
        tzip = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
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

        return faces + [k[0] for k in hash_space.values() if len(k) == 1]

    def plot(self, scene: Optional[pv.Plotter] = None, node_colour='r', node_size=10, labels=False, res=10,
             mesh_color='gray', mesh_opacity=0.1, elem_labels=False):
        # evaluate the mesh surface and evaluate all of the elements
        lines = self.get_lines()
        node_dots = np.array([node.loc for node in self.nodes])
        s = pv.Plotter() if scene is None else scene
        s.add_mesh(lines, line_width=2, color='k')
        node_dots_m = pv.PolyData(node_dots)
        # node_dots_m['col'] = np.arange(node_dots.shape[0])
        s.add_mesh(node_dots, render_points_as_spheres=True, color=node_colour, point_size=node_size)

        tri_surf, tris = self.get_triangle_surface(res=res)
        surf_mesh = pv.PolyData(tri_surf)
        surf_mesh.faces = np.concatenate((3 * np.ones((tris.shape[0], 1)), tris), axis=1).astype(int)
        s.add_mesh(surf_mesh, style='wireframe', color=mesh_color, opacity=mesh_opacity)
        if labels:
            s.add_point_labels(points=node_dots, labels=[str(i) for i in range(node_dots.shape[0])])
        if elem_labels:
            elem_locs = np.ones((1, self.elements[0].ndim)) * 0.5
            # breakpoint()
            pts = np.array(self.evaluate_embeddings_in_every_element(elem_locs))
            s.add_point_labels(points=pts, labels=[f"elem: {i}" for i in range(pts.shape[0])])

        if scene is not None:
            return
        s.show()

    def transform(self, htform):
        for node in self.nodes:
            node.loc = h_tform(node.loc, htform, fill=1)
            for k, v in node.items():
                node[k] = htform(v, htform, fill=0)
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

        def evaluate_embeddings(element_ids, xis, fit_params=self.optimisable_param_array, ele_map=self.ele_map):

            param_data = jnp.asarray(self.true_param_array)
            if not len(fit_params) == len(param_data):
                fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            params = jnp.asarray(fit_params)[map]

            p_array = params[jnp.asarray(element_ids).astype(int)]
            outputs = jax.vmap(lambda x: self.elem_evals(x, jnp.asarray(xis)).reshape(-1, 3))
            res = outputs(p_array)
            return res.reshape(-1, 3)

            outputs = [None] * len(element_ids)
            for ide in range(jnp.asarray(element_ids).shape[0]):
                outputs[ide] = self.elem_evals(params[ide], jnp.asarray(xis)).reshape(-1, 3)
            return jnp.concatenate(outputs, axis=0)

        self.evaluate_embeddings = evaluate_embeddings

    def _generate_deriv_function(self):
        """
            Generates the internal functions that evaluate the derivatives of embeddings
            Code is structured so that the result can express custom derivatives
        """

        # @partial(jax.jit, static_argnums=2)
        def evaluate_deriv_embeddings(element_ids, xis, derivs, fit_params=self.optimisable_param_array,
                                      ele_map=self.ele_map):

            param_data = jnp.asarray(self.true_param_array)
            if not len(fit_params) == len(param_data):
                fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids).astype(int)].astype(int)
            params = jnp.asarray(fit_params)[map]
            p_array = params[jnp.asarray(element_ids).astype(int)]

            outputs = jax.vmap(lambda x: self.elem_deriv_evals(x, jnp.asarray(xis), derivs).reshape(-1, 3))
            res = outputs(p_array)
            return res.reshape(-1, 3)

            outputs = [None] * len(element_ids)
            for ide in range(jnp.asarray(element_ids).shape[0]):
                outputs[ide] = self.elem_deriv_evals(params[ide], jnp.asarray(xis), derivs).reshape(-1, 3)
            return jnp.concatenate(outputs, axis=0)

        self.evaluate_deriv_embeddings = evaluate_deriv_embeddings

    ################################# useful utils.

    def embed_points(self, points, verbose=0):

        # generate a KD tree of self
        if self.elements[0].ndim == 2:
            res = 50
            xis = self.xi_grid(res, 2)
            ndim = 2
        else:
            res = 30
            xis = self.xi_grid(res, 3)
            ndim = 3
        tree = KDTree(self.evaluate_embeddings_in_every_element(xis))
        _, i = tree.query(points, k=1, workers=-1)
        elem_num = np.array(i) // xis.shape[0]

        unique_elem, inv = np.unique_inverse(elem_num)
        xi_ind = np.array(i) % xis.shape[0]
        init_xi = xis[xi_ind].flatten()

        def optim_embed(params):
            xi_data = params.reshape(-1, ndim)
            out_data = []
            for ide, e in enumerate(unique_elem):
                mask = ide == inv
                dist = points[mask] - self.evaluate_embeddings([e], xi_data[mask])
                out_data.append(dist.flatten())
            return jnp.concatenate(out_data)

        function, jac = jacobian(optim_embed, init_estimate=init_xi)
        bounds = (np.zeros_like(init_xi), np.ones_like(init_xi))
        result = least_squares(function, init_xi, jac=jac, bounds=bounds, verbose=verbose)

        final_mean_dist = np.mean(np.linalg.norm(result.fun.reshape(-1, ndim), axis=-1))
        if verbose == 2:
            print(f"final mean error of {final_mean_dist:.2f} units")

        return elem_num, result.x.reshape(-1, ndim)

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
        deriv_combos = list(product(*[range(d) for d in n_derivs]))[1:]  # skip the no deriv case
        n_eles = len(self.elements)

        if weights is None:
            weights = np.ones(len(deriv_combos))
        else:
            if not len(weights) == len(deriv_combos):
                raise ValueError(
                    "The length of the provided weights did not match the number of sobolev terms associated with this element")

        out_data = []
        for d, sw in zip(deriv_combos, weights):
            data = self.evaluate_deriv_embeddings_in_every_element(gp, d, fit_params=fit_params)
            weighted = (data.reshape(n_eles, -1, 3) * w[None, :, None]).ravel() * sw
            out_data.append(weighted)

        return jnp.concatenate(out_data)

    ################################# REFINEMENT

    def refine(self, refinement_factor: Optional[int] = None, by_xi_refinement: Optional[tuple[np.ndarray]] = None,
               clean_nodes=True):
        """
            This code refines a mesh, increasing the resolution of every element.
            A refinement factor can be provided, or alternatively a tuple of xi_locations to refine the nodes at can be provided.
        :param refinement_factor: an integer factor to increase the resolution by.
        :param by_xi_refinement: an ndtuple of monotonically increasing points to sample the new nodes at.
        """
        assert not (
                    refinement_factor is not None and by_xi_refinement is not None), "Refinement factor and refining by defined xi are mutually exclusive."

        new_elements = []
        spatial_hash = {tuple(np.round(node.loc, 6).tolist()): idn for idn, node in enumerate(self.nodes)}

        for ide, e in enumerate(self.elements):  # MAKE THE POINTS TO EVAL AT
            intermediate_node_number = np.array(e.n_in_dim) - 2
            if refinement_factor is not None:
                if refinement_factor < 2:
                    raise ValueError("Refining by less than 2 will not change the mesh")
                lr = (intermediate_node_number + 1) * refinement_factor + 1
                if e.ndim == 2:
                    eval_pts = np.column_stack(
                        [x.flatten() for x in np.array(np.mgrid[:lr[0], :lr[1]]) / (lr[:, None, None] - 1)])
                elif e.ndim == 3:
                    eval_pts = np.column_stack([x.flatten() for x in np.array(np.mgrid[:lr[0], :lr[1], :lr[2]]) / (
                                lr[:, None, None, None] - 1)])
                ref_array = np.ones(e.ndim) * refinement_factor
            if by_xi_refinement is not None:
                for b in by_xi_refinement:
                    assert b[0] == 0 and b[-1] == 1, "Provided b arrays must start with 0 and and with 1"
                new_xi_refinement = []
                lr = []
                for idb, b in enumerate(by_xi_refinement):
                    n_points = (len(b) - 1) * (intermediate_node_number[idb] + 1) + 1
                    lr.append(n_points)
                    new_xi_refinement.append(np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(b)), b))
                eval_pts = np.column_stack([x.flatten() for x in np.meshgrid(*by_xi_refinement, indexing='ij')])
                ref_array = np.array([len(by_xi_refinement[i]) for i in [0, 1, 2]]) - 1

            pts = self.evaluate_embeddings(np.array([ide]), eval_pts)
            additional_pts = []
            deriv_bound = np.where([np.any([st[:2] == 'dx' for st in b.weights]) for b in e.basis_functions])[0]

            for d_val in EVAL_PATTERN[len(e.used_node_fields)]:
                # calculate the additional derivatives in the directions that need them
                derivs = [0, 0, 0]
                for dl, di in zip(deriv_bound, d_val):
                    derivs[dl] = di
                d_scale = np.mean(ref_array[np.where(np.array(d_val))])
                additional_pts.append(
                    self.evaluate_deriv_embeddings(np.array([ide]), eval_pts, derivs=derivs) / d_scale)

            # check the generated points against the element hashmap.
            pt_index_array = []
            for idpt, pt in enumerate(pts):
                ind = spatial_hash.get(hashp := tuple(np.round(np.asarray(pt), 6)), None)
                new_vals = {k: np.array(v) for k, v in zip(e.used_node_fields, [a[idpt] for a in additional_pts])}
                if ind is None:
                    node = MeshNode(pt, **new_vals)
                    self.add_node(node)
                    pt_index_array.append(len(self.nodes) - 1)
                    spatial_hash[hashp] = len(self.nodes) - 1
                else:
                    pt_index_array.append(ind)
                    self.nodes[ind].update(new_vals)

            if refinement_factor is not None:
                if e.ndim == 2:
                    n_new = [refinement_factor, refinement_factor]
                elif e.ndim == 3:
                    n_new = [refinement_factor] * 3
            elif by_xi_refinement is not None:
                n_new = [len(b) - 1 for b in by_xi_refinement]

            # lr now defines the total number of points per dimension of this elementlet
            pt_inds = np.array(pt_index_array).reshape(lr)

            for i in range(n_new[0]):
                i_loc = i * (e.n_in_dim[0] - 1)
                for j in range(n_new[1]):
                    j_loc = j * (e.n_in_dim[1] - 1)
                    if e.ndim == 2:
                        points = pt_inds[i_loc:i_loc + e.n_in_dim[0], j_loc:j_loc + e.n_in_dim[1]]

                        elem_id = None
                        if e.id is not None:
                            elem_id = str(e.id) + f"_subelem_{i}_{j}"

                        new_e = MeshElement(node_indexes=points.T.flatten().tolist(), basis_functions=e.basis_functions,
                                            id=elem_id)
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

                        new_e = MeshElement(node_indexes=points.tolist(), basis_functions=e.basis_functions,
                                            BP_inds=e.BasisProductInds, id=elem_id)
                        new_elements.append(new_e)

        self.elements = new_elements
        if clean_nodes:
            self._clean_pts()
        self.generate_mesh()

    def _clean_pts(self):
        """
        Removes nodes unreferenced by all elements, and then reorderers the associated nodes of each element.
        """
        used_points = []
        for element in self.elements:
            if element.used_index:
                used_points.extend(element.nodes)
            else:
                used_points.extend([self.node_id_to_ind[id] for id in element.nodes])

        bool_array = np.zeros(len(self.nodes), dtype=bool)
        bool_array[used_points] = True
        new_inds = np.array([0] + np.cumsum(bool_array).tolist())
        for element in self.elements:
            if element.used_index:
                element.nodes = [new_inds[n] for n in element.nodes]

            if not element.used_index:
                new_node_nums = new_inds[[self.node_id_to_ind[id] for id in element.nodes]]
                node_ids = [self.nodes[n].id for n in new_node_nums]
                element.nodes = node_ids
        self.nodes = [n for idn, n in enumerate(self.nodes) if bool_array[idn]]


def make_eval(basis_funcs: BasisGroup, bp_inds: list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, b_inds=bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])
            w1 = basis_funcs[1].fn(xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(-1, 3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, b_inds=bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(-1, 3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval


def make_deriv_eval(basis_funcs: BasisGroup, bp_inds: list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, d_inds, b_inds=bp_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(-1, 3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, d_inds, b_inds=bp_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            w2 = basis_funcs[2].deriv[d_inds[2]](xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(-1, 3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval


GAUSS = {
    2: [np.array([[0.21132486540518708], [0.78867513459481287]]),
        np.array([0.5, 0.5])],
    3: [np.array([[0.1127016653792583], [0.5], [0.8872983346207417]]),
        np.array([5. / 18., 4. / 9., 5. / 18])],
    4: [np.array([[0.33000947820757187, 0.6699905217924281, 0.06943184420297371, 0.9305681557970262]]).T,
        np.array([0.32607257743127305, 0.32607257743127305, 0.1739274225687269, 0.1739274225687269])],
    5: [np.array([[0.5, 0.230765344947, 0.769234655053, 0.0469100770307, 0.953089922969]]).T,
        np.array([0.284444444444, 0.23931433525, 0.23931433525, 0.118463442528, 0.118463442528])],
    6: [np.array([[0.8306046932331322, 0.1693953067668678, 0.3806904069584016, 0.6193095930415985, 0.0337652428984240,
                   0.9662347571015760]]).T,
        np.array([0.1803807865240693, 0.1803807865240693, 0.2339569672863455, 0.2339569672863455, 0.0856622461895852,
                  0.0856622461895852])],
}