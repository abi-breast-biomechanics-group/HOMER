from typing import Optional, Callable
import numpy as np
import jax.numpy as jnp
import jax
import pyvista as pv
from matplotlib import pyplot as plt

from functools import reduce, partial
from itertools import groupby, combinations_with_replacement, product

from HOMER.basis_definitions import N2_weights, N3_weights, AbstractBasis, BasisGroup, DERIV_ORDER, EVAL_PATTERN



class mesh_node(dict):
    def __init__(self, loc, **kwargs):
        """
        The base node class, handling arbitrary properties over the mesh surface.
        """
        self.loc = np.array(loc)
        self.update(kwargs)
        self.fixed_params = set()

        for key, value in self.items():
            if isinstance(value, list):
                self[key] = np.array(value)
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"Only np.ndarray are valid additional data, but found key: {key}, value: {value} pair")

    def fix_parameter(self, param_names: list | str, values: Optional[list[np.ndarray]|np.ndarray]=None) -> None:
        if isinstance(param_names, str):
            param_names = [param_names]
        if not isinstance(values, list):
            values = [values] * len(param_names)
        for idp, param in enumerate(param_names):
            self.fixed_params.add(param)
            if values[idp] is not None:
                if param == 'loc':
                    self.loc = values[idp]
                else:
                    self[param] = values[idp]

    def get_optimisability_arr(self):
        list_data = [np.ones(3) * (not "loc" in self.fixed_params)]
        for key in self.keys():
            list_data.append(np.ones(3) * (not key in self.fixed_params))
        return np.concatenate(list_data, axis=0)


    def plot(self, scene: Optional[pv.Plotter] = None) -> pv.Plotter | None:
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



class mesh_element:
    def __init__(self, nodes: list[int], basis_functions: BasisGroup, BP_inds: Optional = None):
        """
            A high order mesh element. This element is constructed from 

        """
        self.nodes = nodes
        self.basis_functions = basis_functions
        self.ndim: int = len(self.basis_functions)
        self.n_in_dim = [sum([l[0]=='x' for l in b.weights]) for b in self.basis_functions]

        self.get_used_fields()
        self.BasisProductInds = self._calc_basis_product_inds() if BP_inds is None else BP_inds


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
        """
        indexed_keys = [
            (i, (abs(lst[0]), (order_dict[tuple(lst[1:])] if len(lst) > 1 else 0)))
            for i, lst in enumerate(derivs_struct)
        ]
        
        indexed_keys.sort(key=lambda x: x[1])
        return [i for i,  _ in indexed_keys]





class mesh:
    def __init__(self, nodes:Optional[list[mesh_node]] = None, elements: Optional[list[mesh_element]|mesh_element]=None, jax_compile:bool = False) -> None:
        
        ######### topology of the mesh
        self.nodes: list[mesh_node] = [] if nodes is None else (nodes if isinstance(nodes, list) else [nodes])
        self.elements: list[mesh_element] = [] if elements is None else (elements if isinstance(elements, list) else [elements])

        ######### initialising values to be calculated
        self.elem_evals: Optional[Callable] = None
        self.evaluate_embeddings = lambda element_ids, xis, params: None #def the function signature here
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
    def evaluate_deriv_embeddings(self, element_ids: np.ndarray|list[int]|tuple[int], xis: np.ndarray, deriv: list[int]) -> np.ndarray:
        """
        Evaluates embeddings over the elements of the mesh.
        
        :param element_ids: an iterable containing the int locations to evaluate the xis locations at.
        :param xis: the xi locations (within each element) to evaluate points at.
        :returns outputs: the world location of the evaluated embeddings.
        """
        if not (isinstance(element_ids, np.ndarray) or isinstance(element_ids, list)):
            element_ids = [element_ids]

        outputs = [None] * len(element_ids)
        for ide, ele in enumerate(element_ids):
            params = self.get_element_params(ele)
            data = self.elem_deriv_evals(params, xis, deriv)
            outputs[ide] = data.reshape(-1,3)
        return np.concatenate(outputs, axis=0)


    def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray) -> np.ndarray:
        """
        Returns the normal at the element surface.
        Only valid for manifold meshes.
        :params elemend_ids: The elements to evaluate
        :params xis: the locations to evaluate.
        :returns normals: unit vector directions associated with the mesh surface.
        """
        return


    ################################## CONVENIENCE
    def xi_grid(self, res: int, dim=2, surface=False) -> np.ndarray:
        if dim == 2:
            X,Y = np.mgrid[:res, :res]/(res - 1)
            return np.column_stack((X.flatten(), Y.flatten()))
        else:
            if not surface:
                X,Y,Z = np.mgrid[:res, :res, :res]/(res - 1)
                return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            else:
                raw_x = np.array([x.flatten() for x in np.mgrid[:res, :res]/(res-1)])
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
                gindex = np.array([gindex[n].flatten() for n in [2,1,0]]).T
                Xi = np.array([
                    Xi1[gindex[:, 0]], Xi2[gindex[:, 1]], Xi3[gindex[:, 2]]])[:, :, 0].T
                W = np.array([
                    W1[gindex[:, 0]], W2[gindex[:, 1]], W3[gindex[:, 2, ]]]).T.prod(1)
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
            raise ValueError("Input param array was provided that did not match either that set of parameters, or the optimisable subset of parameters")

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

        self.true_param_array = np.concatenate([np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()
        self.optimisable_param_bool = np.concatenate([node.get_optimisability_arr() for node in self.nodes], axis=0).astype(bool)
        self.optimisable_param_array = self.true_param_array[self.optimisable_param_bool]

        self.update_from_params(np.arange(self.true_param_array.shape[-1]), generate=False)



        ele_maps = []
        for ide, element in enumerate(self.elements):
            param_ids = []
            for idn, node in enumerate([self.nodes[e] for e in element.nodes]):
                param_ids.append(node.loc)
                for field in element.used_node_fields: 
                    try:
                        param_ids.append(node[field].flatten())
                    except KeyError:
                        raise ValueError(f"Provided node id: {idn} did not have the field '{field}' which is required by element: {ide}")
            ele_maps.append(np.concatenate(param_ids))
        self.ele_map = np.array(ele_maps)
        self.update_from_params(self.true_param_array, generate=False)

        self._generate_elem_functions()
        self._generate_elem_deriv_functions()
        self._generate_eval_function()

    def add_node(self, node:mesh_node) -> None:
        self.nodes.append(node)
        # self.generate_mesh()

    def add_element(self, element:mesh_element) -> None:
        self.elements.append(element)
        self.generate_mesh()

    def get_element(self, element_ids: np.ndarray | int) -> list[mesh_element]:
        return [self.elements[id] for id in ([element_ids] if isinstance(element_ids, int) else element_ids)]

    def get_node(self, node_ids: np.ndarray | int) -> list[mesh_node]:
        return [self.nodes[id] for id in ([node_ids] if isinstance(node_ids, int) else node_ids)]

    ################################## PLOTTING
    def get_surface(self, element_ids: Optional[np.ndarray] = None, res:int = 20, just_faces=False) -> np.ndarray:
        """
        Returns a set of points evaluated over the mesh surface.
        """
        ele_iter  = [element_ids] if not isinstance(element_ids, list) else element_ids
        elements_to_iter = self.elements if element_ids is None else ele_iter
        if not just_faces:
            all_points = []
            for ne, e in enumerate(elements_to_iter):
                grid = self.xi_grid(res=res, ndim=e.ndim, surface=True)
                all_points.append(self.evaluate_embeddings(np.array([ne]), grid))
            return np.concatenate(all_points, axis=0) 
        else:
            face_pts = []
            faces = self.get_faces()
            for face in faces:
                xi2grid = self.xi_grid(res=res, dim=2)
                xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)

                element = self.elements[face[0]]
                if element.ndim == 2:
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]), xi2grid))
                elif element.ndim ==3:
                    grid_def = xi3grid[face[1], face[2]]
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]),grid_def))
            return np.concatenate(face_pts, axis=0)


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
        elements_to_iter = self.elements if element_ids is None else ele_iter
        for ne, e in enumerate(elements_to_iter):
            n_dim = e.ndim
            residual_size = n_dim - 1 
            vals = [0, 1]
            combs = list(product(vals, repeat=residual_size)) #the combinations 
            for i in range(n_dim):
                d = list(range(n_dim))
                d.pop(i)
                for comb in combs:
                    xi_list = [0] * n_dim
                    for cs, ind in zip(comb, d):
                        xi_list[ind] = cs * np.ones(res)
                    xi_list[i] = np.linspace(0, 1, res)
                    xis = np.column_stack(xi_list)
                    comb_pts = self.evaluate_embeddings(np.array([ne]), xis)

                    l_pts = line_points.shape[0]
                    line_points = np.concatenate((line_points, comb_pts))
                    connectivity = np.concatenate((
                        connectivity,
                        blank_connectivity + [0, l_pts, l_pts],
                    ))
        mesh = pv.PolyData(line_points, lines=connectivity.astype(int))
        return mesh

    def get_faces(self, rounding_res = 10) -> list[tuple[int]]:
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

        return faces + [k[0] for k in hash_space.values() if len(k) == 1]

    def plot(self, scene:Optional[pv.Plotter] = None, node_colour='r', node_size=10, labels = False):
        #evaluate the mesh surface and evaluate all of the elements
        lines = self.get_lines()
        node_dots = np.array([node.loc for node in self.nodes])
        s=pv.Plotter() if scene is None else scene
        s.add_mesh(lines, line_width=2, color='k')
        node_dots_m = pv.PolyData(node_dots)
        # node_dots_m['col'] = np.arange(node_dots.shape[0])
        s.add_mesh(node_dots, render_points_as_spheres=True, color=node_colour, point_size=node_size)

        tri_surf, tris = self.get_triangle_surface()
        surf_mesh = pv.PolyData(tri_surf)
        surf_mesh.faces = np.concatenate((3 * np.ones((tris.shape[0], 1)), tris), axis=1).astype(int)
        s.add_mesh(surf_mesh, style='wireframe', color='gray', opacity=0.1)
        if labels:
            s.add_point_labels(points = node_dots, labels=[str(i) for i in range(node_dots.shape[0])])
        if scene is not None:
            return
        s.show()

    def transform(self, htform):
        for node in self.nodes:
            node.loc = h_tform(node.loc, htform, fill=1)
            for k,v in node.items():  
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
        
        def evaluate_embeddings(element_ids, xis, fit_params = self.optimisable_param_array, ele_map= self.ele_map):

            param_data = jnp.asarray(self.true_param_array)
            fit_params = param_data.at[self.optimisable_param_bool].set(fit_params)

            map = jnp.asarray(ele_map)[jnp.asarray(element_ids)].astype(int)
            params = jnp.asarray(fit_params)[map]
            outputs = [None] * len(element_ids)
            for ide in range(jnp.asarray(element_ids).shape[0]):
                outputs[ide] = self.elem_evals(params[ide], jnp.asarray(xis)).reshape(-1, 3)
            return jnp.concatenate(outputs, axis=0)

        # def deriv_xis(element_ids, xis, fit_params=self.param_array, ele_map = self.ele_map):
        #     map = jnp.asarray(ele_map)[jnp.asarray(element_ids)].astype(int)
        #     params = jnp.asarray(fit_params)[map]
        #     outputs = [None] * len(element_ids)
        #     for ide, ele in enumerate(element_ids):
        #         outputs[ide] = self.elem_xi_deriv(params[ide], jnp.asarray(xis))
        #     return jnp.concatenate(outputs, axis=0)
        #
        # def deriv_params(element_ids, xis, fit_params=self.param_array, ele_map=self.ele_map):
        #     map = jnp.asarray(ele_map)[jnp.asarray(element_ids)].astype(int)
        #     params = jnp.asarray(fit_params)[map]
        #     outputs = [None] * len(element_ids)
        #     for ide, ele in enumerate(element_ids):
        #         outputs[ide] = self.elem_param_deriv(params[ide], jnp.asarray(xis))
        #     return jnp.concatenate(outputs, axis=0)
        #
        # def not_impl(val):
        #     raise ValueError("This value isn't continuous")
        #
        # evaluate_embeddings.custom_deriv([not_impl, deriv_xis, deriv_params, not_impl])
        
        self.evaluate_embeddings = evaluate_embeddings



    ################################# REFINEMENT

    def refine(self, refinement_factor: Optional[int]=None, by_xi_refinement: Optional[tuple[np.ndarray]] =  None):
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
                ref_array = np.array([len(by_xi_refinement[i]) for i in [0,1,2]]) - 1
        
            pts = self.evaluate_embeddings(np.array([ide]), eval_pts)
            additional_pts = []
            deriv_bound = np.where([np.any([st[:2] == 'dx' for st in b.weights]) for b in e.basis_functions] )[0]
            
            for d_val in EVAL_PATTERN[len(e.used_node_fields)]:
                #calculate the additional derivatives in the directions that need them
                derivs = [0,0,0]
                for dl, di in zip(deriv_bound, d_val): 
                    derivs[dl] = di
                d_scale = np.mean(ref_array[np.where(np.array(d_val))])
                additional_pts.append(self.evaluate_deriv_embeddings(np.array([ide]), eval_pts, deriv=derivs)/d_scale)

            #check the generated points against the element hashmap.
            pt_index_array = [] 
            for idpt, pt in enumerate(pts):
                ind = spatial_hash.get(hashp:=tuple(np.round(np.asarray(pt), 6)), None) 
                new_vals = {k:v for k, v in zip(e.used_node_fields, [a[idpt] for a in additional_pts])}
                if ind is None:
                    node = mesh_node(pt, **new_vals)
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
                        # breakpoint()
                        new_e = mesh_element(points.T.flatten().tolist(), basis_functions=e.basis_functions)
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

                        new_e = mesh_element(nodes=points.tolist(), basis_functions=e.basis_functions, BP_inds=e.BasisProductInds)
                        new_elements.append(new_e)
          

        self.elements = new_elements
        self._clean_pts()
        self.generate_mesh()
            

    def _clean_pts(self):
        """
        Removes nodes unreferenced by all elements, and then reorderers the associated nodes of each element.
        """
        used_points = []
        for element in self.elements:
            used_points.extend(element.nodes)
        
        bool_array = np.zeros(len(self.nodes), dtype=bool)
        bool_array[used_points] = True
        new_inds = np.array([0] + np.cumsum(bool_array).tolist())
        for element in self.elements:
            element.nodes = [new_inds[n] for n in element.nodes]
        self.nodes = [n for idn, n in enumerate(self.nodes) if bool_array[idn]]




        



def make_eval(basis_funcs: BasisGroup, bp_inds:list[tuple[int]]):
    """
        Returns a jax compliant function which evaluates a single element from a 
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            weights = N2_weights(w0, w1, b_inds)
            output = jnp.sum(elem_params.reshape(-1,3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, b_inds = bp_inds):
            w0 = basis_funcs[0].fn(xis[:, 0])  
            w1 = basis_funcs[1].fn(xis[:, 1])
            w2 = basis_funcs[2].fn(xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(-1,3)[:, None] * weights[..., None], axis=0).flatten()
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
            output = jnp.sum(elem_params.reshape(-1,3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis, d_inds, b_inds = bp_inds):
            w0 = basis_funcs[0].deriv[d_inds[0]](xis[:, 0])  
            w1 = basis_funcs[1].deriv[d_inds[1]](xis[:, 1])
            w2 = basis_funcs[2].deriv[d_inds[2]](xis[:, 2])
            weights = N3_weights(w0, w1, w2, b_inds)
            output = jnp.sum(elem_params.reshape(-1,3)[:, None] * weights[..., None], axis=0).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval

GAUSS = { 
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

