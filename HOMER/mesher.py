from typing import Optional, Callable
import numpy as np
import jax.numpy as jnp
from functools import reduce
from itertools import groupby

from HOMER.basis_definitions import N2_weights, N3_weights, AbstractBasis, BasisGroup, DERIV_ORDER



class mesh_node(dict):
    def __init__(self, loc, **kwargs):
        """
        The base node class, handling arbitrary properties over the mesh surface.
        """
        self.loc = loc
        self.update(kwargs)

        for key, value in self.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Only np.ndarray are valid additional data, but found key: {key}, value: {value} pair")



class mesh_element:
    def __init__(self, nodes: list[int], basis_functions: BasisGroup):
        """
            A high order mesh element. This element is constructed from 

        """
        self.nodes = nodes
        self.basis_functions = basis_functions
        self.get_used_fields()
        self._calc_basis_product_inds()

    def get_used_fields(self):
        """
        Calculates the used node fields for field objects.
        This represents the increasing derivative pattern du -> du, dw, dudw -> du ... dudvdw
        """
        raw_fields = [b.node_fields for b in self.basis_functions if b.node_fields is not None]
        sorted_objects = sorted(raw_fields, key=lambda x: x.__class__.__name__)
        grouped = [list(group) for _, group in groupby(sorted_objects, key=lambda x: x.__class__)]
        fields = reduce(lambda x,y:x+y,[f.get_needed_fields() for f in [reduce(lambda x,y: x+y, g) for g in grouped]])
        self.used_node_fields = [fields] if isinstance(fields, str) else fields

    def _calc_basis_product_inds(self):
        """
        Given the definition of the basis functions, this creates the indexes used to populate the weighting matrix.
        The weighting matrix is defined as the outer product of the basis functions for each element.
        :params b_def: the definition of the parameters associated with the basis functions.
        """
        n_in_dim = [sum([l[0]=='x' for l in b.weights]) for b in self.basis_functions]
        dim_step =  [1] + np.cumprod(n_in_dim)[:-1].tolist()
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

        self.BasisProductInds = [tuple(l_mat[i].tolist()) for i in self.argsort_derivs(keyvals, DERIV_ORDER)]

    def argsort_derivs(self, derivs_struct: list[list[str]], order_dict: dict[tuple]):
        """
        Given a derivs struct defined iternally, returns the canonical ordering according to a given order dict.
        """
        indexed_keys = [
            (i, (abs(lst[0]), order_dict[tuple(lst[1:])] if len(lst) > 1 else 0))
            for i, lst in enumerate(derivs_struct)
        ]
        indexed_keys.sort(key=lambda x: x[1])
        return [i for i, _ in indexed_keys]





class mesh:
    def __init__(self, nodes:Optional[list[mesh_node]] = None, elements: Optional[list[mesh_element]|mesh_element]=None) -> None:
        
        ######### topology of the mesh
        self.nodes: list[mesh_node] = [] if nodes is None else (nodes if isinstance(nodes, list) else [nodes])
        self.elements: list[mesh_element] = [] if elements is None else (elements if isinstance(elements, list) else [elements])

        ######### initialising values to be calculated
        self.elem_evals: Optional[Callable] = None
        self.fit_param: Optional[np.ndarray] = None

        ######### optimisation
        self.param_array: Optional[np.ndarray] = None
        self.ele_map: Optional[np.ndarray] = None
        if not len(self.nodes) == 0 and not len(self.elements) == 0:
            self.generate_mesh()

    ################################## MAIN FUNCTIONS
    def evaluate_embeddings(self, element_ids: np.ndarray|list[int]|tuple[int], xis: np.ndarray) -> np.ndarray:
        """
        Evaluates embeddings over the elements of the mesh.
        
        :param element_ids: an iterable containing the int locations to evaluate the xis locations at.
        :param xis: the xi locations (within each element) to evaluate points at.
        :returns outputs: the world location of the evaluated embeddings.
        """
        outputs = [None] * len(element_ids)
        for ele in element_ids:
            params = self.get_element_params(ele)
            data = self.elem_evals[ele](params, xis)
            outputs[ele] = data.reshape(-1,3)
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

    def fitting_params(self, params):
        """
            Stores a param set to evaluate within the mesh
        """
        self.fit_param = params

    ################################## CONVENIENCE
    def xi_grid(self, res: int, dim=2) -> np.ndarray:
        if dim == 2:
            X,Y = np.mgrid[:res+1, :res+1]/res
            return np.column_stack((X.flatten(), Y.flatten()))
        else:
            X,Y,Z = np.mgrid[:res+1, :res+1, :res+1]/res
            return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    def get_element_params(self, ele_num: int) -> np.ndarray:
        """
        returns the flat vector of node parameters associated with this element.
        """
        return self.param_array[self.ele_map[ele_num].astype(int)]

    def update_from_params(self, params):
        """
            Updates all nodes with data from an input param array.
        """
        for node in self.nodes:
            node.loc, params = params[:3], params[3:]
            for key, value in node.items():
                l_val = value.flatten().shape[0]
                flat_node = node[key].ravel()
                flat_node[:], params = params[:l_val], params[l_val:] 

    ################################## MORPHIC INTERFACE COMPATIBILITY
    def generate_mesh(self) -> None:
        """
        Builds the mesh representation on call.

        This code is responsible for handling on-the-fly functions, and the generation of the
        'fast' pathway numpy array representation.

        """

        self.param_array = np.concatenate([np.concatenate([node.loc] + [d.flatten() for d in node.values()]) for node in self.nodes]).copy()

        self.update_from_params(np.arange(self.param_array.shape[-1]))
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
        self.update_from_params(self.param_array)

        self._generate_elem_functions()

    def add_node(self, node:mesh_node) -> None:
        self.nodes.append(node)
        # self.generate_mesh()

    def add_element(self, element:mesh_element) -> None:
        self.elements.append(element)
        # self.generate_mesh()


    def get_element(self, element_ids: np.ndarray | int) -> np.ndarray:
        return

    def get_node(self, element_ids: np.ndarray | int) -> np.ndarray:
        return

    ################################## PLOTTING
    def get_surface(self, element_ids: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return 

    def get_lines(self, element_ids: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return 

    def get_faces(self, element_ids: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return

    def plot(self):
        return

    ################################## IO
    def save(self):
        return

    def load(self) -> "mesh":
        return mesh()

    ################################## INTERNAL

    def _generate_elem_functions(self):
        """
            Creates the internal function evaluation structure.
        """
        self.elem_evals = [make_eval(elem.basis_functions, elem.BasisProductInds) for elem in self.elements]


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
