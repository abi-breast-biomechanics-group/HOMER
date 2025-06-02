from typing import Optional, Callable
import numpy as np
import jax.numpy as jnp
from basis_definitions import N2_weights










class mesh:
    def __init__(self, elem_spec) -> None:
        #what does the mesh need.

        self.elem_function_spec: list[tuple[Callable]] | tuple[Callable] = elem_spec #can be one, or on a per element basis
        self.nodes: Optional[np.ndarray] = None
        self.elements: Optional[np.ndarray] = None

        self.eval_xi: Optional[Callable] = None

################################## MAIN FUNCTIONS
    def evaluate_embeddings(self, element_ids: np.ndarray, xis: np.ndarray) -> np.ndarray:
        return

    def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray) -> np.ndarray:
        return


################################## MORPHIC INTERFACE COMPATIBILITY
    def generate_mesh(self) -> None:
        return

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

    def load(self) -> mesh:
        return mesh()

################################## INTERNAL

    def _generate_elem_functions(self):
        """
            Creates the internal function evaluation structure.
        """
        if isinstance(self.elem_function_spec, list):
            self.elem_evals = [make_eval(fspec) for fspec in self.elem_function_spec]
        else 
            self.elem_evals = make_eval(self.elem_function_spec)

     

def make_eval(basis_funcs: tuple[Callable, Callable] | tuple[Callable, Callable, Callable]):
    """
        Returns a jax compliant function which evaluates a single element from a 
    """
    if len(basis_funcs) == 2:
        def xi_eval(elem_params, xis):
            w0 = basis_funcs[0](xis[:, 0])  
            w1 = basis_funcs[1](xis[:, 1])
            weights = N2_weights(w0, w1)
            output = jnp.sum(elem_params * weights[:, None, :], axis=-1).flatten()
            return output
    elif len(basis_funcs) == 3:
        def xi_eval(elem_params, xis):
            w0 = basis_funcs[0](xis[:, 0])  
            w1 = basis_funcs[1](xis[:, 1])
            w2 = basis_funcs[1](xis[:, 2])
            weights = N3_weights(w0, w1, w2)
            output = jnp.sum(elem_params * weights[:, None, :], axis=-1).flatten()
            return output
    else:
        raise ValueError("Currently, meshes must be 2D or 3D")
    return xi_eval
