import numpy as np
import jax.numpy as jnp

from HOMER.mesher import Mesh
from HOMER.optim import jax_comp_kdtree_distance_query
from HOMER.jacobian_evaluator import jacobian

from matplotlib import pyplot as plt

def point_cloud_fit(mesh:Mesh, data, res=20, compile=True):
    """
        An example that creates a fitting problem for a mesh.
    """

    data_tree = jax_comp_kdtree_distance_query(data, kdtree_args={"workers":-1})
    eval_points = mesh.xi_grid(res)
    sob_points = mesh.gauss_grid([4, 4])

    mesh_elements = np.arange(len(mesh.elements))
    mesh.compile = compile #force the mesh to be compiled because we are running it a lot of times.
    mesh.generate_mesh()

    #get the initial params
    # create the handler object.

    def fitting_function(params):
        outputs = []
        wpts = mesh.evaluate_embeddings(mesh_elements, eval_points, params[:])
        dists = data_tree(wpts)
        outputs.append(dists.flatten())
        outputs = jnp.concatenate(outputs)
        return outputs

    fitting_function, jacobian_fun = jacobian(fitting_function, init_estimate=mesh.optimisable_param_array)


    # plt.imshow(jacobian_fun(mesh.optimisable_param_array).todense())
    # plt.show()
    return fitting_function, jacobian_fun






