from functools import partial
from jax import jacfwd
from HOMER import cube
import jax.numpy as jnp
import jax
import numpy as np

from HOMER.jacobian_evaluator import jacobian
from scipy.optimize import least_squares
import pyvista as pv



# test the different ways that this function can be used and optimised.


# move points to minimise residual

# move points so that they have specific xi locations + force them to be close to said location
if False:
    cube_mesh = cube(centre=np.ones(3)*.5)
    grid = np.mgrid[:5,:5,:5]
    target_xi = np.array([a.flatten() for a in grid]).T/4
    def pts_to_xi(params): #maybe some kind of bug for points outside of the mesh on this embedding.
        points = params.reshape(-1,3)
        (_, xis), res = cube_mesh.embed_points(points, return_residual=True)
        return jnp.concatenate(((xis - target_xi).flatten(), res.flatten()))

    init_params = np.ones_like(target_xi).ravel()/2
    fitting_function, jacobian_fun = jacobian(pts_to_xi, init_estimate=init_params)
    res = least_squares(fitting_function, x0=init_params, jac=jacobian_fun, verbose=2)

    new_points = np.array(res.x.reshape(-1, 3))
    s = pv.Plotter()
    cube_mesh.plot(s)
    s.add_mesh(new_points, render_points_as_spheres=True, point_size=15, color='b')
    s.add_mesh(target_xi, render_points_as_spheres=True, point_size=10, color='g')
    s.show()

# move params to minimise residual #tested in the point to plane fit.

# move params so p that points they have specific xi locations
# note this is just the linear solve with waaaaaay more steps, but we can test it in the same way.

cube_0 = cube()
cube_1 = cube(
    scale=2,
    centre=np.ones(3)*1.5,
)

xi_grid = cube_0.xi_grid(4)
points = cube_0.evaluate_embeddings([0], xi_grid)
cube_1.embed_points(points, verbose=3)

def params_to_xi(params): #params to xi is borked.
    _, xis = cube_1.embed_points(points, fit_params=params)
    return (xis - xi_grid).flatten()

cube_1.embed_points(points, verbose=3)

init_params = cube_1.optimisable_param_array
fitting_function, jacobian_fun = jacobian(params_to_xi, init_estimate=init_params)
res = least_squares(fitting_function, x0=init_params, jac=jacobian_fun, verbose=2)

cube_1.update_from_params(res.x)
cube_1.embed_points(points, verbose=3)







# tval = jnp.concatenate((init_params, locs.flatten()))
# test(tval)


