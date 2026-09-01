from HOMER import cube, L2Basis
from HOMER.utils import rodrigues_exp
import jax.numpy as jnp
from scipy.optimize import least_squares
import numpy as np
import jax
from scipy.sparse.linalg import LinearOperator

base_mesh = cube(basis=[L2Basis]*3)
surface = base_mesh.eval_surface(res=10)

# base_mesh.refine(2)
init_struct = base_mesh.optimisable_param_array

tform = rodrigues_exp(jnp.array([10,145,65]))
surface = surface @tform.T + jnp.array([1,-1,4])
# surface = surface[0]


def model_track(params, plot=False):
    R0 = rodrigues_exp(params[:3])
    # R0 = jnp.eye(3)
    T0 = params[3:]
    new_params = init_struct.reshape(-1, 3) @ R0.T + T0
    _, residual = base_mesh.embed_points(surface, fit_params=new_params.ravel(), return_residual=True, verbose=3 if plot else 0, surface_embed=True, grid_res=50, iterations=10, robust_init_est=False)
    return residual.ravel()
fmodel_track = jax.jit(model_track)


init_params = jnp.zeros(6)


deriv = jax.jacfwd(model_track)(init_params)
# breakpoint()

# a comparison, but this gets slower and slower as the models get bigger.
brute_jac = jax.jacfwd(model_track)

##################################
##################################
"""
The following code demonstrates how you can use vmapping over a single element gather operation to solve an ICP
problem. 
Because there are a known set of parameters effecting each element, vmapping over the operation in this way removes the need to express
the change in every data point with respect to the parameters.
For large meshes with simplified parameterisations (PCA, physics) this can make optimsiation tractable.
"""

@jax.jacfwd ############################################ use jax to skip a bunch of questions
def jac_model_nodes_given_tform(params):
    R0 = rodrigues_exp(params[:3])
    T0 = params[3:]
    new_params = init_struct.reshape(-1, 3) @ R0.T + T0
    return new_params

phantom_l3 = cube(basis=[L2Basis]*3) #this little cube defines all necessary operations.

@jax.jit
def single_ele_embed_jvp(point, params, local_tangent):
    local_e = lambda p: phantom_l3.embed_points(point, fit_params=p, return_residual=True)[1]
    local_jvp = lambda t_p: jax.jvp(local_e, (params,), (t_p,))[1]
    batched_jvp = jax.vmap(local_jvp, in_axes=1, out_axes=-1)
    res = batched_jvp(local_tangent)
    return res
    # return res[0] if res.ndim == 3 else res

emap = jnp.array(base_mesh.ele_map)

@jax.jit
def model_track_jac(params, plot=False):
    p_len = params.shape[0]
    R0 = rodrigues_exp(params[:3])
    T0 = params[3:]
    new_params = init_struct.reshape(-1, 3) @ R0.T + T0
    embed_locs, _ = base_mesh.embed_points(surface, fit_params=new_params.ravel(), return_residual=True, verbose=3 if plot else 0, surface_embed=True, grid_res=50, iterations=50)

    jac_mp0 =  jac_model_nodes_given_tform(params)

    mesh_primal = new_params.ravel()[emap[embed_locs[0]].astype(int)]
    mesh_tangent = jac_mp0.reshape(-1, p_len)[emap[embed_locs[0]].astype(int)]
    ptcd_res0_tangent = jax.vmap(single_ele_embed_jvp)(surface, mesh_primal, mesh_tangent)

    out_array = jnp.concatenate([
        ptcd_res0_tangent.reshape(-1, p_len),
    ])
    return out_array

fmodel_track_jac = jax.jit(model_track_jac)

res = least_squares(fmodel_track, init_params, verbose=2, max_nfev=100, 
                    jac=fmodel_track_jac,
                    # jac=brute_jac,
                    )
model_track(res.x, plot=True)
