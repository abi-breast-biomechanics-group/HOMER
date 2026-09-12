# Mesh Fitting and Optimisation

HOMER offers two main pre-built fitting strategies:

1. **Linear least-squares** (`get_xi_weight_mat` + `linear_fit`) – fast when
   the parametric embeddings are fixed.
2. **Nonlinear optimisation** (`point_cloud_fit` + `scipy.optimize`) – for
   free-form shape optimisation against a point cloud.

However, as HOMER uses JAX to implement the mesh representations, it allows very expressive manipulations of mesh geometry.
Importantly, this still provides the ability to automatically define Jacobians on these manipulations.

---

## Linear Fitting

Use this when you can compute (or embed) the xi coordinates in advance and
only need to update nodal values.

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3

# 1. Assume you have a mesh and some target points
mesh = ...   # your MeshField or Mesh

res = 5
xis = mesh.xi_grid(res)
# Repeat the grid for each of 4 elements
xis_tiled = np.tile(xis, (4, 1))
elem_ids = np.repeat(np.arange(4), res**2)

# 2. Sample target geometry at those (elem, xi) locations
target_pts = target_mesh.evaluate_embeddings_ele_xi_pair(elem_ids, xis_tiled)

# 3. Build the weight matrix
W = mesh.get_xi_weight_mat(elem_ids, xis_tiled)

# 4. Solve the linear system in-place
mesh.linear_fit(target_pts, weight_mat=W)
```

!!! note
    `linear_fit` needs at least as many points as the weight matrix has
    columns (`n_pts >= weight_mat.shape[1]`).  The columns are *parameters*,
    not nodes: a Hermite node carries four parameter blocks in 2-D and eight
    in 3-D, so a 4-node H3 patch already needs 16 points.  Increase `res` if
    you see an assertion error.

!!! warning "linear_fit ignores fixed parameters"
    Parameters fixed with `MeshNode.fix_parameter()` are **not** respected by
    `linear_fit`: it solves for every parameter and overwrites the constrained
    ones.  This is deliberate and pinned by
    `test_linear_fit_does_not_respect_fixed_parameters`.  When you need
    constraints honoured, use the nonlinear pathway below, which optimises
    `optimisable_param_array` and leaves fixed parameters alone.

---

## Nonlinear Point-Cloud Fitting

Use `point_cloud_fit` when you want to optimise node positions (and optionally
derivative vectors) to best match an unstructured point cloud.

```python
from HOMER.fitting import point_cloud_fit
from scipy.optimize import least_squares

# 1. Optionally fix some nodes to prevent the mesh from drifting
mesh.get_node('corner').fix_parameter('loc')

# 2. Build the cost function and Jacobian
fitting_fn, jac_fn = point_cloud_fit(
    mesh, target_pts, compile=True, sob_weight=0.01
)

# 3. Run the optimiser
result = least_squares(
    fitting_fn,
    mesh.optimisable_param_array.copy(),
    jac=jac_fn,
    verbose=2,
)

# 4. Apply the optimised parameters
mesh.update_from_params(result.x)
```

### Sobolev Regularisation

The `sob_weight` parameter adds a Sobolev smoothness term that penalises
high curvature in the mesh surface.  Increase it if the mesh develops
wrinkles:

```python
fitting_fn, jac_fn = point_cloud_fit(mesh, pts, sob_weight=0.1)
```

### Fitting with Surface Normals

When surface normals are available, pass them to project the residuals along
the normal direction:

```python
fitting_fn, jac_fn = point_cloud_fit(mesh, pts, normals=normal_vectors)
```

---

## Fits Too Large to Hold a Jacobian

`point_cloud_fit` and `jacobian` both build the Jacobian as a matrix, with a
sparsity colouring to keep it affordable. That stops working on two kinds of
problem: one where the matrix itself is too big (a residual of ~1e5 entries in
~1e4 parameters costs gigabytes), and one where the blocks that matter are
dense, so the colouring saves nothing — every dof supporting a sample excites
the same rows, which is what a field fitted over embedded points looks like.

`matrix_free_jacobian` hands SciPy a `LinearOperator` over JAX's `jvp`/`vjp`
instead. Nothing larger than a residual vector is ever allocated, and
`tr_solver='lsmr'` — the one SciPy trust-region solver that accepts an
operator — solves each subproblem from matrix-vector products alone.

```python
from HOMER import matrix_free_jacobian
from scipy.optimize import least_squares

def cost(params):
    D = W @ params.reshape(n_dof, 6)      # a field evaluated at the samples
    return jnp.ravel(render(D) - observed)

fwd, jac, scale = matrix_free_jacobian(cost, p_start)

result = least_squares(fwd, p_start / scale, jac=jac, tr_solver='lsmr',
                       tr_options=dict(maxiter=50))
params = result.x * scale
```

The `scale` is the point of the divide-and-multiply. It holds the reciprocal
column norms of the Jacobian, estimated at `p_start` with a handful of random
probes (`||J_j||^2 = E[(J^T z)_j^2]`, so one `vjp` prices every column at
once), and the returned operator has that scaling folded in — the same
equilibration `column_equilibrated_lstsq` applies to an assembled matrix, on a
matrix you never assemble.

!!! warning "The scaling is not optional"
    A Hermite basis carries derivative dofs whose columns are orders of
    magnitude shorter than the value dofs'. On the raw operator lsmr never
    solves the trust-region subproblem well enough to take a Newton-sized step,
    and the fit stalls. SciPy's own `x_scale='jac'` cannot stand in for it:
    that squares the Jacobian elementwise, so it needs a real matrix.

Pass `precondition=False` to see what the scaling buys — a `scale` of ones and
the bare operator.

---

## Fixing Parameters During Optimisation

`MeshNode.fix_parameter()` excludes specific degrees of freedom from
optimisation.  This is useful for anchoring corners or enforcing symmetry:

```python
# Fix the full location of node at index 0
mesh.nodes[0].fix_parameter('loc')

# Fix only the x-component of the u-derivative on node 1
mesh.nodes[1].fix_parameter('du', inds=[0])

# Fix the location and set it to a specific value
mesh.nodes[2].fix_parameter('loc', values=np.array([0., 0., 1.]))

# Regenerate after fixing
mesh.generate_mesh()
```

To remove all fixed parameters:

```python
mesh.unfix_mesh()
```
