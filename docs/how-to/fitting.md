# Mesh Fitting and Optimisation

HOMER offers two main fitting strategies:

1. **Linear least-squares** (`get_xi_weight_mat` + `linear_fit`) – fast when
   the parametric embeddings are fixed.
2. **Nonlinear optimisation** (`point_cloud_fit` + `scipy.optimize`) – for
   free-form shape optimisation against a point cloud.

---

## Linear Fitting

Use this when you can compute (or embed) the xi coordinates in advance and
only need to update nodal values.

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3Basis

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
    `linear_fit` requires an **overdetermined** system (`n_pts > n_nodes`).
    Increase `res` if you see an assertion error.

---

## Nonlinear Point-Cloud Fitting

Use `point_cloud_fit` when you want to optimise node positions (and optionally
derivative vectors) to best match an unstructured point cloud.

```python
from HOMER.fitting import point_cloud_fit
from scipy.optimize import least_squares

# 1. Optionally fix some nodes to prevent the mesh from drifting
mesh.get_node(node_id='corner').fix_parameter('loc')

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
