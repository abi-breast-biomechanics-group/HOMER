# Mesh Fitting and Optimisation

HOMER offers two main pre-built fitting strategies:

1. **Linear least-squares** (`get_xi_weight_mat` + `linear_fit`) – fast when
   the parametric embeddings are fixed.
2. **Nonlinear optimisation** (`point_cloud_fit` + `scipy.optimize`) – for
   free-form shape optimisation against a point cloud.

However, as HOMER uses JAX to implement the mesh representations, it allows very expressive manipulations of mesh geometry
while still automatically defining the Jacobian.

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
    ones. When you need constraints honoured, use the nonlinear pathway below, which optimises
    `optimisable_param_array` and leaves fixed parameters alone.

---

## Nonlinear Fitting

The function `point_cloud_fit` gives an example of a model-to-data based fit.
It samples the mesh on an xi grid with `evaluate_embeddings`, measures the
distance from each sample to its nearest target point through a SciPy KD-tree
(wrapped in a JAX custom-JVP callback, see `HOMER.optim`), appends a Sobolev
smoothness term, and hands the whole residual to `jacobian` so the derivative
comes out of JAX rather than a finite difference.  Use it -- or the same three
steps written out for your own cost -- when you want to optimise node positions
(and optionally derivative vectors) to best match an unstructured point cloud.

```python
from HOMER.fitting import point_cloud_fit
from scipy.optimize import least_squares

# 1. Optionally fix some nodes to prevent the mesh from drifting
mesh.get_node('corner').fix_parameter('loc')

# 2. Build the cost function and Jacobian
fitting_fn, jac_fn = point_cloud_fit(
    mesh, target_pts, sob_weight=0.01
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

The term itself is `mesh.evaluate_sobolev()`, which evaluates every non-trivial
combination of derivative orders at the element's Gauss points.  Call it
directly, with one `weights` entry per term, when a single scalar is too blunt
-- to smooth across a surface without penalising curvature through its
thickness, say.

---

## Jacobian Calculation

Because the mesh evaluation is written in JAX, any residual you can express
with it is differentiated for you: `jacobian(cost_function,
init_estimate=...)` returns the JIT-compiled cost and a Jacobian function, both
in the shape `scipy.optimize.least_squares` expects.  Nothing about the cost
has to be a built-in fit -- deform the mesh, evaluate a secondary field, embed
points, compose the lot, and the derivative still follows.

```python
from HOMER import jacobian

def cost(params):
    pts = mesh.evaluate_embeddings(elements, xis, fit_params=params)
    return jnp.ravel(pts - target_pts)

fitting_fn, jac_fn = jacobian(cost, init_estimate=mesh.optimisable_param_array)
```

These Jacobians are almost always sparse: a residual sampled inside one element
depends only on the parameters of that element's nodes.  `jacobian` exploits
that by default (`sparse=True`), building the derivative with `sparsejac`'s
forward-mode AD over a sparsity pattern and returning a
`scipy.sparse.coo_array`; `sparse=False` falls back to a dense `jax.jacfwd`,
which is the better choice for small problems.

With no `sparsity` given, the pattern is found by `estimate_sparsity`, which
perturbs one parameter at a time and records which outputs move.  It is exact
enough, and memory-cheap, but it costs one residual evaluation per parameter --
slow on any real mesh.  It also detects *structural* dependence only: an output
that happens not to move under a unit step at `init_estimate` is recorded as
independent, so pass a starting estimate representative of where the optimiser
will actually work.

### Recovering the Pattern From the Topology

A mesh already knows its sparsity pattern -- it is the element-to-node map (see
[Topology mapping](topology.md)).  `get_colouring_dict` turns that into a
colouring, grouping parameters that can never influence the same output:

```python
colours, seed_values, seed_indices = mesh.get_colouring_dict(
    fields_seperable=True, seed_matrix=True)
n_colours = max(colours.values()) + 1
```

`fields_seperable=True` treats each component of a vector field as its own
output, which is true of embedding evaluation (but not of, say, a local
Jacobian determinant) and needs markedly fewer colours.  The seed matrices are
`(n_parameters, n_colours)`, one entry per parameter: `seed_values` holds a 1
there and `seed_indices` holds the parameter's own index.

One `jvp` per colour then compresses the whole Jacobian, because at most one
parameter of a given colour reaches any given row.  `J @ seed_values` gives the
nonzero values and `J @ seed_indices` the same entries weighted by their column
index, so dividing one by the other says which column each value came from.
`n_colours` evaluations recover a pattern that probing would have charged
`n_parameters` for.

`make_jac_for_mesh_func` is that loop, wrapped:

```python
from HOMER import make_jac_for_mesh_func, make_static_jac_for_mesh_func

jac_fn = make_jac_for_mesh_func(mesh, residual, fields_seperable=True)
jac = jac_fn(params)          # BCOO, (n_residuals, n_parameters)
```

Because the indices are decoded on every call, the *pattern* is free to move
between them -- which is what a data-to-model term needs.  A residual built on
`embed_points` re-embeds its data as the mesh deforms, so a point can land in a
different element from one iteration to the next; the colouring stays valid
throughout, since every residual entry still draws on a single element's
parameters, while the decoded indices follow the data.

Pass `approx_jac=True` to `embed_points` for this.  It holds each point at the
`(element, xi)` it embedded to instead of letting it slide, which is what keeps
each residual component dependent on only the matching field component --
i.e. what makes `fields_seperable=True` true:

```python
def residual(params):
    return mesh.embed_points(points, fit_params=params, return_residual=True,
                             approx_jac=True)[1].flatten()
```

!!! warning "`fields_seperable` has to match how the residual couples"
    A mismatch is silent.  Entries the index decode cannot resolve are dropped,
    so the Jacobian comes back sparse, plausible and wrong rather than raising.
    The exact embedding Jacobian (`approx_jac=False`, the default) slides the
    point as the mesh deforms, which makes every residual component depend on
    every component of the element's parameters: separable in elements, not in
    fields, so it needs `fields_seperable=False`.

### Freezing the Pattern

Most fits are not data-to-model.  Anything evaluated at fixed `(element, xi)`
pairs -- a grid fit, a Sobolev term, a field residual -- has a pattern that
never moves, and decoding it on every call is half the work done twice.
`make_static_jac_for_mesh_func` decodes once at a starting estimate, keeps the
indices, and leaves one `jvp` per colour to do per call:

```python
jac_fn = make_static_jac_for_mesh_func(mesh, residual, p_start,
                                       fields_seperable=True)
```

It costs about half as much per call as the dynamic version.  The caveats are
the two that go with any frozen pattern: an entry that happens to vanish at
`p_start` is recorded as absent and stays absent, so start somewhere
representative; and if the residual *does* re-embed its data, the indices are
wrong the moment a point changes element -- that case wants the dynamic
version.

Both return a `BCOO`.  `scipy.optimize.least_squares` wants a SciPy matrix, so
convert on the way out:

```python
jac = jac_fn(params)
scipy.sparse.coo_array((jac.data, jac.indices.T), shape=jac.shape)
```

---

### Fits Too Large to Hold a Jacobian

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

This parameter fixing is achieved by representing a scatter from the optimisable 
subset of the parameters to the true parameter array of the mesh.
A similar strategy can be used to represent nodes with shared values, or with dependencies 
on other functions or parameters.
