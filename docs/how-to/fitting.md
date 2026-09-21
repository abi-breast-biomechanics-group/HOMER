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

```python exec="true" source="above" session="fitting"
from copy import deepcopy

import numpy as np
import pyvista as pv

from HOMER import L3
from HOMER.examples import bulged_patch
from HOMER.geometry import basic_surface

# 1. A curved target and a flat cubic mesh to pull onto it, as
#    tests/test_fitting.py fits them: both refined from one quad, so element
#    k of each covers the same parametric patch
target_mesh = bulged_patch()
target_mesh.refine(2)

mesh = basic_surface(basis=L3**2)
mesh.refine(2)
before = deepcopy(mesh)

res = 5
xis = mesh.xi_grid(res)
# Repeat the grid for each of 4 elements
xis_tiled = np.tile(xis, (4, 1))
elem_ids = np.repeat(np.arange(4), res**2)

# 2. Sample target geometry at those (elem, xi) locations
target_pts = np.asarray(target_mesh.evaluate_embeddings_ele_xi_pair(elem_ids, xis_tiled))

# 3. Build the weight matrix
W = mesh.get_xi_weight_mat(elem_ids, xis_tiled)

# 4. Solve the linear system in-place
mesh.linear_fit(target_pts, weight_mat=W)

def with_targets(scene):
    scene.add_points(target_pts, color='b', point_size=3,
                     render_points_as_spheres=True)

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); before.plot(s); with_targets(s)
s.subplot(0, 1); mesh.plot(s, node_colour='g'); with_targets(s)
s.link_views()
s.show()
```

!!! note
    `linear_fit` needs at least as many points as the system has *free*
    columns.  The columns are parameters, not nodes: a Hermite node carries
    four parameter blocks in 2-D and eight in 3-D, so a 4-node H3 patch with
    nothing fixed already needs 16 points.  Increase `res` if you see an
    assertion error.

### Fixed Parameters in a Linear Fit

Parameters fixed with `MeshNode.fix_parameter()` are held at their current
values.  They are a known contribution to the targets, so the solve moves them
to the right-hand side and fits only the free columns -- the constrained
minimiser, not an unconstrained fit that overwrites the constraint afterwards:

```python exec="true" source="above" session="fitting"
mesh.nodes[0].fix_parameter('loc')          # this corner stays where it is
mesh.nodes[4].fix_parameter('loc', inds=[2])  # this one only holds its z
mesh.generate_mesh()

mesh.linear_fit(target_pts, weight_mat=W)
```

A held parameter comes back bit-for-bit, and every other parameter is the best
it can be given it.  Fixing also makes the system *smaller*, so it needs fewer
points, not more.

!!! note "Fixing is per component"
    `fix_parameter('loc', inds=[2])` pins only *z*, so the free set can differ
    between the components of the field.  `W` is shared across them, so
    components that share a free set share a solve: one solve for the usual
    mesh, at most `fdim` when the constraints cut across components.

---

## Nonlinear Fitting

The function `point_cloud_fit` is an example of a model-to-data based fit.
It samples the mesh on an xi grid with `evaluate_embeddings`, measures the
distance from each sample to its nearest target point through a SciPy KD-tree
(wrapped in a JAX custom-JVP callback, see `HOMER.optim`), and appends a Sobolev
smoothness term. 

The fitting function is passed through `jacobian` using JAX to define the sparse 
representation of the jacobian.

```python exec="true" source="above" session="fitting"
from HOMER import H3
from HOMER.fitting import point_cloud_fit
from scipy.optimize import least_squares

# 1. The target as an unstructured cloud, and a flat Hermite patch to fit
cloud = bulged_patch()
cloud_grid = cloud.xi_grid(20)
target_cloud = np.asarray(cloud.evaluate_embeddings_ele_xi_pair(
    np.zeros(len(cloud_grid), int), cloud_grid))

fit_mesh = basic_surface(basis=H3**2)
start = np.asarray(fit_mesh.optimisable_param_array)
flat = deepcopy(fit_mesh)

# 2. Build the cost function and Jacobian
fitting_fn, jac_fn = point_cloud_fit(
    fit_mesh, target_cloud, sob_weight=0.0, compile=True
)

# 3. Run the optimiser
result = least_squares(fitting_fn, start, jac=jac_fn, max_nfev=60)

# 4. Apply the optimised parameters
fit_mesh.update_from_params(result.x)

def with_cloud(scene):
    scene.add_points(target_cloud, color='b', point_size=2,
                     render_points_as_spheres=True)

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); flat.plot(s); with_cloud(s)
s.subplot(0, 1); fit_mesh.plot(s, node_colour='g'); with_cloud(s)
s.link_views()
s.show()
```

The flat patch it started from is on the left and the fitted one on the right,
against the same cloud.  Only the four corner nodes moved: what pulls the
interior onto the curve is the Hermite tangents, which are parameters of those
same four nodes.

### Sobolev Regularisation

The `sob_weight` parameter adds a Sobolev smoothness term that penalises
high curvature in the mesh surface.  Increase it if the mesh develops
wrinkles:

```python exec="true" source="above" session="fitting"
fitting_fn, jac_fn = point_cloud_fit(fit_mesh, target_cloud, sob_weight=0.1)
```

The term itself is `mesh.evaluate_sobolev()`, which evaluates every non-trivial
combination of derivative orders at the element's Gauss points.  

---

## Jacobian Calculation

Because the mesh evaluation is written in JAX, any residual you can express
with it is differentiated for you: `jacobian(cost_function,
init_estimate=...)` returns the JIT-compiled cost and a Jacobian function, both
in the shape `scipy.optimize.least_squares` expects.  

```python exec="true" source="above" session="fitting"
import jax.numpy as jnp

from HOMER import jacobian

elements, xis = elem_ids, xis_tiled

def cost(params):
    pts = mesh.evaluate_embeddings_ele_xi_pair(elements, xis, fit_params=params)
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

```python exec="true" source="above" session="fitting"
from HOMER import L1
from HOMER.geometry import cube

coloured = cube(basis=L1**3)
coloured.refine(4)

colours, seed_values, seed_indices = coloured.get_colouring_dict(
    fields_seperable=True, seed_matrix=True)
n_colours = max(colours.values()) + 1
print(f"{n_colours} colours over {len(colours)} parameters")
```
Drawn on the mesh it is the pattern you would guess: a trilinear node's
parameters reach the eight elements around it, so the colouring is the
eight-way checkerboard that lets every node be perturbed alongside its
next-but-one neighbours.  

```python exec="true" source="above" session="fitting"
node_colours = np.array([colours[3 * i] for i in range(len(coloured.nodes))])
coloured.plot(node_colour=node_colours, node_size=25,
              node_col_scalar_name='colour')
```

`fields_seperable=True` treats each component of a vector field as its own
output, which is true of embedding evaluation (but not of, say, a local
Jacobian determinant) and needs markedly fewer colours.  The seed matrices are
`(n_parameters, n_colours)`, one entry per parameter: `seed_values` holds a 1
there and `seed_ranks` holds the parameter's position within its own colour.

One `jvp` per colour then compresses the whole Jacobian, because at most one
parameter of a given colour reaches any given row.  `J @ seed_values` gives the
nonzero values and `J @ seed_ranks` the same entries weighted by that position,
so dividing one by the other says which parameter of the colour each value came
from -- and the colour is the row, so that names the column.  The weight is a
rank rather than a parameter index because the decode recovers it by dividing:
ranks run to the size of one colour instead of to the parameter count, which is
what keeps the quotient sharp enough to round correctly in float32.
`n_colours` evaluations recover a pattern that probing would have charged
`n_parameters` for.

`make_jac_for_mesh_func` is that loop, wrapped:

```python exec="true" source="above" session="fitting"
from HOMER import make_jac_for_mesh_func, make_static_jac_for_mesh_func

params = np.asarray(mesh.optimisable_param_array)

def residual(params):
    return cost(params)

jac_fn = make_jac_for_mesh_func(mesh, residual, fields_seperable=True)
jac = jac_fn(params)          # coo_array, (n_residuals, n_parameters)
print(jac.shape, jac.nnz, "non-zeros")
```

Because the indices are decoded on every call, the *pattern* is free to move
between them -- which is what a data-to-model term needs.  A residual built on
`embed_points` re-embeds its data as the mesh deforms, so a point can land in a
different element from one iteration to the next.
Despite the changing location, the colouring stays valid, as each point only embeds 
into one element at a time.

Pass `approx_jac=True` to `embed_points` for this.  It holds each point at the
`(element, xi)` it embedded to instead of letting it slide, which is what keeps
each residual component dependent on only the matching field component --
i.e. what makes `fields_seperable=True` true:

```python exec="true" source="above" session="fitting"
points = np.asarray(target_pts)

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
never moves, so repeatedly decoding it is uneccessary.
`make_static_jac_for_mesh_func` decodes once at a starting estimate, keeps the
indices, and leaves one `jvp` per colour to do per call:

```python exec="true" source="above" session="fitting"
p_start = np.asarray(mesh.optimisable_param_array)

jac_fn = make_static_jac_for_mesh_func(mesh, cost, p_start,
                                       fields_seperable=True)
```

It costs about half as much per call as the dynamic version.  The caveats are
the two that go with any frozen pattern: an entry that happens to vanish at
`p_start` is recorded as absent and stays absent, so start somewhere
representative; and if the residual *does* re-embed its data, the indices are
wrong the moment a point changes element -- that case wants the dynamic
version.

Both return a `scipy.sparse.coo_array`, which is what
`scipy.optimize.least_squares` wants, so they go straight to it with no
conversion.  The static one holds its indices on the host, so only the values
come back from the device on each call.

A residual whose data changes between solves -- a tracker stepping through
frames, a correspondence refound each iteration -- takes that data as keyword
arguments, and both makers forward them to it behind the parameter vector.
That is the route `least_squares(..., kwargs=...)` already uses, so one dict
feeds the residual and its Jacobian alike:

```python
jac_fn = make_static_jac_for_mesh_func(
    mesh, residual, p_start, fields_seperable=True,
    further_args={'weights': np.ones(n_obs)})

result = least_squares(residual, p_start, jac=jac_fn,
                       kwargs={'weights': visibility[frame]})
```

`further_args` is deliberately not the data later calls pass.  An entry the
probe masks to zero is recorded as absent for good, so read the pattern with
data that leaves every entry live -- unit weights rather than a visibility
mask -- and pass the real thing per call.

---

### Fits Too Large to Hold a Jacobian

`point_cloud_fit` and `jacobian` both build the Jacobian as a matrix, with a
sparsity colouring to keep it affordable. That stops working on two kinds of
problem: one where the matrix itself is too big (a residual of ~1e5 entries in
~1e4 parameters costs gigabytes), and one where the blocks that matter are
dense, so the colouring cannot compress the parameters.

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

```python exec="true" source="above" session="fitting"
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

```python exec="true" source="above" session="fitting"
mesh.unfix_mesh()
```

This parameter fixing is achieved by representing a scatter from the optimisable 
subset of the parameters to the true parameter array of the mesh.
A similar strategy can be used to represent nodes with shared values, or with dependencies 
on other functions or parameters.

Both fitting pathways read that same subset: the nonlinear one optimises
`optimisable_param_array` directly, and `linear_fit` solves for the columns it
selects.

### A Constrained Fit

Nothing else changes: pin what must not move, regenerate, and hand the same
optimiser the shorter parameter vector.  Here the patch of the nonlinear fit
above is refitted with one corner nailed down.

```python exec="true" source="above" session="fitting"
pinned = basic_surface(basis=H3**2)

# pin the corner somewhere the fit would never have put it, so the constraint
# is visible rather than merely asserted
anchor = pinned.nodes[0].loc + np.array([0.4, 0., -0.3])
pinned.nodes[0].fix_parameter('loc', values=anchor)
pinned.generate_mesh()

fitting_fn, jac_fn = point_cloud_fit(pinned, target_cloud, sob_weight=0.0,
                                     compile=True)
result = least_squares(fitting_fn, np.asarray(pinned.optimisable_param_array),
                       jac=jac_fn, max_nfev=60)
pinned.update_from_params(result.x)

print(f"free parameters: {len(fit_mesh.optimisable_param_array)}"
      f" -> {len(pinned.optimisable_param_array)}")
print("anchor held exactly:", np.array_equal(pinned.nodes[0].loc, anchor))
```

The free fit is on the left and the constrained one on the right.  The pinned
corner is pulled well clear of the cloud and stays exactly there, while the
rest of the patch does the best it can around it — the constrained minimiser,
not a free fit with the constraint stamped back on afterwards.

```python exec="true" source="above" session="fitting"
s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); fit_mesh.plot(s); with_cloud(s)
s.subplot(0, 1); pinned.plot(s, node_colour='g'); with_cloud(s)
s.link_views()
s.show()
```
