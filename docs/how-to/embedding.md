# Embedding Points into a Mesh

`embed_points()` finds the parametric coordinates `(element_id, xi)` that
correspond to given physical-space points.
If these points lie outside of the mesh, it will find the parametric coordinates that minimise the distance 
to the physical-space points.

This is the first step in many workflows: fitting secondary fields, evaluating data at mesh locations, or
computing embedding errors.
The solve runs on a JAX backend.  

---

## Basic Usage

```python exec="true" source="above" session="embedding"
import numpy as np

from HOMER import L1
from HOMER.geometry import cube

mesh = cube(basis=L1**3)
mesh.refine(2)

# Embed 1000 random points
rng = np.random.default_rng(0)
pts = rng.random((1000, 3))
elem_ids, xis = mesh.embed_points(pts)

# elem_ids: shape (1000,) – element index for each point
# xis:      shape (1000, ndim) – parametric coordinates in [0,1]^ndim
```

---

## Embedding onto a 2-D Surface

A volume mesh has somewhere to put an interior point; a surface mesh does not,
so every point off the sheet keeps a residual.  That makes a 2-D manifold the
clearer picture of what the solve is actually doing — it is a closest-point
projection, and the answer it converges to is the foot of the perpendicular.

```python exec="true" source="above" session="embedding"
from HOMER.examples import bulged_patch

# the curved quadratic patch, subdivided so the projection has to choose
# between elements as well as within one
patch = bulged_patch()
patch.refine(4)

# a slab of points sitting off the patch
cloud = rng.random((1000, 3))
cloud[:, 0] = 0.6

(ele, xi), residual = patch.embed_points(cloud, return_residual=True,
                                         iterations=20, verbose=3)
```

The surface normal is the check that costs nothing here: a converged projection
leaves a residual with no component along the surface, so the residual and the
normal must be parallel.

```python exec="true" source="above" session="embedding"
def unit(v):
    v = np.asarray(v)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

alignment = np.abs(np.sum(unit(patch.evaluate_normals_ele_xi_pair(ele, xi))
                          * unit(residual), axis=-1))
print(f"mean |n . r| = {alignment.mean():.4f}, worst {alignment.min():.3f}")
```

The worst points are the ones whose projection lands on the *edge* of the
patch: they are as close as they can get, but the direction back to them is not
a normal, because the constrained minimum is on the boundary rather than in the
interior.

---

## Checking Embedding Quality

Any `verbose` above 0 prints the mean and max residual error:

```python exec="true" source="above" session="embedding"
elem_ids, xis = mesh.embed_points(pts, verbose=1)
```

`verbose=3` also renders the errors as a PyVista scene.  Pass `vis_max_norm`
to drop points worse than a cutoff from that picture, and `max_c` to set the
colour-bar maximum (it defaults to the 99th percentile of the error).

```python exec="true" source="above" session="embedding"
elem_ids, xis = mesh.embed_points(pts, verbose=3)
```

---

## Recovering the Residual

When `return_residual=True`, the function returns the vector distance between
each query point and its nearest mesh location:

```python exec="true" source="above" session="embedding"
(elem_ids, xis), residuals = mesh.embed_points(pts, return_residual=True)
# residuals: shape (1000, 3) – vector error for each embedded point
mean_error = np.mean(np.linalg.norm(residuals, axis=-1))
print(f"{residuals.shape} residuals, mean error {mean_error:.2e}")
```

This is intended for use as a fitting metric, allowing optimisation over the 
data-to-model distance from each point in pts to the mesh.

---

## Initial Estimates

If you already have approximate embeddings (e.g. from a previous solve), pass
them as `init_elexi` to skip the coarse nearest-neighbour search:

```python exec="true" source="above" session="embedding"
(elem_ids, xis), res = mesh.embed_points(
    pts,
    init_elexi=(elem_ids, xis),        # the solve above
    return_residual=True,
)
```

If an estimate is not provided, an initial estimate is found using either a
Morton Z-curve or approximate K-nn algorithm, depending on field dimension
and the provided `dim mask`.


---

## Surface Embedding (3-D Volume Meshes)

Often, volumetric data provides point surfaces, which must be embedded into the surface of the mesh.
To restrict the search to the external faces of a volume mesh, pass
`surface_embed=True`:

```python exec="true" source="above" session="embedding"
surface_pts = rng.random((200, 3))
elem_ids, xis = mesh.embed_points(surface_pts, surface_embed=True)
```

---

## Controlling Convergence

The `iterations` parameter controls how many Newton-Raphson refinement steps
are taken after the coarse nearest-neighbour search (default 15).  The
refinement converges quadratically, so more steps buy little once the solve has
converged; raise it when an element is strongly curved relative to its size:

```python exec="true" source="above" session="embedding"
(elem_ids, xis), res = mesh.embed_points(pts, iterations=20,
                                          return_residual=True)
```

The embedding algorithm allows for early stopping if all points are converged, so 
this will only increase runtime if the worst case embeddings were slow.

---

## Embedding in a Subset of Dimensions

`dim_mask` restricts the residual to some components of the field, which is
how you embed against a projection: a silhouette, a single slice, a
multi-state field where only some states are observed, or a time-varying mesh.


```python exec="true" source="above" session="embedding"
# match x and y only, and let z fall where it may
elem_ids, xis = mesh.embed_points(pts, dim_mask=np.array([True, True, False]))

# or a per-point mask, shape (n_pts, fdim)
per_point_mask = np.tile([True, True, False], (len(pts), 1))
elem_ids, xis = mesh.embed_points(pts, dim_mask=per_point_mask)
```

Masked components come back as exactly zero in both the residual and its
Jacobian.  The mask is *static* — it states which residual components exist,
so differentiating never produces a derivative with respect to it.  A row
that masks out more dimensions than the mesh has parametric directions
leaves the solve under-determined, and the derivative then takes the
minimum-norm solution.

A non-trivial mask also switches the coarse search from the Morton Z-curve
lookup to an exact one: a Z-curve code describes a whole coordinate vector,
so it cannot express "these components only".

---

## Differentiating Through the Embedding

`embed_points` carries a custom JVP, so it can sit inside a larger
differentiable pipeline — a loss defined on where data lands in parametric
space, or on the embedding residual.

```python exec="true" source="above" session="embedding"
import jax
import jax.numpy as jnp

def loss(params):
    (_, xis), res = mesh.embed_points(pts, fit_params=params,
                                      return_residual=True)
    return jnp.sum(res ** 2)

grad = jax.grad(loss)(mesh.optimisable_param_array)
print(f"gradient of {grad.shape} over the mesh parameters")
```

`approx_jac=True` drops the sliding term from the residual gradient.  It is
less accurate but keeps the derivative separable by dimension.
This approximation increases the sparsitty of the jacobian by a factor of the mesh field dimension.
As the estimate keeps the right sign, the worse convergence per step is often worth the speed up,
or only possible because of the reduced memory usage.

---

## Tuning the Solve

| parameter | default | what it does |
| --- | --- | --- |
| `grid_res` | `10` | xi samples per direction in the coarse search that seeds the solve. Raise it when elements are large or strongly curved, so the seed starts in the right element. |
| `window_size` | `16` | width of the Morton-code window the coarse search examines. Larger is more accurate per seed, at the cost of memory and time. Ignored when `dim_mask` masks anything off, since that path searches exactly. |
| `chunk_size` | `None` | processes query points in batches of at most this size, bounding peak memory at `O(chunk_size)` rather than `O(n_pts)`. |
| `tol` | `None` | residual norm at which refinement stops, making `iterations` an upper bound. The default is `1e-6` times the mesh extent — float32 round-off, which the solve reaches in two or three iterations. Pass `0` to iterate unconditionally. |
| `robust_init_est` | `False` | takes the first Newton step from a numeric Jacobian rather than the analytic one. The Jacobian returned is analytic either way. |

The refinement is vectorised over the query points, so a batch runs until
its *slowest* point finishes: one point that cannot converge — lying well
off the mesh, say — keeps the whole batch going to `iterations`.  That costs
only the saving; a converged point's state is frozen, so its answer is
identical to the one it would get on its own.

Changing `approx_jac` or `robust_init_est` away from their defaults builds a
second compiled variant, cached per flag pair, so a fitting loop that passes
the same flags every call pays the retrace once.

---

## Using Embedded Coordinates

Once embedded, evaluate any mesh field at those locations:

```python exec="true" source="above" session="embedding"
# a secondary field to read back -- see the secondary fields guide
mesh.new_field('fibre', field_dimension=3, new_basis=L1**3,
               field_locs=pts, field_values=np.tile([1., 0., 0.], (len(pts), 1)))

# Evaluate primary geometry
locs = mesh.evaluate_embeddings_ele_xi_pair(elem_ids, xis)

# Evaluate a secondary field
fibre_values = mesh['fibre'].evaluate_embeddings_ele_xi_pair(elem_ids, xis)
print(locs.shape, fibre_values.shape)
```
