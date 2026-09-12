# Embedding Points into a Mesh

`embed_points()` finds the parametric coordinates `(element_id, xi)` that
correspond to given physical-space points.  This is the first step in many
workflows: fitting secondary fields, evaluating data at mesh locations, or
computing embedding errors.
The solve runs on a JAX backend.  Its closures are built once per
`generate_mesh()` call rather than per `embed_points()` call, so repeated
embeddings against the same mesh do not retrace.

---

## Basic Usage

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3Basis

# ... build your mesh ...

# Embed 1000 random points
pts = np.random.rand(1000, 3)
elem_ids, xis = mesh.embed_points(pts)

# elem_ids: shape (1000,) – element index for each point
# xis:      shape (1000, ndim) – parametric coordinates in [0,1]^ndim
```

---

## Checking Embedding Quality

Any `verbose` above 0 prints the mean and max residual error:

```python
elem_ids, xis = mesh.embed_points(pts, verbose=1)
# final mean error of 0.0012 units, max error of 0.0041
```

`verbose=3` also renders the errors as a PyVista scene.  Pass `vis_max_norm`
to drop points worse than a cutoff from that picture, and `max_c` to set the
colour-bar maximum (it defaults to the 99th percentile of the error).

---

## Recovering the Residual

When `return_residual=True`, the function returns the vector distance between
each query point and its nearest mesh location:

```python
(elem_ids, xis), residuals = mesh.embed_points(pts, return_residual=True)
# residuals: shape (1000, 3) – vector error for each embedded point
import numpy as np
mean_error = np.mean(np.linalg.norm(residuals, axis=-1))
```

---

## Providing Initial Estimates

If you already have approximate embeddings (e.g. from a previous solve), pass
them as `init_elexi` to skip the coarse nearest-neighbour search:

```python
(elem_ids, xis), res = mesh.embed_points(
    pts,
    init_elexi=(prev_elem_ids, prev_xis),
    return_residual=True,
)
```

---

## Surface Embedding (3-D Volume Meshes)

Often, volumetric data provides point surfaces, which must be embedded into the surface of the mesh.
To restrict the search to the external faces of a volume mesh, pass
`surface_embed=True`:

```python
elem_ids, xis = mesh.embed_points(surface_pts, surface_embed=True)
```

---

## Controlling Convergence

The `iterations` parameter controls how many Newton-Raphson refinement steps
are taken after the coarse nearest-neighbour search (default 15).  The
refinement converges quadratically, so more steps buy little once the solve has
converged; raise it when an element is strongly curved relative to its size:

```python
(elem_ids, xis), res = mesh.embed_points(pts, iterations=20,
                                          return_residual=True)
```

---

## Embedding in a Subset of Dimensions

`dim_mask` restricts the residual to some components of the field, which is
how you embed against a projection: a silhouette, a single slice, or a
multi-state field where only some states are observed.

```python
# match x and y only, and let z fall where it may
elem_ids, xis = mesh.embed_points(pts, dim_mask=np.array([True, True, False]))

# or a per-point mask, shape (n_pts, fdim)
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
space, say.  The JVP reuses the Jacobian the Newton solve already converged
to rather than recomputing one.

```python
import jax
import jax.numpy as jnp

def loss(params):
    (_, xis), res = mesh.embed_points(pts, fit_params=params,
                                      return_residual=True)
    return jnp.sum(res ** 2)

grad = jax.grad(loss)(mesh.optimisable_param_array)
```

`approx_jac=True` drops the sliding term from the residual gradient.  It is
less accurate but keeps the derivative separable by dimension, which
compresses the Jacobian further; the estimate keeps the right sign.

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

```python
# Evaluate primary geometry
locs = mesh.evaluate_embeddings_ele_xi_pair(elem_ids, xis)

# Evaluate a secondary field
fibre_values = mesh['fibre'].evaluate_embeddings_ele_xi_pair(elem_ids, xis)
```
