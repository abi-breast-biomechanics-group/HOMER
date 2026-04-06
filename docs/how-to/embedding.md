# Embedding Points into a Mesh

`embed_points()` finds the parametric coordinates `(element_id, xi)` that
correspond to given physical-space points.  This is the first step in many
workflows: fitting secondary fields, evaluating data at mesh locations, or
computing embedding errors.
This uses a JAX based backend, so is still relatively performant, even for large numbers of points.
Future versions of HOMER will include an embedding deriv function, which returns the sparse derivatives of the underlying function.

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

Set `verbose=2` to print mean and max residual errors:

```python
elem_ids, xis = mesh.embed_points(pts, verbose=2)
# final mean error of 0.0012 units, max error of 0.0041
```
Set `verbose=3` to render an interactive visualisation of the embedding
errors with PyVista.

---

## Recovering the Residual

When `return_residual=True`, the function returns the vector distance between
each query point and its nearest mesh location:
This is a fully JAX compliant function, so can be used in optimisation.
However, it is currently dense and quickly becomes prohibitively expensive to compute.

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

The `iterations` parameter controls how many RK4 refinement steps are taken.
Increase for difficult geometries:

```python
(elem_ids, xis), res = mesh.embed_points(pts, iterations=20,
                                          return_residual=True)
```

---

## Using Embedded Coordinates

Once embedded, evaluate any mesh field at those locations:

```python
# Evaluate primary geometry
locs = mesh.evaluate_embeddings_ele_xi_pair(elem_ids, xis)

# Evaluate a secondary field
fibre_values = mesh['fibre'].evaluate_embeddings_ele_xi_pair(elem_ids, xis)
```
