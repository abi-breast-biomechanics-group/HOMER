# Fitting Secondary Mesh Vector and Scalar Fields

Secondary fields allow you to store arbitrary spatially-varying data
(fibre directions, velocity vectors, stresses, material properties, etc.)
as a smooth interpolation over the same mesh topology as the primary geometry.

---

## Concepts

A secondary field is a `MeshField` whose values are not 3-D physical
coordinates but some other quantity:

| Field type | `field_dimension` | Example |
|---|---|---|
| Scalar | 1 | pressure, temperature, Z-height |
| n-D vector | n | fibre direction, velocity, surface normal |

Secondary fields:

- Share the **same parametric topology** as the primary `Mesh` but can use
  *different* basis functions.
- Are created with `mesh.new_field(...)` and stored in `mesh.fields`.
- Are accessed as `mesh['field_name']`.

---

## How HOMER Fits a Secondary Field

`new_field()` with sample data follows three steps:

1. **Embed sample points** – call `embed_points()` on the primary mesh to
   find the `(elem_id, xi)` coordinates of every sample location.
2. **Build the weight matrix** – call `get_xi_weight_mat(elem_ids, xis)` on
   the new field to relate sample locations to nodal degrees of freedom.
3. **Solve the linear system** – call `linear_fit(targets, W)` to compute
   the optimal nodal parameters.

---

## Creating a Secondary Field

### Basic API

```python
mesh.new_field(
    field_name='field_key',        # access key: mesh['field_key']
    field_dimension=3,             # 1=scalar, 3=vector
    new_basis=[H3Basis]*3,         # one basis per parametric direction
    field_locs=sample_pts,         # shape (N, 3) – physical sample locations
    field_values=sample_values,    # shape (N,) or (N, 3)
)
```

When `field_locs` and `field_values` are `None`, an empty field topology is
created but the nodal parameters are left at zero.

---

## Worked Example – Normal Vector + Scalar Height Fields

This example reproduces the workflow from `tests/create_mesh_field.py`.

```python
import math
import numpy as np
import pyvista as pv
from HOMER import Mesh, MeshNode, MeshElement, L1Basis, H3Basis

# ── 1. Build a unit-cube mesh in H3×H3×H3 ──────────────────────────────────
nodes = [MeshNode(loc=[x, y, z])
         for x in [0,1] for y in [0,1] for z in [0,1]]
element = MeshElement(node_indexes=list(range(8)),
                      basis_functions=(L1Basis, L1Basis, L1Basis))
mesh = Mesh(nodes=nodes, elements=element)
mesh = Mesh(nodes=mesh.rebase([H3Basis]*3).nodes,
            elements=mesh.rebase([H3Basis]*3).elements)

# Alternatively, a one-liner rebase:
mesh.rebase([H3Basis]*3)   # returns a MeshField; use Mesh(...) to re-wrap

# ── 2. Generate sample data ─────────────────────────────────────────────────
def fibonacci_sphere(n, radius=0.5, centre=(0., 0., 0.)):
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(n)
    z = radius * (1.0 - 2.0 * i / (n - 1))
    r_xy = radius * np.sqrt(1.0 - (z / radius) ** 2)
    theta = golden_angle * i
    x = r_xy * np.cos(theta);  y = r_xy * np.sin(theta)
    return np.vstack((x, y, z)).T + centre

# Sample points on three concentric shells inside the cube
data = np.concatenate([
    fibonacci_sphere(300, radius=0.49, centre=(0.5, 0.5, 0.5)),
    fibonacci_sphere(300, radius=0.39, centre=(0.5, 0.5, 0.5)),
    fibonacci_sphere(300, radius=0.29, centre=(0.5, 0.5, 0.5)),
])

# Outward-pointing unit normals
normal_field = data - (0.5, 0.5, 0.5)
normal_field /= np.linalg.norm(normal_field, axis=-1, keepdims=True)

# Scalar height (Z-coordinate)
z_field = data[:, 2]

# ── 3. Refine the mesh to increase resolution ────────────────────────────────
mesh.refine(2)

# ── 4. Fit the vector field ──────────────────────────────────────────────────
mesh.new_field(
    'vec_dir',
    field_dimension=3,
    field_locs=data,
    field_values=normal_field,
    new_basis=[H3Basis]*3,
)

# ── 5. Fit the scalar field ──────────────────────────────────────────────────
mesh.new_field(
    'vec_mag',
    field_dimension=1,
    field_locs=data,
    field_values=z_field,
    new_basis=[L1Basis]*3,
)

# ── 6. Evaluate the fitted field ─────────────────────────────────────────────
xis = mesh.xi_grid(4, boundary_points=False)
locs   = mesh.evaluate_embeddings_in_every_element(xis)   # (n, 3)
norms  = mesh['vec_dir'].evaluate_embeddings_in_every_element(xis)   # (n, 3)
heights = mesh['vec_mag'].evaluate_embeddings_in_every_element(xis)  # (n, 1)

# ── 7. Visualise ─────────────────────────────────────────────────────────────
s = pv.Plotter()
mesh.plot(s, field_to_draw='vec_dir', default_xi_res=6)
s.add_arrows(data, normal_field, mag=0.1)  # raw samples for comparison
s.show()

# Plot the scalar field alone
mesh['vec_mag'].plot()
```

---

## Accessing and Evaluating a Fitted Field

```python
# Retrieve the secondary MeshField
fibre_field = mesh['vec_dir']      # MeshField instance

# Evaluate at arbitrary parametric locations
elem_ids = np.array([0, 0, 1])
xis      = np.array([[0.2, 0.3, 0.4],
                     [0.5, 0.5, 0.5],
                     [0.1, 0.9, 0.5]])
values = fibre_field.evaluate_embeddings(elem_ids, xis)  # (3, 3)

# Or across the whole mesh at once
all_values = fibre_field.evaluate_embeddings_in_every_element(
    mesh.xi_grid(5)
)  # (n_elements * 125, 3):wa

```

---

## Visualising Secondary Fields

```python
# Draw the mesh + vector field overlaid
mesh.plot(field_to_draw='vec_dir', default_xi_res=6)

# Draw only the secondary field (without primary geometry)
mesh['vec_dir'].plot()

# Custom artist for arrows instead of line segments
import pyvista as pv
def arrow_artist(scene, locs, values):
    scene.add_arrows(locs, values, mag=0.1)

mesh.plot(field_to_draw='vec_dir', field_artist=arrow_artist)
```

---

## Tips

- Use **`H3Basis`** for smooth vector fields (fibre directions, velocities)
  that must interpolate continuously across element boundaries.
- Use **`L1Basis`** or **`L2Basis`** for simpler scalar fields (pressure,
  temperature) where smoothness is less critical.
- Ensure you have **more sample points than nodal degrees of freedom**.  If
  `linear_fit` raises an assertion error, add more sample points or reduce
  the basis order.
- After fitting, check the residual printed by `linear_fit` to assess fit
  quality.  The residual is the total squared-norm of the fitting error.
