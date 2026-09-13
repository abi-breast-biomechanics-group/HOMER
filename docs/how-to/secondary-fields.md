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

!!! note "Fixed parameters are respected"
    Constraints set with `MeshNode.fix_parameter()` on a secondary field's
    nodes are held through the fit: `linear_fit` solves for the free
    parameters only.  See
    [Fixed parameters in a linear fit](fitting.md#fixed-parameters-in-a-linear-fit).

---

## Creating a Secondary Field

### Basic API

```python
mesh.new_field(
    field_name='field_key',        # access key: mesh['field_key']
    field_dimension=3,             # 1=scalar, 3=vector
    new_basis=H3**3,               # one basis per parametric direction
    field_locs=sample_pts,         # shape (N, 3) – physical sample locations
    field_values=sample_values,    # shape (N,) or (N, 3)
)
```

When `field_locs` and `field_values` are `None`, an empty field topology is
created but the nodal parameters are left at zero.

---

## Worked Example – Normal Vector + Scalar Height Fields

The same workflow is exercised by `tests/test_fields.py`.

```python exec="true" source="above" session="secondary-fields"
import math
import numpy as np
import pyvista as pv
from HOMER import Mesh, MeshNode, MeshElement, L1, H3

# ── 1. Build a unit-cube mesh in H3×H3×H3 ──────────────────────────────────
nodes = [MeshNode(loc=[x, y, z])
         for x in [0,1] for y in [0,1] for z in [0,1]]
element = MeshElement(node_indexes=list(range(8)), basis_functions=L1**3)
mesh = Mesh(nodes=nodes, elements=element)
mesh.rebase(H3**3, in_place=True)

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
    new_basis=H3**3,
)

# ── 5. Fit the scalar field ──────────────────────────────────────────────────
mesh.new_field(
    'vec_mag',
    field_dimension=1,
    field_locs=data,
    field_values=z_field,
    new_basis=L1**3,
)

# ── 6. Evaluate the fitted field ─────────────────────────────────────────────
xis = mesh.xi_grid(4, boundary_points=False)
locs   = mesh.evaluate_embeddings_in_every_element(xis)   # (n, 3)
norms  = mesh['vec_dir'].evaluate_embeddings_in_every_element(xis)   # (n, 3)
heights = mesh['vec_mag'].evaluate_embeddings_in_every_element(xis)  # (n, 1)

# ── 7. Visualise ─────────────────────────────────────────────────────────────
# the values are unit normals, so drawn at their own length they are as long as
# the cube is wide; an artist scales them to something the eye can read
def arrows(scene, locs, values, field_xi):
    scene.add_arrows(np.asarray(locs), np.asarray(values), mag=0.1)

s = pv.Plotter()
mesh.plot(s, field_to_draw='vec_dir', default_xi_res=3, field_artist=arrows)
# every eighth raw sample, in red, to compare the fit against its data
s.add_arrows(data[::8], normal_field[::8], mag=0.1, color='r')
s.show()

# A scalar field has no geometry of its own, so it is drawn on the mesh
mesh.plot(field_to_draw='vec_mag', default_xi_res=6)
```

!!! note "A field drawn over a volume is hard to read"
    Every one of these pictures puts a 3-D field inside a solid, so the markers
    behind the front face are drawn through it and the ones in front hide what
    is behind them.  A slice — `field_xi` restricted to one parametric plane —
    or a low `default_xi_res` usually says more than a dense cloud does.  Read
    the render as a sanity check on direction and magnitude, and go to
    `evaluate_embeddings` for anything you need to be sure of.

---

## Accessing and Evaluating a Fitted Field

```python exec="true" source="above" session="secondary-fields"
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
)  # (n_elements * 125, 3)

```

---

## Visualising Secondary Fields

`Mesh.plot` draws a field in two steps.  It picks the parametric locations —
`field_xi`, defaulting to a uniform grid at `default_xi_res` — then evaluates
both the geometry and the field there and hands the pair to an *artist*:

```
field_artist(plotter, locs, values, field_xi) -> None
```

`locs` is where the samples are in space, `values` is what the field says
there, and `field_xi` is the grid they came from, in case the artist wants to
reshape it.  The artist owns the whole drawing decision; nothing is added to
the scene except what it adds.  The default one draws a 3-D field as line
segments from each location, coloured by magnitude, and a 1-D field as
coloured spheres — and raises for any other field dimension, which is the
signal to write your own.

Replacing it is how you control size, glyph and colour together:

```python exec="true" source="above" session="secondary-fields"
# Draw the mesh + vector field overlaid, at a readable scale
mesh.plot(field_to_draw='vec_dir', default_xi_res=3, field_artist=arrows)

# Draw only the secondary field (without primary geometry)
mesh.plot(field_to_draw='vec_dir', draw_xyz_field=False, field_artist=arrows)

# A field is itself a MeshField, so it can also draw its own geometry
mesh['vec_dir'].plot()
```

The last of those is a different picture from the other two: a `MeshField`
drawn on its own plots its *values* as if they were coordinates, so a field of
unit normals comes out as the unit sphere rather than as anything laid over the
cube.

---

## Tips

- Use **`H3`** for smooth vector fields (fibre directions, velocities)
  that must interpolate continuously across element boundaries.
- Use **`L1`** or **`L2`** for simpler scalar fields (pressure,
  temperature) where smoothness is less critical.
- Ensure you have **more sample points than free nodal degrees of freedom**.
  If `linear_fit` raises an assertion error, add more sample points or reduce
  the basis order.
