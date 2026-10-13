# Topology Mapping

HOMER's topology map (`topomap`) allows seamless evaluation and point
embedding across element boundaries.  It is built automatically when
`generate_mesh()` is called.

---

## What is the topology map?

When a parametric coordinate `xi` exceeds the [0, 1] boundary of an element,
`topomap(elem, xi)` looks up the neighbouring element and maps `xi` into the
neighbour's parametric space.  This enables:

- **Point embedding** – `embed_points()` can follow trajectories across
  multiple elements without manual boundary handling.
- **Cross-element derivatives** – derivative computations remain valid near
  element boundaries.

---

## Topology structure

After `generate_mesh()`, the following topology attributes are available:

| Attribute | Type | Description |
|---|---|---|
| `mesh.faces` | `list[tuple]` | External faces: `(elem_id, dim, side)` |
| `mesh.bmap` | `dict` | `(elem, dim, side) → [(neighbor, dim, side), rel_dirs]` |
| `mesh.hanging` | `dict` | Hanging faces: `(elem, dim, side) → (coarse elem, dim, side)` |
| `mesh.topomap` | `Callable` | JAX-JIT function `(elem, xi) → (elem', xi', valid)` |

---

## Accessing external faces

```python exec="true" source="above" session="topology"
import numpy as np

from HOMER import L1
from HOMER.geometry import cube

# a 2x2x2 block of trilinear elements, as tests/test_topology.py builds it
mesh = cube(scale=1, centre=np.zeros(3), basis=L1**3)
mesh.refine(2)
mesh.plot()

# All external faces of a 3-D mesh
faces = mesh.get_faces()
# Each face is a tuple: (elem_index, parametric_dim, 0_or_1)
# (elem_index, -1, -1) indicates a 2-D manifold element

print(f"{len(faces)} external faces")
for face in faces[:4]:
    elem_id, dim, side = face
    print(f"Element {elem_id}, face at xi_{dim} = {side}")
```

---

## Selecting the nodes on one boundary

`get_xi_surface_nodes(xi_dim, bound_val)` asks the same question the other way
round: which elements have *no* neighbour at one end of one parametric
direction, and which nodes have basis support on the face they expose.  That is
the selection you want for a boundary condition, a fit that is free to move
only on one surface, or a landmark set.

It reads the answer off the topology and the 1-D basis rather than off the
coordinates, so it is exact on a deformed mesh and picks up a B-spline's
off-surface control points as well as an interpolatory basis's face nodes.

```python exec="true" source="above" session="topology"
import pyvista as pv

# the xi_0 = 1 boundary of the block
face_elements, face_nodes = mesh.get_xi_surface_nodes(0, 1)
print(f"{len(face_elements)} elements expose it, across {len(face_nodes)} nodes")

s = pv.Plotter()
mesh.plot(s, node_size=4)                    # every node, drawn small
s.add_points(np.array([mesh.nodes[i].loc for i in face_nodes]),
             color='b', point_size=10, render_points_as_spheres=True)
s.show()
```

The blue spheres are the selected layer; the small red markers are every other
node of the mesh.

---

## Using the topology map directly

The `topomap` function is a JAX-JIT-compiled function:

```python exec="true" source="above" session="topology"
import jax.numpy as jnp

elem = jnp.array(0)
xi   = jnp.array([1.05, 0.5, 0.5])  # slightly outside element 0

new_elem, new_xi, valid = mesh.topomap(elem, xi)
print(f"elem {elem} xi {xi} -> elem {new_elem} xi {new_xi} (valid={valid})")
# new_elem: the neighbouring element
# new_xi:   xi mapped into the neighbour's parameter space
# valid:    True if a valid neighbour was found
```

---

## Checking mesh connectivity

`bmap` is a dictionary mapping element face identifiers to their neighbours:

```python exec="true" source="above" session="topology"
# Key: (element_index, parametric_dim, side)
# Value: [(neighbour_index, dim, side), rel_dirs_bool_array]
print(f"{len(mesh.bmap)} internal faces")
for key, (neighbour, rel_dirs) in list(mesh.bmap.items())[:4]:
    elem, dim, side = key
    n_elem, n_dim, n_side = neighbour
    print(f"Elem {elem} face (dim={dim}, side={side}) "
          f"→ Elem {n_elem} face (dim={n_dim}, side={n_side})")
```

`rel_dirs` is a boolean array of length `ndim` indicating whether the tangent
directions of the two faces are aligned (``True``) or anti-aligned (``False``).

---

## Hanging faces

After [`refine_elements`](refinement.md#refining-only-some-elements), a child
can meet an unrefined element across a face the two do not share node for
node.  `mesh.hanging` pairs each such child face with the coarse face it lies
against.  `topomap` crosses them both ways, so `embed_points` follows a point
from a coarse element into the children beside it and back:

```python exec="true" source="above" session="topology"
local = cube(scale=1, centre=np.zeros(3), basis=L1**3)
local.refine(2)
local.refine_elements([0])

(child, dim, side), (coarse, _, _) = next(iter(local.hanging.items()))
xi = np.full(3, 0.5)
xi[dim] = 1.05 if side == 1 else -0.05      # just outside the child's face
across, xi_across, valid = local.topomap(child, jnp.asarray(xi))
print(f"child {child} -> element {int(across)} (the coarse one is {coarse}), valid={bool(valid)}")
```

The crossing is read off the elements' parameter maps rather than the
geometry: the child's corners are drawn from the coarse face's parameters,
which places them on that face exactly.

---

## Notes

- Two faces are connected when the same parameters shape them both, read off
  the bases and the elements' parameter maps, so it is exact and independent
  of the geometry.
- Multi-element junctions (more than 2 elements meeting at a face) are not
  supported, and raise a `ValueError` when the mesh is generated.
- For 2-D manifold meshes, each element is its own "face" and `topomap` still
  handles cross-element boundary embedding.
