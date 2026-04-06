# Topology Mapping

HOMER's topology map (`topomap`) allows seamless evaluation and point
embedding across element boundaries.  It is built automatically when
`generate_mesh()` is called.

---

## What Is the Topology Map?

When a parametric coordinate `xi` exceeds the [0, 1] boundary of an element,
`topomap(elem, xi)` looks up the neighbouring element and maps `xi` into the
neighbour's parametric space.  This enables:

- **Point embedding** – `embed_points()` can follow trajectories across
  multiple elements without manual boundary handling.
- **Cross-element derivatives** – derivative computations remain valid near
  element boundaries.

---

## Topology Structure

After `generate_mesh()`, the following topology attributes are available:

| Attribute | Type | Description |
|---|---|---|
| `mesh.faces` | `list[tuple]` | External faces: `(elem_id, dim, side)` |
| `mesh.bmap` | `dict` | `(elem, dim, side) → [(neighbor, dim, side), rel_dirs]` |
| `mesh.topomap` | `Callable` | JAX-JIT function `(elem, xi) → (elem', xi', valid)` |

---

## Accessing External Faces

```python
# All external faces of a 3-D mesh
faces = mesh.get_faces()
# Each face is a tuple: (elem_index, parametric_dim, 0_or_1)
# (elem_index, -1, -1) indicates a 2-D manifold element

for face in faces:
    elem_id, dim, side = face
    print(f"Element {elem_id}, face at xi_{dim} = {side}")
```

---

## Using the Topology Map Directly

The `topomap` function is a JAX-JIT-compiled function:

```python
import jax.numpy as jnp

elem = jnp.array(0)
xi   = jnp.array([1.05, 0.5, 0.5])  # slightly outside element 0

new_elem, new_xi, valid = mesh.topomap(elem, xi)
# new_elem: the neighbouring element
# new_xi:   xi mapped into the neighbour's parameter space
# valid:    True if a valid neighbour was found
```

---

## Checking Mesh Connectivity

`bmap` is a dictionary mapping element face identifiers to their neighbours:

```python
# Key: (element_index, parametric_dim, side)
# Value: [(neighbour_index, dim, side), rel_dirs_bool_array]
for key, (neighbour, rel_dirs) in mesh.bmap.items():
    elem, dim, side = key
    n_elem, n_dim, n_side = neighbour
    print(f"Elem {elem} face (dim={dim}, side={side}) "
          f"→ Elem {n_elem} face (dim={n_dim}, side={n_side})")
```

`rel_dirs` is a boolean array of length `ndim` indicating whether the tangent
directions of the two faces are aligned (``True``) or anti-aligned (``False``).

---

## Notes

- The topology exploration uses spatial hashing: two faces are considered
  connected if their midpoint coordinates are equal to 10 decimal places.
  As the same parameters define this midpoint, they are definitionally equal.
  This rounding factor can be changed for very small meshes, but maybe just consider a change of scale and make your computer happier.
- Multi-element junctions (more than 2 elements meeting at a face) are not
  supported and raise a `ValueError`.
- For 2-D manifold meshes, each element is its own "face" and `topomap` still
  handles cross-element boundary embedding.
