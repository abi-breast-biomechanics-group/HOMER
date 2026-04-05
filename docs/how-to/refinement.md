# Mesh Refinement

Refinement subdivides every element into smaller sub-elements, increasing the
spatial resolution of the mesh while preserving the underlying high-order
geometry.

---

## Uniform Refinement

`mesh.refine(refinement_factor=n)` splits each parametric direction into
`n` sub-intervals, creating `n ** ndim` sub-elements per original element.

```python
from HOMER import Mesh, MeshNode, MeshElement, H3Basis
import numpy as np

# 1. Build a coarse 3-D mesh
nodes = [MeshNode(loc=[x,y,z], du=np.zeros(3), dv=np.zeros(3), dw=np.zeros(3),
                  dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3), dudvdw=np.zeros(3))
         for x in [0,1] for y in [0,1] for z in [0,1]]
element = MeshElement(node_indexes=list(range(8)),
                      basis_functions=(H3Basis, H3Basis, H3Basis))
mesh = Mesh(nodes=nodes, elements=element)

# 2. Refine: each element is subdivided into 2×2×2 = 8 sub-elements
mesh.refine(refinement_factor=2)
print(f"Elements after refinement: {len(mesh.elements)}")  # → 8
```

---

## Non-Uniform Refinement

Provide a tuple of xi breakpoint arrays to place element boundaries at
specific parametric locations:

```python
# Refine with 2 sub-intervals in u, 3 in v, and 2 in w
mesh.refine(by_xi_refinement=(
    np.array([0, 0.5, 1.0]),   # u: 2 sub-intervals
    np.array([0, 1/3, 2/3, 1.0]),  # v: 3 sub-intervals
    np.array([0, 0.5, 1.0]),   # w: 2 sub-intervals
))
```

!!! warning
    The two parameters `refinement_factor` and `by_xi_refinement` are
    mutually exclusive.  Providing both raises an `AssertionError`.

---

## Refining a `Mesh` with Secondary Fields

When the mesh has secondary fields, `Mesh.refine()` automatically refines all
fields simultaneously:

```python
# mesh has a secondary field 'fibre'
mesh.new_field('fibre', field_dimension=3, new_basis=[H3Basis]*3,
               field_locs=data_pts, field_values=fibre_vectors)

# Refine both the geometry and the 'fibre' field
mesh.refine(refinement_factor=2)

# Both mesh geometry and mesh['fibre'] are now at 2× resolution
```

---

## Visualising Before and After

```python
import pyvista as pv

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0)
mesh_before.plot(s)
s.subplot(0, 1)
mesh_after.plot(s)
s.show()
```
