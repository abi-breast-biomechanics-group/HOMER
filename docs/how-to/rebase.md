# Basis Conversion (Rebase)

`rebase()` converts a mesh from one set of basis functions to another by:

1. Computing the new node positions from the current geometry.
2. Sampling a dense xi grid on the current mesh.
3. Linearly fitting the new nodal parameters to match the sampled geometry.

This is the recommended way to build high-order meshes: start with a coarse
Lagrange mesh that is easy to set up manually, then rebase to a higher-order
or Hermite basis.

---

## Trilinear → Cubic Hermite Conversion

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L1Basis, H3Basis

# 1. Create a coarse trilinear mesh (L1 × L1 × L1)
nodes = [MeshNode(loc=[x, y, z])
         for x in [0., 1.] for y in [0., 1.] for z in [0., 1.]]
element = MeshElement(node_indexes=list(range(8)),
                      basis_functions=(L1Basis, L1Basis, L1Basis))
seed = Mesh(nodes=nodes, elements=element)

# 2. Rebase to cubic Hermite
mesh = seed.rebase([H3Basis, H3Basis, H3Basis])

# The resulting mesh has the same shape but smooth H3 interpolation
mesh.plot()
```

---

## Converting a 2-D Manifold Mesh

The same workflow applies to 2-D surface meshes:

```python
from HOMER import L1Basis, H3Basis

# Linear seed
seed_2d = Mesh(nodes=four_nodes, elements=MeshElement(
    node_indexes=[0,1,2,3], basis_functions=(L1Basis, L1Basis)))

# Convert to cubic Hermite surface
smooth_2d = seed_2d.rebase([H3Basis, H3Basis])
```

---

## Rebase to Lagrange Basis

You can also rebase from Hermite to Lagrange (e.g. for export or
compatibility with other solvers):

```python
from HOMER import L3Basis

lagrange_mesh = hermite_mesh.rebase([L3Basis, L3Basis, L3Basis])
```

---

## Resolution Control

The `res` parameter of `rebase()` controls how many xi samples are used for
the linear fit.  Increase it for better accuracy when rebasing to a
significantly different basis:

```python
mesh = seed.rebase([H3Basis]*3, res=20)  # default res=10
```

---

## Notes

- `rebase()` always returns a **new** `MeshField` object.  The original mesh
  is not modified.
- The returned object is a `MeshField`, not a `Mesh`.  If you need a `Mesh`
  with secondary fields after rebasing, wrap it:
  ```python
  from HOMER import Mesh
  from copy import deepcopy
  rebased = seed.rebase([H3Basis]*3)
  # Use the returned MeshField directly, or re-assign nodes/elements
  ```
