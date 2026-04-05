# Mixed Basis Functions

HOMER supports mixing different 1-D basis functions across parametric
directions within the same element.  This is useful when you need smooth
interpolation in some directions but only linear continuity (or fewer degrees
of freedom) in others.

---

## L2 × L2 Surface Mesh

A quadratic-Lagrange surface mesh requires 3 × 3 = 9 nodes per element.
No derivative fields are needed on the nodes.

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L2Basis

# 9-node quadratic patch (xi=0,0.5,1 in each direction)
nodes = [
    MeshNode(loc=[0, 0, 1]),   # (0,0)
    MeshNode(loc=[0, 0, 0.5]), # (0,0.5)
    MeshNode(loc=[0, 0, 0]),   # (0,1)
    MeshNode(loc=[0, 0.5, 1]), # (0.5,0)
    MeshNode(loc=[0.5, 0.5, 0.5]), # middle
    MeshNode(loc=[0, 0.5, 0]), # (0.5,1)
    MeshNode(loc=[0, 1, 1]),   # (1,0)
    MeshNode(loc=[0, 1, 0.5]), # (1,0.5)
    MeshNode(loc=[0, 1, 0]),   # (1,1)
]

element = MeshElement(
    node_indexes=list(range(9)),
    basis_functions=(L2Basis, L2Basis),
)
mesh = Mesh(nodes=nodes, elements=element)
mesh.plot()
```

---

## H3 × L2 Mixed Surface Mesh

Use `H3Basis` in the *u* direction for smooth derivatives and `L2Basis` in
the *v* direction for simpler parametric variation:

```python
from HOMER import H3Basis, L2Basis

# 2 × 3 = 6 nodes per element
# Nodes at xi_u ∈ {0, 1} and xi_v ∈ {0, 0.5, 1}
element = MeshElement(
    node_indexes=[0, 1, 2, 3, 4, 5],
    basis_functions=(H3Basis, L2Basis),
)
```

!!! note
    In a mixed element, only nodes contributing to `H3Basis` directions need
    derivative fields.  Here, `du` is required but `dv` is not.

---

## Choosing the Right Basis

| Requirement | Recommended basis |
|---|---|
| C¹ smooth geometry, shape optimisation | `H3Basis` |
| Simple coarse mesh before rebasing | `L1Basis` |
| Mid-order accuracy, fewer DoF than H3 | `L2Basis` or `L3Basis` |
| High-accuracy Lagrange interpolation | `L4Basis` |

---

## Rebasing Between Bases

Any mesh can be converted to a different basis with `rebase()`:

```python
# Start with a coarse linear mesh
linear_mesh = Mesh(nodes=nodes, elements=MeshElement(
    node_indexes=list(range(4)),
    basis_functions=(L1Basis, L1Basis),
))

# Convert to cubic Hermite
from HOMER import H3Basis
smooth_mesh = linear_mesh.rebase([H3Basis, H3Basis])
```

See the [Basis conversion guide](rebase.md) for full details.
