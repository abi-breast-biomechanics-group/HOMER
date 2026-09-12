# Mixed Basis Functions

HOMER supports mixing different 1-D basis functions across parametric
directions within the same element.  This is useful when you need smooth
interpolation in some directions but only linear continuity (or fewer degrees
of freedom) in others.

---

## L2 × L2 Surface Mesh

A quadratic-Lagrange surface mesh requires 3 × 3 = 9 nodes per element.
No derivative fields are needed on the nodes.

Nodes are listed with `xi_u` varying fastest, then `xi_v` — the ordering
described under [Node indexing](node-indexing.md#node-ordering-within-an-element).

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L2

# 9-node quadratic patch, xi_u and xi_v each at 0, 0.5, 1
nodes = [
    MeshNode(loc=[0.0, 0.0, 0.0]),   # (u, v) = (0,   0)
    MeshNode(loc=[0.5, 0.0, 0.0]),   #          (0.5, 0)
    MeshNode(loc=[1.0, 0.0, 0.0]),   #          (1,   0)
    MeshNode(loc=[0.0, 0.5, 0.0]),   #          (0,   0.5)
    MeshNode(loc=[0.5, 0.5, 0.3]),   #          (0.5, 0.5)  raised centre
    MeshNode(loc=[1.0, 0.5, 0.0]),   #          (1,   0.5)
    MeshNode(loc=[0.0, 1.0, 0.0]),   #          (0,   1)
    MeshNode(loc=[0.5, 1.0, 0.0]),   #          (0.5, 1)
    MeshNode(loc=[1.0, 1.0, 0.0]),   #          (1,   1)
]

element = MeshElement(
    node_indexes=list(range(9)),
    basis_functions=(L2, L2),
)
mesh = Mesh(nodes=nodes, elements=element)
mesh.plot()
```

---

## H3 × L2 Mixed Surface Mesh

Use `H3` in the xi_0 direction for smooth derivatives and `L2` in
the xi_1 direction for simpler parametric variation:

```python
from HOMER import H3, L2

# 2 × 3 = 6 nodes per element
# Nodes at xi_u ∈ {0, 1} and xi_v ∈ {0, 0.5, 1}
element = MeshElement(
    node_indexes=[0, 1, 2, 3, 4, 5],
    basis_functions=H3 * L2,
)
```

`*` joins the directions in order, so `H3 * L2` is Hermite in xi_0 and
quadratic Lagrange in xi_1.  A plain tuple — `(H3, L2)` — means the same
thing and is still accepted everywhere.

---

## Choosing the Right Basis

| Requirement | Recommended basis |
|---|---|
| C¹ smooth geometry, shape optimisation | `H3` |
| Simple coarse mesh before rebasing | `L1` |
| Mid-order accuracy, fewer DoF than H3 | `L2` or `L3` |
| High-accuracy Lagrange interpolation | `L4` |

---

## Rebasing Between Bases

Any mesh can be converted to a different basis with `rebase()`:

```python
# Start with a coarse linear mesh
linear_mesh = Mesh(nodes=nodes, elements=MeshElement(
    node_indexes=list(range(4)),
    basis_functions=L1**2,
))

# Convert to cubic Hermite
from HOMER import H3
smooth_mesh = linear_mesh.rebase(H3**2)
```

See the [Basis conversion guide](rebase.md) for full details.
