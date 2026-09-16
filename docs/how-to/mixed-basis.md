# Mixed Basis Functions

HOMER supports mixing different 1-D basis functions across parametric
directions within the same element.  This is useful when you need smooth
interpolation in some directions but only linear continuity (or fewer degrees
of freedom) in others.

---

## Combining bases

A basis is a *value* — `H3`, `L1`, … are frozen instances, not classes — and
an element's parametric directions are built out of them with arithmetic.  `*`
joins directions, the operator nearest the outer product the element actually
takes, and `**` is the tensor power.  Against an `int`, `*` repeats instead:

```python exec="true" source="above" session="mixed-basis"
from HOMER import H3, L1

H3 ** 3            # tricubic-Hermite volume
H3**2 * L1         # Hermite surface extruded linearly
H3 * H3 * L1       # the same shape, written out
(H3 * L1)**2       # H3, L1, H3, L1
H3 * 3             # a spelling of H3 ** 3
```

The result is a `BasisGroup`, a `tuple` subclass, so a plain tuple or list of
bases — `(H3, H3, L1)` — means the same thing and is accepted everywhere.  The
algebra is the spelling used throughout these guides.

---

## L2 × L2 Surface Mesh

A quadratic-Lagrange surface mesh requires 3 × 3 = 9 nodes per element.
No derivative fields are needed on the nodes.

Nodes are listed with `xi_u` varying fastest, then `xi_v` — the ordering
described under [Node indexing](node-indexing.md#node-ordering-within-an-element).

```python exec="true" source="above" session="mixed-basis"
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L2

# 9-node quadratic patch, xi_u and xi_v each at 0, 0.5, 1
nodes = [
    MeshNode(loc=[0.0, 0.0, 0.0]),   # (u, v) = (0,   0)
    MeshNode(loc=[0.5, 0.0, 0.0]),   #          (0.5, 0)
    MeshNode(loc=[1.0, 0.0, 0.0]),   #          (1,   0)
    MeshNode(loc=[0.0, 0.5, 0.0]),   #          (0,   0.5)
    MeshNode(loc=[0.5, 0.5, 0.1]),   #          (0.5, 0.5)  raised centre
    MeshNode(loc=[1.0, 0.5, 0.0]),   #          (1,   0.5)
    MeshNode(loc=[0.0, 1.0, 0.0]),   #          (0,   1)
    MeshNode(loc=[0.5, 1.0, 0.0]),   #          (0.5, 1)
    MeshNode(loc=[1.0, 1.0, 0.0]),   #          (1,   1)
]

element = MeshElement(node_indexes=list(range(9)), basis_functions=L2**2)
l2_mesh = Mesh(nodes=nodes, elements=element)
l2_mesh.plot()
```

---

## H3 × L2 Mixed Surface Mesh

Use `H3` in the xi_0 direction for smooth derivatives and `L2` in
the xi_1 direction for simpler parametric variation:

```python exec="true" source="above" session="mixed-basis"
from HOMER import H3

# 2 × 3 = 6 nodes per element
# Nodes at xi_u ∈ {0, 1} and xi_v ∈ {0, 0.5, 1}
element = MeshElement(node_indexes=list(range(6)), basis_functions=H3 * L2)
```

`H3 * L2` is Hermite in xi_0 and quadratic Lagrange in xi_1 — the directions
are read left to right, in the order the element takes its product.

---

## Choosing the Right Basis

| Requirement | Recommended basis |
|---|---|
| C¹ smooth geometry, shape optimisation | `H3` |
| C² smooth geometry, shape optimisation | `B3` |
| Simple coarse mesh before rebasing | `L1` |
| Mid-order accuracy, fewer DoF than H3 | `L2` or `L3` |
| High-accuracy Lagrange interpolation | `L4` |

---

## Rebasing Between Bases

Any mesh can be converted to a different basis with `rebase()` — here the
quadratic patch built above, taken to a cubic B-spline control net:

```python exec="true" source="above" session="mixed-basis"
from HOMER import B3

smooth_mesh = l2_mesh.rebase(B3**2)
smooth_mesh.plot()
```

The nodes have moved off the surface: `B3` is not interpolatory, so its
parameters are control points shared with the neighbouring elements rather
than positions on the patch.

This can be a convienient way to manipulate meshes, exploiting different properties of mesh basis.
In particular, it's very useful for building B3 meshes.
However, rebase is not exact, typically converging to 1e-6.
As such, drift could occur over millions of rebase operations.

See the [Basis conversion guide](rebase.md) for full details.
