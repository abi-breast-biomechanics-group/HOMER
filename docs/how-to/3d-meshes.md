# Working with 3-D Meshes

This guide covers creating 3-D volume meshes and collapsed (degenerate)
hexahedral meshes.

---

## H3 × H3 × H3 Volume Mesh

A tri-cubic-Hermite volume mesh requires 8 corner nodes, each carrying seven
derivative vectors.

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3Basis

# Eight corner nodes of a unit cube
# Every node needs: du, dv, dw, dudv, dudw, dvdw, dudvdw
def corner_node(loc, dw):
    return MeshNode(
        loc=np.array(loc), 
        du=np.zeros(3), dv=np.zeros(3), dw=np.array(dw),
        dudv=np.zeros(3), dudw=np.zeros(3),
        dvdw=np.zeros(3), dudvdw=np.zeros(3),
    )

nodes = [
    corner_node([0,0,1], [2,-0.5, 0.5]),
    corner_node([0,0,0], [0, 0,   0  ]),
    corner_node([0,1,1], [0, 0,   0  ]),
    corner_node([0,1,0], [2, 0.5,-0.5]),
    corner_node([1,0,1], [1,-0.5, 0.5]),
    corner_node([1,0,0], [1,-0.5,-0.5]),
    corner_node([1,1,1], [1, 0.5, 0.5]),
    corner_node([1,1,0], [1, 0.5,-0.5]),
]

element = MeshElement(
    node_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
    basis_functions=(H3Basis, H3Basis, H3Basis),
)

mesh = Mesh(nodes=nodes, elements=element)
mesh.plot()
```

---

## H3 × H3 × L1 Mixed Volume Mesh

Use `L1Basis` in the *w* direction for a mesh that is linear along one axis
but smooth in the other two:

```python
from HOMER import L1Basis

element = MeshElement(
    node_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
    basis_functions=(H3Basis, H3Basis, L1Basis),
)
```

!!! note
    Nodes used with `L1Basis` in a given direction do **not** need a derivative field.
    By convention, we use `du`for the first derivative dimension, `dv` the second, and `dw` the third.
    An L1H3H3 mesh will still have the derivative fields `du`, `dv`, `dudv`.

---

## Collapsed (Degenerate) Hexahedral Mesh

A collapsed element shares two or more corner nodes to create wedge or
pyramid shapes:

```python
# 6-node wedge: share node 0 and node 4 (the "apex")
wedge_element = MeshElement(
    node_indexes=[0, 1, 2, 3, 0, 1, 2, 3],  # apex collapsed
    basis_functions=(H3Basis, H3Basis, H3Basis),
)
```

Collapsed elements are useful for certain geometries with a shared central point.

---

## Building from a Trilinear Base Mesh

A convenient workflow is to create a coarse `L1` × `L1` × `L1` mesh, then
rebase it to the desired configuration (e.g. `H3`).
This is especially useful for `H3` meshes, which otherwise have large numbers of non-zero derivatives:

```python
from HOMER.geometry import cube

# Creates a unit-cube mesh in H3×H3×H3 automatically
mesh = cube(scale=1.0)

# Or manually:
from HOMER import L1Basis
seed = Mesh(nodes=nodes, elements=MeshElement(
    node_indexes=list(range(8)),
    basis_functions=(L1Basis, L1Basis, L1Basis),
))
mesh = seed.rebase([H3Basis, H3Basis, H3Basis])
```

---

## Evaluating Mesh Quantities

```python
import numpy as np

# 5×5×5 interior grid per element
xis = mesh.xi_grid(5)                              # (125, 3)
pts = mesh.evaluate_embeddings_in_every_element(xis)  # (125, 3)

# Jacobian at Gauss points
gp, gw = mesh.gauss_grid([4, 4, 4])               # 64 Gauss points
Jmats = mesh.evaluate_jacobians_in_every_element(gp)  # (64, 3, 3)

# Volume
vol = mesh.get_volume()
print(f"Volume: {vol:.4f}")
```
