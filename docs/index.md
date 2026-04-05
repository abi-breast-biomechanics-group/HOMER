# HOMER – High Order Mesh Embeddings and Refinement

HOMER is a Python library for constructing, fitting, evaluating, and visualising
**high-order finite-element meshes** using JAX for fast, differentiable
computations.  It supports arbitrary tensor-product basis functions
(cubic Hermite, linear/quadratic/cubic/quartic Lagrange) in 2-D and 3-D, and
provides tools for:

- Building manifold surface meshes and volume meshes
- Fitting meshes to point clouds via nonlinear least squares
- Embedding arbitrary points into the parametric coordinates of a mesh
- Refining mesh resolution while preserving the underlying geometry
- Converting between basis functions (rebasing)
- Storing and evaluating secondary vector/scalar fields over the mesh topology
- Computing Green-Lagrange strain tensors
- Saving and loading meshes to/from JSON

---

## Quick Example

```python
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3Basis

# 1. Create four corner nodes for a flat 2-D patch
node0 = MeshNode(loc=np.array([0., 0., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node1 = MeshNode(loc=np.array([1., 0., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node2 = MeshNode(loc=np.array([0., 1., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node3 = MeshNode(loc=np.array([1., 1., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

# 2. Link the nodes through a bicubic-Hermite element
element = MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=(H3Basis, H3Basis))

# 3. Create the mesh
mesh = Mesh(nodes=[node0, node1, node2, node3], elements=element)

# 4. Evaluate the surface at a 10×10 grid
xis = mesh.xi_grid(10)                          # (100, 2)
pts = mesh.evaluate_embeddings_in_every_element(xis)  # (100, 3)

# 5. Visualise
mesh.plot()
```

---

## Getting Started

### Installation

```bash
pip install HOMER
```

For the optional documentation extras:

```bash
pip install "HOMER[docs]"
```

### Key Concepts

| Term | Description |
|---|---|
| `MeshNode` | Physical location + Hermite derivative vectors |
| `MeshElement` | Connects nodes via tensor-product basis functions |
| `MeshField` | Collection of nodes + elements; evaluates/fits a field |
| `Mesh` | Primary coordinate field that can carry secondary fields |
| Basis | 1-D interpolation building block: `H3`, `L1`–`L4` |
| xi | Parametric coordinates in [0, 1]ⁿ |

### Supported Basis Functions

| Class | Type | Nodes/dir | C⁰ | Node fields |
|---|---|---|---|---|
| `H3Basis` | Cubic Hermite | 2 | C¹ | `du`, `dv`, … |
| `L1Basis` | Linear Lagrange | 2 | C⁰ | – |
| `L2Basis` | Quadratic Lagrange | 3 | C⁰ | – |
| `L3Basis` | Cubic Lagrange | 4 | C⁰ | – |
| `L4Basis` | Quartic Lagrange | 5 | C⁰ | – |

---

## Workflow

The core workflow demonstrated in the test suite is:

1. **Create nodes** – instantiate `MeshNode` objects with physical coordinates
   and (for Hermite bases) derivative vectors.
2. **Create elements** – combine nodes with a tuple of basis classes.
3. **Build the mesh** – pass nodes and elements to `Mesh(...)`.
4. **Evaluate** – call `evaluate_embeddings()`, `evaluate_jacobians()`, etc.
5. **Fit** – use `linear_fit()` or `point_cloud_fit()` to update node parameters.
6. **Refine** – call `mesh.refine(2)` to subdivide elements.
7. **Save / load** – `mesh.save('path.json')` / `load_mesh('path.json')`.

See the [How-To Guides](how-to/3d-meshes.md) for detailed walk-throughs.

---

## Next Steps

- [Architecture overview](architecture.md) – understand the class hierarchy.
- [How-To Guides](how-to/3d-meshes.md) – feature-specific recipes.
- [API Reference](api/mesher.md) – full docstring reference.
