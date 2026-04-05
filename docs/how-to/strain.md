# Strain Evaluation

HOMER can compute the **Green-Lagrange strain tensor** between two meshes that
share the same topology but differ in their nodal coordinates (reference and
deformed configurations).

---

## Theory

Given a reference mesh **X** and a deformed mesh **x**, the deformation
gradient at parametric location ξ is:

**F**(ξ) = **J**_X(ξ)⁻¹ · **J**_x(ξ)

where **J**(ξ) = ∂position/∂ξ is the Jacobian matrix.

The Green-Lagrange strain tensor is then:

**E** = (**F**ᵀ **F** − **I**) / 2

---

## Basic Usage

```python
from copy import deepcopy
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L1Basis, H3Basis

# 1. Build a reference mesh
nodes = [MeshNode(loc=[x,y,z])
         for x in [0,1] for y in [0,1] for z in [0,1]]
element = MeshElement(node_indexes=list(range(8)),
                      basis_functions=(L1Basis, L1Basis, L1Basis))
mesh_ref = Mesh(nodes=nodes, elements=element).rebase([H3Basis]*3)

# 2. Copy and apply a deformation
mesh_def = deepcopy(mesh_ref)

grid = mesh_ref.xi_grid(res=10)
elem_ids = np.zeros(grid.shape[0], dtype=int)

ref_pts = mesh_ref.evaluate_embeddings_in_every_element(grid)

# Apply a shear + quadratic deformation
def_pts = np.stack([
    ref_pts[:, 0],
    ref_pts[:, 1] + ref_pts[:, 0] * 0.1,   # shear in y
    ref_pts[:, 2] ** 2,                     # nonlinear z
], axis=1)

# Fit the deformed mesh
W = mesh_def.get_xi_weight_mat(elem_ids, grid)
mesh_def.linear_fit(def_pts, weight_mat=W)

# 3. Evaluate strain
eval_grid = mesh_ref.xi_grid(res=5)
strains = mesh_ref.evaluate_strain_in_every_element(eval_grid, mesh_def)
# strains: shape (n_pts, 3, 3)

print("E_zz at xi=0.5:", strains[len(strains)//2, 2, 2])
```

---

## Strain on a 2-D Manifold Mesh

For 2-D surface meshes, the 3-D strain tensor is not well-defined without
specifying a local coordinate frame.  Provide a `coord_function`:

```python
def local_frame(mesh, eles, xis, Jmats):
    """Project Jacobians into a local (tangent, normal) frame."""
    ...
    return projected_Jmats

strains = mesh_ref.evaluate_strain(elem_ids, xis, mesh_def,
                                   coord_function=local_frame)
```

---

## Returning the Deformation Gradient

Pass `return_F=True` to get **F** instead of **E**:

```python
F = mesh_ref.evaluate_strain_in_every_element(eval_grid, mesh_def,
                                              return_F=True)
# F: shape (n_pts, ndim, ndim)
```

---

## Visualising Strain

```python
import pyvista as pv

slocs  = mesh_ref.evaluate_embeddings(0, eval_grid)
svecs  = mesh_def.evaluate_jacobians(0, eval_grid)

s = pv.Plotter()
mesh_ref.plot(s, mesh_opacity=0.1)
mesh_def.plot(s, node_colour='g', mesh_opacity=0.1)

# Draw strain vectors along each principal axis
s.add_arrows(slocs, strains[:, 0, 0][:, None] * svecs[:, 0], color='r')
s.add_arrows(slocs, strains[:, 1, 1][:, None] * svecs[:, 1], color='g')
s.add_arrows(slocs, strains[:, 2, 2][:, None] * svecs[:, 2], color='b')
s.show()
```
