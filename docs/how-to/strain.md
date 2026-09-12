# Strain Evaluation

HOMER can compute the **Green-Lagrange strain tensor** between two meshes that
share the same topology but differ in their nodal coordinates (reference and
deformed configurations).

---

## Theory

Given a reference mesh **X** and a deformed mesh **x**, the deformation
gradient at parametric location ξ is:

**F**(ξ) = **J**_x(ξ) · **J**_X(ξ)⁻¹

where **J**(ξ) = ∂position/∂ξ is the Jacobian returned by
`evaluate_jacobians`: rows are physical directions and columns parametric
ones, so `J[i, j]` is ∂x_i/∂ξ_j.  A uniform stretch of 3 along x therefore
gives **F** = diag(3, 1, 1).

The Green-Lagrange strain tensor is then:

**E** = (**F**ᵀ **F** − **I**) / 2

---

## Basic Usage

```python
from copy import deepcopy
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, L1, H3

# 1. Build a reference mesh
nodes = [MeshNode(loc=[x,y,z])
         for x in [0,1] for y in [0,1] for z in [0,1]]
element = MeshElement(node_indexes=list(range(8)),
                      basis_functions=(L1, L1, L1))
mesh_ref = Mesh(nodes=nodes, elements=element).rebase([H3]*3)

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

A 2-D surface in 3-D has a `(3, 2)` Jacobian, which has no determinant and no
inverse, so there is no deformation gradient to form.  Calling
`evaluate_strain` on one without a `coord_function` says so:

```
ValueError: Strain tensor on manifold mesh requires a coord function to
provide a meaninful basis
```

`surface_normal_mapping` is the ready-made answer.  It prepends the unit
surface normal as a third Jacobian column, giving a square frame in which the
out-of-plane direction carries no stretch:

```python
from HOMER.geometry import basic_surface
from HOMER.utils import surface_normal_mapping

strains = mesh_ref.evaluate_strain(elem_ids, xis, mesh_def,
                                   coord_function=surface_normal_mapping)
```

Stretch a flat patch by 1.5 along one in-plane axis and the centre comes back
with `E_yy = (1.5**2 - 1) / 2 = 0.625` and zeros elsewhere, which is the
analytic Green-Lagrange answer.

A `coord_function` is any callable `(mesh, eles, xis, Jmats) -> Jmats`, so
pass your own when you want a different frame — fibre-aligned axes, say.

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
