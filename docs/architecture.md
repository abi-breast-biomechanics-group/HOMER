# Architecture

This page explains the core abstractions in HOMER and how they relate to each
other.

---

## Class Hierarchy

```
AbstractBasis  (basis_definitions.py)
  ├── H3Basis   – cubic Hermite
  ├── L1Basis   – linear Lagrange
  ├── L2Basis   – quadratic Lagrange
  ├── L3Basis   – cubic Lagrange
  └── L4Basis   – quartic Lagrange

MeshNode(dict)  (mesher.py)
  └── Physical coordinates + derivative fields

MeshElement     (mesher.py)
  └── Links nodes via tensor-product basis

MeshField       (mesher.py)
  ├── Mesh(MeshField)  – primary coordinate mesh
  │     └── fields : dict[str, MeshField]
  └── Secondary fields (fibre directions, stresses, …)
```

---

## `MeshNode` – Physical Coordinates and Derivatives

A `MeshNode` subclasses `dict` so it can carry *named* derivative arrays
alongside the spatial location `loc`.

```python
node = MeshNode(
    loc=np.array([0., 0., 1.]),   # world-space position
    du=np.zeros(3),               # ∂x/∂u tangent
    dv=np.zeros(3),               # ∂x/∂v tangent
    dudv=np.zeros(3),             # ∂²x/∂u∂v cross-derivative
)
```

The Hermite basis (`H3Basis`) requires derivative fields; the Lagrange bases
(`L1Basis`–`L4Basis`) do not.  Parameters can be *fixed* (excluded from
optimisation) via `node.fix_parameter(...)`.

---

## `MeshElement` – Tensor-Product Element

An element connects nodes through a **product of 1-D basis functions**, one
per parametric direction:

```python
# 2-D cubic-Hermite surface element: 2 × 2 = 4 nodes
elem2d = MeshElement(node_indexes=[0, 1, 2, 3],
                     basis_functions=(H3Basis, H3Basis))

# 3-D volume element with trilinear basis: 2 × 2 × 2 = 8 nodes
elem3d = MeshElement(node_indexes=[0,1,2,3,4,5,6,7],
                     basis_functions=(L1Basis, L1Basis, L1Basis))
```

The element computes the **tensor-product weight matrix** at construction time
(`BasisProductInds`) which drives all subsequent evaluations.

---

## `MeshField` – Evaluatable Field

`MeshField` holds a list of nodes and elements, and provides all evaluation
and fitting methods.  When `generate_mesh()` is called (automatically in the
constructor), it:

1. Assembles the flat parameter vector `true_param_array` from all node data.
2. Identifies the `optimisable_param_array` subset (parameters not fixed).
3. Compiles JAX evaluation functions via `_generate_eval_function()` etc.
4. Explores the mesh topology (`_explore_topology()`) to build the `bmap`
   and `topomap` for cross-element point embedding.

The `@expand_wide_evals` decorator automatically adds two convenience
variants for every `@wide_eval` method `foo`:

| Variant | Signature | Purpose |
|---|---|---|
| `foo(eles, xis)` | base | evaluate at paired (element, xi) locations |
| `foo_in_every_element(xis)` | IEE | evaluate the same xi grid in every element |
| `foo_ele_xi_pair(eles, xis)` | pair | same as base, different batching |

---

## `Mesh(MeshField)` – Primary Coordinate Mesh

`Mesh` extends `MeshField` by adding a `fields` dictionary of secondary
`MeshField` objects:

```
mesh.fields = {
    'fibre': MeshField(...),   # 3-D vector field
    'pressure': MeshField(...),  # 1-D scalar field
}
```

Secondary fields are created with `mesh.new_field(...)` and accessed via
`mesh['field_name']`.  They share the same parametric topology as the primary
mesh but can use different basis functions.

---

## Basis Hierarchy

All basis classes are frozen dataclasses inheriting from `AbstractBasis`.
They carry:

| Attribute | Description |
|---|---|
| `fn` | Evaluation function `fn(x) → (n_pts, n_basis)` |
| `deriv` | List `[fn, d1, d2, …]` of derivative functions |
| `weights` | Ordered weight names, e.g. `['x0', 'dx0', 'x1', 'dx1']` |
| `order` | Polynomial order |
| `node_locs` | Node positions in `[0, 1]` |
| `node_fields` | `DerivativeField` instance (Hermite), or `None` (Lagrange) |

---

## JAX Integration

All evaluation functions are JAX-compatible.  The key integration points are:

- `evaluate_embeddings`, `evaluate_deriv_embeddings`, `evaluate_jacobians`
  are JIT-compiled via `jax.jit` when `jax_compile=True`.
- `_xis_to_points` uses `jax.lax.fori_loop` and `jax.vmap` for
  batch-parallel point embedding.
- `topomap` is a `@jax.jit`-compiled function for cross-element boundary
  mapping.
- `jacobian_evaluator.jacobian` uses `sparsejac` (forward-mode AD with
  sparsity exploitation) to build efficient Jacobians for `scipy.optimize`.

---

## Data Flow for Fitting

```
Target data
    │
    ▼
embed_points()  →  (elem_ids, xis)
    │
    ▼
get_xi_weight_mat(elem_ids, xis)  →  W  (n_pts × n_nodes)
    │
    ▼
linear_fit(targets, W)  →  updated node parameters
    │
    ▼
generate_mesh()  →  recompile JAX functions
```

For nonlinear fitting (shape optimisation):

```
point_cloud_fit(mesh, target_pts)  →  fitting_fn, jacobian_fn
    │
    ▼
scipy.optimize.least_squares(fitting_fn, mesh.optimisable_param_array,
                              jac=jacobian_fn)
    │
    ▼
mesh.update_from_params(result.x)
```
