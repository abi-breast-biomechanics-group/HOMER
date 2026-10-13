# Architecture

This page explains the core abstractions in HOMER and how they relate to each
other.

---

## Class hierarchy

```
Basis  (basis_definitions.py)      – one frozen instance per 1-D basis
  ├── H3   – cubic Hermite
  ├── L1   – linear Lagrange
  ├── L2   – quadratic Lagrange
  ├── L3   – cubic Lagrange
  ├── L4   – quartic Lagrange
  └── B3   – cubic B-spline
BasisGroup(tuple)                  – the bases of one element, one per direction

MeshNode(dict)  (mesh/node.py)
  └── Physical coordinates + derivative fields

MeshElement     (mesh/element.py)
  └── Links nodes via tensor-product basis

MeshField       (mesh/field.py)
  ├── Mesh(MeshField)  – primary coordinate mesh
  │     └── fields : dict[str, MeshField]
  └── Secondary fields (fibre directions, stresses, …)
```

`MeshField` owns the state — nodes, elements, the parameter vector, the JAX
closures compiled from them — and the lifecycle that keeps those in step
(`generate_mesh`).  What a field *does* lives in a module per concern, whose
functions take the field as their first argument and are bound into the class
body by `mesh/field.py`:

```
HOMER/mesh/
    node.py  element.py  field.py  mesh.py     the classes
    evaluation.py    evaluating the field, its derivatives, embedded points
    parameters.py    the parameter vector, and methods for fast linear parameter fits
    topology.py      shared nodes, faces, surfaces, colouring
    refinement.py    refine and rebase, and the fixed-parameter transfer
    plotting.py      drawing; the only module that touches PyVista
    element_eval.py  per-element evaluation kernels and quadrature rules
    reordering.py    node renumbering, after an operation rebuilds the list
```

`reordering.py` is the exception to the binding above: renumbering is done
*to* a field rather than *by* one, so `reorder_nodes` stays a plain function
and is called as `reorder_nodes(mesh)`.

Binding rather than inheriting is deliberate: `@expand_wide_evals` reads
`vars(cls)`, so a method reached through a base class would be invisible to it
and the generated `*_in_every_element` / `*_ele_xi_pair` variants would
silently disappear.  Two consequences for anything added to those modules: the
first parameter is `self`, and a zero-argument `super()` will not work (there
is no `__class__` cell outside a class body) — call the sibling function
directly instead.

`HOMER/mesher.py` remains as a re-export shim so `from HOMER.mesher import
Mesh` keeps working.

---

## `MeshNode` – physical coordinates and derivatives

A `MeshNode` subclasses `dict` so it can carry *named* derivative arrays
alongside the spatial location `loc`.

```python exec="true" source="above" session="architecture"
import numpy as np

from HOMER import MeshNode

node = MeshNode(
    loc=np.array([0., 0., 1.]),   # world-space position
    du=np.zeros(3),               # ∂x/∂u tangent
    dv=np.zeros(3),               # ∂x/∂v tangent
    dudv=np.zeros(3),             # ∂²x/∂u∂v cross-derivative
)
```

The Hermite basis (`H3`) requires derivative fields; the Lagrange bases
(`L1`–`L4`) do not.  Parameters can be *fixed* (excluded from
optimisation) via `node.fix_parameter(...)`.

---

## `MeshElement` – tensor-product element

An element connects nodes through a **product of 1-D basis functions**, one
per parametric direction:

```python exec="true" source="above" session="architecture"
from HOMER import MeshElement, H3, L1

# 2-D cubic-Hermite surface element: 2 × 2 = 4 nodes
elem2d = MeshElement(node_indexes=[0, 1, 2, 3],
                     basis_functions=H3**2)

# 3-D volume element with trilinear basis: 2 × 2 × 2 = 8 nodes
elem3d = MeshElement(node_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                     basis_functions=L1**3)
```

The element computes the **tensor-product weight matrix** at construction time
(`BasisProductInds`) which drives all subsequent evaluations.

### Parameter maps

Each parameter an element's basis contracts against is a *slot*: a lattice
position `j` and a field (`'loc'`, then the element's `used_node_fields`).  By
default a slot is that field of the node at that position.  `param_map`
generalises this to a weighted sum of node fields, so a slot can be scaled, or
interpolated from several nodes:

```python exec="true" source="above" session="architecture"
param_map = {slot: [(slot[0], slot[1], 1.0)] for slot in elem2d.slots}  # the identity
param_map[(0, 'du')] = [(0, 'du', 0.5)]                                 # one scaled slot
scaled = MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=H3**2, param_map=param_map)
print(scaled.terms((0, 'du')))
```

Terms point at positions in the element's `nodes` list, which is the element's
local node pool, so renumbering the mesh's nodes never touches a map.
`generate_mesh` assembles every element's map into `ele_map` and `ele_weights`,
both `(n_elements, n_slots, K)`.  A slot is the weighted sum of up to `K`
DOFs, where a DOF is one node field, `fdim` parameters long.  An evaluation
applies every element's map once and then gathers the elements it needs, so a
point costs the same however many terms the longest map has.

---

## `MeshField` – evaluatable field

`MeshField` holds a list of nodes and elements, and provides all evaluation
and fitting methods.  When `generate_mesh()` is called (automatically in the
constructor), it:

1. Assembles the flat parameter vector `true_param_array` from all node data.
2. Identifies the `optimisable_param_array` subset (parameters not fixed).
3. Compiles JAX evaluation functions via `_generate_eval_function()` etc.
4. Explores the mesh topology (`_explore_topology()`) to build the `bmap`
   and `topomap` for cross-element point embedding.

The `@expand_wide_evals` decorator automatically adds two convenience
variants for every `@wide_eval`. For an example function `foo`, the following 
additional variants are defined:

| Variant | Signature | Purpose |
|---|---|---|
| `foo(eles, xis)` | base | evaluate at paired (element, xi) locations |
| `foo_in_every_element(xis)` | IEE | evaluate the same xi grid in every element |
| `foo_ele_xi_pair(eles, xis)` | pair | same as base, different batching |

---

## `Mesh(MeshField)` – primary coordinate mesh

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
mesh but can use different basis functions; the
[worked example](how-to/secondary-fields.md#worked-example-normal-vector-scalar-height-fields)
fits an `H3` vector field on an `L1` mesh.

---

## Bases and basis groups

A basis is a *value*, not a type: `H3`, `L1`, … are frozen `Basis`
instances, interned by name in a registry.  Each carries:

| Attribute | Description |
|---|---|
| `name` | Serialisation key and repr, e.g. `'H3'` |
| `fn` | Evaluation function `fn(x) → (n_pts, n_basis)` |
| `deriv` | Tuple `(fn, d1, d2, …)` of derivative functions |
| `weights` | Ordered weight names, e.g. `('x0', 'dx0', 'x1', 'dx1')` |
| `order` | Polynomial order |
| `node_locs` | Node positions in `[0, 1]` |
| `node_fields` | `DerivativeField` instance (Hermite), or `None` (Lagrange) |
| `interpolatory` | Whether nodal parameters are field values at the nodes |

An element's parametric directions are built from those values with
arithmetic, and the result is a `BasisGroup`, a `tuple` subclass, so lists and
tuples of bases remain valid input everywhere — see
[Combining bases](how-to/mixed-basis.md#combining-bases) for the algebra.

Equality and hashing are by name, so a basis survives a deepcopy, a pickle
and a JSON round-trip as the same value.  `Basis` validates itself on
construction (`deriv[0]` must be `fn`; `fn` must return one column per weight
name), and registers itself, which is how a user-defined basis round-trips
through `HOMER.io` without touching the reader.

---

## JAX integration

All evaluation functions are JAX-compatible.  The key integration points are:

- `evaluate_embeddings`, `evaluate_deriv_embeddings`, `evaluate_jacobians`
  are JIT-compiled via `jax.jit` when `jax_compile=True`.
- `HOMER.embedding.build_embedding_fn` builds the point-embedding closures
  once per `generate_mesh()` — a coarse nearest-neighbour search, a vmapped
  Newton-Raphson solver, and `mesh_embed_points` with a custom JVP that
  reuses the converged Jacobian.  Building them once removes the XLA
  retracing a per-call definition caused, and the iteration count is passed
  as a traced value so changing `iterations` does not retrace either.
- `topomap` is a `@jax.jit`-compiled function for cross-element boundary
  mapping.
- `jacobian_evaluator.jacobian` uses `sparsejac` (forward-mode AD with
  sparsity exploitation) to build efficient Jacobians for `scipy.optimize`.
