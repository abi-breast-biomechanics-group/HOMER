# Mesh Refinement

Refinement subdivides every element into smaller sub-elements, increasing the
spatial resolution of the mesh while preserving the underlying high-order
geometry.

---

## Uniform Refinement

`mesh.refine(refinement_factor=n)` splits each parametric direction into
`n` sub-intervals, creating `n ** ndim` sub-elements per original element.

```python exec="true" source="above" session="refinement"
import numpy as np

from HOMER.examples import hermite_cube

# a curved tricubic-Hermite cube: its w-direction edges bow outwards, so a
# refinement that preserves the geometry is visible and one that does not is too
mesh = hermite_cube()

# each element is subdivided into 2×2×2 = 8 sub-elements
mesh.refine(refinement_factor=2)
print(f"Elements after refinement: {len(mesh.elements)}")  # → 8
mesh.plot()
```

---

## Non-Uniform Refinement

`refinement_factor` also takes one integer per parametric direction, which
subdivides each of them uniformly but by a different amount:

```python exec="true" source="above" session="refinement"
# 2 sub-intervals in u, 1 (i.e. none) in v, 3 in w -- 6 sub-elements
graded = hermite_cube()
graded.refine([2, 1, 3])
print(f"{len(graded.elements)} elements")
graded.plot()
```

For boundaries that are not evenly spaced, pass `by_xi_refinement` instead: one
array of xi breakpoints per direction, each running from 0 to 1.

!!! warning
    Hermite meshes, without the introduction of "scale factors" cannot nicely represent 
    arbitrary surface. You can see that the very nonuniform refinement performed below
    results in imperfect surface reconstructions.

```python exec="true" source="above" session="refinement"
# thin layers at both ends of w, and a coarse split in u
graded = hermite_cube()
graded.refine(by_xi_refinement=(
    np.array([0, 0.5, 1.0]),
    np.array([0, 1.0]),
    np.array([0, 0.1, 0.9, 1.0]),
))
graded.plot()
```

!!! warning
    The two parameters `refinement_factor` and `by_xi_refinement` are
    mutually exclusive.  Providing both raises an `AssertionError`.

---

## Refining a `Mesh` with Secondary Fields

When the mesh has secondary fields, `Mesh.refine()` automatically refines all
fields simultaneously, so the geometry and every field stay at the same
resolution:

```python exec="true" source="above" session="refinement"
from HOMER import L1
from HOMER.geometry import cube

field_mesh = cube(basis=L1**3)

rng = np.random.default_rng(0)
data_pts = rng.random((500, 3))
fibre_vectors = np.tile([1., 0., 0.], (len(data_pts), 1))

field_mesh.new_field('fibre', field_dimension=3, new_basis=L1**3,
                     field_locs=data_pts, field_values=fibre_vectors)

# Refine both the geometry and the 'fibre' field
field_mesh.refine(refinement_factor=2)
```

---

## Node Numbering

Refinement rebuilds the node list, so node *indices* do not survive one that
adds nodes.  By default the refined nodes are renumbered along the mesh's
parametric lattice — `xi_0` fastest, the last direction slowest — so a refined
axis-aligned cube comes out in lexicographic `(z, y, x)` order, and the same
mesh reached by two different routes numbers its nodes the same way:

```python exec="true" source="above" session="refinement"
a = cube(basis=L1**3); a.refine(4)
b = cube(basis=L1**3); b.refine(2); b.refine(2)
# a and b now have identical node orderings
print(np.allclose([n.loc for n in a.nodes], [n.loc for n in b.nodes]))
```

A refinement that adds *no* nodes — a factor of one in every direction — is
the exception: there is an old numbering to keep, so it is kept and node
indices survive unchanged.

Pass `reorder_nodes=False` to keep the raw ordering the subdivision sweep
produces, or a strategy name (`'lattice'`, `'spatial'`, `'bandwidth'`) to pick
another — see [Node indexing](node-indexing.md#node-ordering-across-the-mesh).

```python exec="true" source="above" session="refinement"
coarse = cube(basis=L1**3)
coarse.refine(2, reorder_nodes=False)      # leave the numbering alone
coarse.refine(2, reorder_nodes='spatial')  # sort on coordinates instead
```

Each secondary field is renumbered from its own topology, so a field stays
co-located with the geometry without the two sharing a node numbering.

---

## Visualising Before and After

Subdivision must leave the surface it passes through unmoved. The refined mesh draws its
nodes in green, the convention these guides use for the *after* of a pair.

```python exec="true" source="above" session="refinement"
import pyvista as pv

mesh_before = hermite_cube()
mesh_after = hermite_cube()
mesh_after.refine(2)

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0)
mesh_before.plot(s)
s.subplot(0, 1)
mesh_after.plot(s, node_colour='g')
s.link_views()
s.show()
```
