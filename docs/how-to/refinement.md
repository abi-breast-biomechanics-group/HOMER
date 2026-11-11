# Mesh Refinement

Refinement subdivides every element into smaller sub-elements, increasing the
spatial resolution of the mesh while preserving the underlying high-order
geometry.

---

## Uniform refinement

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

## Non-uniform refinement

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

A Hermite node between two children of different widths carries one
derivative, but each child needs it scaled by its own width.  Each child's
[parameter map](../architecture.md#parameter-maps) does that scaling, so an
uneven refinement of `H3` is as exact as an even one, and an even one writes no
maps at all.  `B3` is the exception: an uneven split makes its knot vector
non-uniform, which no scaling recovers.

```python exec="true" source="above" session="refinement"
# thin layers at both ends of w, and a coarse split in u
graded = hermite_cube()
graded.refine(by_xi_refinement=(
    np.array([0, 0.5, 1.0]),
    np.array([0, 1.0]),
    np.array([0, 0.1, 0.9, 1.0]),
))
graded.plot()
print(graded.elements[0].terms((0, 'dw')))   # the shared w derivative, scaled by this child's width
```

!!! warning
    The two parameters `refinement_factor` and `by_xi_refinement` are
    mutually exclusive.  Providing both raises an `AssertionError`.

---

## Refining only some elements

`refine_elements` halves the chosen elements in every direction and leaves the
rest alone.  Refining a child again takes the mesh another level deeper:

```python exec="true" source="above" session="refinement"
local = hermite_cube()
local.refine(2)
local.refine_elements([0])
corner = min(range(len(local.elements)), key=lambda i: sum(local.elements[i].box[1]))
refined = local.refine_elements([corner])
print(f"{len(local.elements)} elements, {len(local.hanging)} hanging faces, "
      f"sizes {sorted({float(e.box[2]) for e in local.elements})}")
local.plot()
```

The refined mesh carries the *truncated hierarchical* (THB) space of its
basis.  Each level of refinement has its own functions; a function is a DOF
where its support is refined to its level and no further, and a coarse function
gives way exactly where the finer ones take over, so the functions still sum to
one.  The children reproduce their parents exactly, so there is no fit, and the
field is as smooth across a change of level as anywhere else.

For a nodal basis that leaves the points a child would put on a face shared
with a coarser element *hanging*: they get no node, and their
[parameter map](../architecture.md#parameter-maps) draws their values from the
coarse element's field there.  New nodes are appended to the node list, so an
index into the existing nodes stays valid, and refining the coarse element later
gives those hanging points nodes again.

`B3` is refined the same way.  Its nodes are control points, the coefficients of
functions rather than values at points, so one can go: once refinement covers
a coarse function's whole support it is retired, and its node with it.  The
control points keep their meaning -- the net still moves the surface rigidly.

```python exec="true" source="above" session="refinement"
from HOMER import B3

net = hermite_cube().rebase(B3**3)
net.refine(2)
net.refine_elements([0])
print(f"{len(net.elements)} elements, {len(net.nodes)} control points")
net.plot()
```

Refinement is kept admissible: an element is refined together with every
coarser element its functions reach, so at most two levels act on any element.
For a basis whose functions reach one cell -- every nodal one -- that is a 2:1
balance across faces, edges and corners; `B3` reaches three cells, so its
refinement spreads further.  `refine_elements` returns the indices of every
element it refined, the balance's included.  Each element's `box` records where
it sits in the lattice the mesh was refined from, as
`(component, origin, size)`.

`mesh.hanging` maps each face lying against a larger one to that face, as
`(element, direction, side)` pairs.  Neither side counts as an exposed face.

!!! warning
    `refine` and `rebase` read the mesh as a lattice of equal elements, and
    raise on a locally refined one.

---

## Refining a `Mesh` with secondary fields

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

## Node numbering

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

## Visualising before and after

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
