# Basis Conversion (Rebase)

`rebase()` converts a mesh from one set of basis functions to another by:

1. Computing the new node positions from the current geometry.
2. Sampling a dense xi grid on the current mesh.
3. Linearly fitting the new nodal parameters to match the sampled geometry.

This is the recommended way to build high-order meshes: start with a coarse
Lagrange mesh that is easy to set up manually, then rebase to a higher-order
or Hermite basis.

---

## Trilinear → Cubic Hermite Conversion

A cubic-Hermite element needs seven derivative vectors per node, and the
obvious shortcut — write the eight corners down and leave the derivatives at
zero — does not give you the mesh you meant.  Rebasing from a trilinear seed
fits those derivatives instead:

```python exec="true" source="above" session="rebase"
import numpy as np
import pyvista as pv
from HOMER import Mesh, MeshNode, MeshElement, L1, H3

corners = [[x, y, z] for z in [0., 1.] for y in [0., 1.] for x in [0., 1.]]

# 1. A coarse trilinear mesh (L1 × L1 × L1), rebased to cubic Hermite
seed = Mesh(nodes=[MeshNode(loc=c) for c in corners],
            elements=MeshElement(node_indexes=list(range(8)),
                                 basis_functions=L1**3))
mesh = seed.rebase(H3**3)

# 2. The same eight corners, written straight into an H3 element
zero = np.zeros(3)
by_hand = Mesh(nodes=[MeshNode(loc=np.array(c), du=zero, dv=zero, dw=zero,
                               dudv=zero, dudw=zero, dvdw=zero, dudvdw=zero)
                      for c in corners],
               elements=MeshElement(node_indexes=list(range(8)),
                                    basis_functions=H3**3))

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); mesh.plot(s)
s.subplot(0, 1); by_hand.plot(s, node_colour='g')
s.link_views()
s.show()
```

Both are unit cubes, and they are not the same mesh.  The surface grid a draw
lays down is a grid in *xi*, so the parameterisation is visible in it: the
hand-written cube has zero tangents at every corner, which makes xi crawl near
the faces and rush through the middle, and its grid lines bunch up against the
edges.  The rebased cube's fitted tangents give the uniform spacing the seed
had.

```python exec="true" source="above" session="rebase"
xi = np.array([[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]])
eles = np.zeros(3, dtype=int)

print("rebased x:", np.asarray(mesh.evaluate_embeddings(eles, xi))[:, 0].round(4))
print("by hand x:", np.asarray(by_hand.evaluate_embeddings(eles, xi))[:, 0].round(4))
```

Geometry that is only ever evaluated as a shape does not care.  Anything that
reads xi — embedding, fitting against sampled data, a secondary field, a strain
measure — does.

---

## Converting a 2-D Manifold Mesh

The same workflow applies to 2-D surface meshes, and on a multi-element one it
buys continuity as well as order: the bilinear seed is faceted, and the
Hermite nodes carry one tangent each, so the creases between elements cannot
survive the conversion.

```python exec="true" source="above" session="rebase"
from HOMER.geometry import basic_surface

# a 3 x 3 patch of bilinear elements, pushed into a bump
seed_2d = basic_surface(corner_locs=np.array([[0, 0, 0], [1, 0, 0],
                                              [0, 1, 0], [1, 1, 0]]),
                        basis=L1**2)
seed_2d.refine(3)
for node in seed_2d.nodes:
    x, y, _ = node.loc
    node.loc = np.array([x, y, 0.3 * np.sin(np.pi * x) * np.sin(np.pi * y)])
seed_2d.generate_mesh()

smooth_2d = seed_2d.rebase(H3**2)

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); seed_2d.plot(s)
s.subplot(0, 1); smooth_2d.plot(s, node_colour='g')
s.link_views()
s.show()
```

---

## Rebase to Lagrange Basis

You can also rebase from Hermite to Lagrange (e.g. for export or
compatibility with other solvers):

```python exec="true" source="above" session="rebase"
from HOMER import L3

hermite_mesh = mesh                 # the H3 mesh rebased above
lagrange_mesh = hermite_mesh.rebase(L3**3)
lagrange_mesh.plot()
```

---

## Resolution Control

The `res` parameter of `rebase()` controls how many xi samples are used for
the linear fit.  Increase it for better accuracy when rebasing to a
significantly different basis:

```python exec="true" source="above" session="rebase"
mesh = seed.rebase(H3**3, res=20)  # default res=10
```

---

## Node Numbering

A rebase builds its node list from the new basis, so the order it comes back
in has nothing to do with the order the mesh had.  What happens to the
numbering depends on whether the two bases share their nodes:

- **Same nodes** (L1 ↔ H3, which differ only in the derivatives each node
  carries): every new node sits on exactly one old node, so the mesh's own
  numbering is reproduced and node indices stay valid.
- **Different nodes** (L1 → L2, which adds mid-element nodes): there is no old
  numbering to keep, so the nodes are ordered along the mesh's parametric
  lattice, the same ordering [refinement](refinement.md#node-numbering) uses.

```python exec="true" source="above" session="rebase"
from HOMER import L2

smooth = mesh.rebase(H3**3)                        # same nodes: numbering kept
denser = mesh.rebase(L2**3)                        # new nodes: lattice ordering
raw    = mesh.rebase(H3**3, reorder_nodes=False)   # as built, neither
```

A rebase to the basis the mesh already has returns an untouched copy, and so
is unaffected either way.

---

## Notes

- `rebase()` returns a new `MeshField` and leaves the original untouched.
- Pass `in_place=True` to rebase the mesh itself: it replaces the mesh's nodes
  and elements, regenerates it, and returns the same object.

```python exec="true" source="above" session="rebase"
new = mesh.rebase(H3**3)                   # mesh is unchanged
same = mesh.rebase(H3**3, in_place=True)   # same is mesh
```
