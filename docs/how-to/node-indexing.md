# Node Indexing

HOMER supports two ways to reference nodes in a `MeshElement`: by
**integer index** (position in the node list) or by **user-assigned ID**.
Understanding the difference is important when building complex multi-element
meshes.

---

## Index-Based Nodes (Default)

By default, `MeshElement` uses **integer indices** into the parent `Mesh`'s
`nodes` list:

```python
from HOMER import Mesh, MeshNode, MeshElement, H3Basis
import numpy as np

node0 = MeshNode(loc=[0., 0., 0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node1 = MeshNode(loc=[1., 0., 0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node2 = MeshNode(loc=[0., 1., 0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node3 = MeshNode(loc=[1., 1., 0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

# Indices 0, 1, 2, 3 refer to positions in [node0, node1, node2, node3]
element = MeshElement(
    node_indexes=[0, 1, 2, 3],
    basis_functions=(H3Basis, H3Basis),
)
mesh = Mesh(nodes=[node0, node1, node2, node3], elements=element)
```

---

## ID-Based Nodes

Assign a string or integer `id` to nodes and reference them by ID in
elements:

```python
node0 = MeshNode(loc=[0., 0., 0.], id='corner_00',
                 du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node1 = MeshNode(loc=[1., 0., 0.], id='corner_10',
                 du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node2 = MeshNode(loc=[0., 1., 0.], id='corner_01',
                 du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node3 = MeshNode(loc=[1., 1., 0.], id='corner_11',
                 du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

# Reference by ID instead of index
element = MeshElement(
    node_ids=['corner_00', 'corner_10', 'corner_01', 'corner_11'],
    basis_functions=(H3Basis, H3Basis),
)
mesh = Mesh(nodes=[node0, node1, node2, node3], elements=element)
```

---

## Shared Nodes Across Elements

For multi-element meshes, neighbouring elements share boundary nodes.  Index-
based referencing makes this straightforward because the same integer refers
to the same node object:

```python
# Two adjacent H3 elements sharing nodes 1 and 3
#  0 - 1 - 4
#  |   |   |
#  2 - 3 - 5

all_nodes = [
    MeshNode(loc=[0.,0.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 0
    MeshNode(loc=[1.,0.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 1 – shared
    MeshNode(loc=[0.,1.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 2
    MeshNode(loc=[1.,1.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 3 – shared
    MeshNode(loc=[2.,0.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 4
    MeshNode(loc=[2.,1.,0.], du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3)),  # 5
]

elem_left  = MeshElement(node_indexes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))
elem_right = MeshElement(node_indexes=[1,4,3,5], basis_functions=(H3Basis, H3Basis))

mesh = Mesh(nodes=all_nodes, elements=[elem_left, elem_right])
```

---

## Looking Up Nodes and Elements by ID

When IDs are set, you can retrieve objects by ID:

```python
# Get a node by ID
node = mesh.get_node('corner_00')

# Get an element by ID
elem = mesh.get_element('my_elem_id')
```

---

## Node Ordering Within an Element

The order of nodes in `node_indexes` follows **Fortran (column-major) order**
for the tensor product: the *first* index varies fastest.

For a 2-D element with basis order ``[u_basis, v_basis]``:

```
node_indexes = [  # xi_u × xi_v grid
    node(u=0, v=0),   # index 0
    node(u=1, v=0),   # index 1
    node(u=0, v=1),   # index 2
    node(u=1, v=1),   # index 3
]
```

For a 3-D element, the order is ``u`` fastest, then ``v``, then ``w``.

---

## Node Ordering Across the Mesh

The order *within* an element is fixed by the basis.  The order of the mesh's
`nodes` list is not: refining, rebasing and most other manipulations rebuild it
from scratch, and the order that falls out is an artefact of the algorithm.
`reorder_nodes` replaces it with an ordering derived from the mesh itself, so
the same mesh built two different ways numbers its nodes the same way:

```python
from HOMER import reorder_nodes

reorder_nodes(mesh)                # 'lattice', the default
reorder_nodes(mesh, 'spatial')     # lexicographic in (z, y, x)
reorder_nodes(mesh, 'bandwidth')   # reverse Cuthill-McKee
```

`refine()` and `rebase()` already call it — pass `reorder_nodes=False` to
either to opt out, or set `HOMER.mesh.reordering.DEFAULT_NODE_ORDERING = False`
to turn it off for the whole session.

An operation that leaves the node *set* alone leaves the numbering alone too:
when every node of the result sits on exactly one node of the original and none
is left over — an L1 → H3 rebase, a refinement by a factor of one — the mesh's
own numbering is reproduced and the strategy is not consulted, so node indices
you are holding stay valid.  A real refinement is not that case: its old nodes
are a strict subset of the new ones, so it renumbers.

| strategy | orders by | use it when |
| --- | --- | --- |
| `'lattice'` | the node's position in the mesh's parametric lattice, `xi_0` fastest | almost always: it is coordinate-free, so it is meaningful for a secondary field and unaffected by how the mesh is posed or deformed |
| `'spatial'` | the nodal coordinates, last coordinate slowest | you want the numbering to follow world space, and the field is geometric |
| `'bandwidth'` | reverse Cuthill-McKee over the node adjacency graph | the mesh is not a lattice, and you care about the bandwidth of the parameter couplings |

Reordering is a *pure renumbering*: node objects, elements, bases and
everything stored on a node — including its fixed parameters — are carried
across untouched, and the field evaluates identically before and after.
Elements referencing nodes by ID need no rewriting at all, since the ID travels
with the node.

The permutation is returned, so an index list of your own can follow it:

```python
perm = reorder_nodes(mesh)
inverse = np.empty_like(perm)
inverse[perm] = np.arange(len(perm))

watched = inverse[watched]     # the same nodes, at their new indices
```
