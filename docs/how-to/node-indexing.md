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
