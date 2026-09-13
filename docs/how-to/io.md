# Saving and Loading Meshes

HOMER stores meshes as plain JSON.  A file holds node locations, every
derivative field, the element node-lists, the basis of each parametric
direction (by name), any fixed-parameter constraints, and any secondary
fields — everything needed to rebuild the mesh, and nothing else.

---

## The Round Trip

```python exec="true" source="above" session="io"
import tempfile
from pathlib import Path

import numpy as np
from HOMER import H3, L1, Mesh, load_mesh, save_mesh
from HOMER.geometry import cube

tmp = Path(tempfile.mkdtemp())

mesh = cube(scale=2, centre=np.zeros(3), basis=H3 * 3)
mesh.refine(2)

mesh.save(tmp / 'cube.json')          # or save_mesh(mesh, tmp / 'cube.json')
loaded = Mesh.load(tmp / 'cube.json')  # or load_mesh(tmp / 'cube.json')

print(f"{len(loaded.nodes)} nodes, {len(loaded.elements)} elements")
print("parameters identical:",
      np.array_equal(loaded.true_param_array, mesh.true_param_array))
```

`Mesh.save` / `Mesh.load` and `save_mesh` / `load_mesh` are the same pair;
the methods are there so you never need the import.  Paths may be `str` or
`Path`, and an existing file is overwritten.

---

## What the File Looks Like

A mesh is written as a `'main'` field plus a dict of named `'fields'`:

```python exec="true" source="above" session="io"
import json

cube(basis=L1 * 3).save(tmp / 'trilinear.json')
payload = json.loads((tmp / 'trilinear.json').read_text())

print("top level :", list(payload))
print("main      :", list(payload['main']))
print("node 0    :", payload['main']['nodes']['0'])
print("element 0 :", payload['main']['elements']['0'])
```

A cubic-Hermite node carries the same `'loc'` plus one entry per derivative
field (`'du'`, `'dv'`, `'dudv'`, …); everything else is unchanged.

A basis serialises as its **name**, and the basis registry resolves that name
back on load.  The file is therefore readable, diffable, and independent of
HOMER's internals as long as the basis names stay stable — which includes a
basis you defined yourself, since constructing it registers it.

---

## Dictionaries Instead of Files

The same conversion is available without touching the disk, which is what you
want for embedding a mesh in a larger document, sending one over a socket, or
storing one in a database:

```python exec="true" source="above" session="io"
payload = mesh.to_dict()          # dump_mesh_to_dict(mesh)
rebuilt = Mesh.from_dict(payload)  # parse_mesh_from_dict(payload)

print("same geometry:",
      np.allclose(rebuilt.true_param_array, mesh.true_param_array))
```

`deepcopy(mesh)` goes through exactly this path, so a copy is as independent
of its original as a reloaded file is.

---

## What Survives a Round Trip

| Carried through | Notes |
|---|---|
| Node locations and all derivative fields | `du`, `dv`, `dudv`, … whatever the node holds |
| Basis functions | By name, per parametric direction |
| Node and element ids | Tuple ids come back as tuples, not lists |
| Fixed parameters | Restored *before* `generate_mesh()`, so the optimisable mask is right |
| Secondary fields | Each under its name in `mesh.fields` |
| Elements referencing nodes by id | `used_index` records which way an element was built |

The loaded mesh is already generated: its topology map and parameter arrays
are built before `load` returns, so it is ready to evaluate.

```python exec="true" source="above" session="io"
mesh.nodes[0].fix_parameter('loc', inds=[2])
mesh.generate_mesh()
mesh.save(tmp / 'constrained.json')

loaded = Mesh.load(tmp / 'constrained.json')
print("fixed on node 0:", loaded.nodes[0].fixed_params)
print("free parameters:", len(loaded.optimisable_param_array))
```

---

## Fields Are Saved the Same Way

A bare `MeshField` — a secondary field on its own, rather than a whole mesh —
writes the older `{'nodes', 'elements'}` shape, and reads back with
`parse_meshfield_from_dict`:

```python
from HOMER.io import dump_meshfield_to_dict, parse_meshfield_from_dict

field.save('fibres.json')
payload = dump_meshfield_to_dict(field)
field_again = parse_meshfield_from_dict(payload)
```

`Mesh.load` accepts that shape too, so any mesh file written before the
`{'main', 'fields'}` schema still loads, and so do files naming the pre-1.0
bases (`'H3Basis'` for `'H3'`, and so on).

---

## Reading Formats HOMER Does Not Write

`HOMER.compat_functions` brings existing models in.  Nothing imports these,
so reach for the one you need directly:

```python
# OpenCMISS ipnode/ipelem pair
from HOMER.compat_functions.load_ipmesh import load_mesh as load_ipmesh

mesh = load_ipmesh('breast.ipnode', 'breast.ipelem',
                   basis=H3**2 * L1,       # one basis per direction
                   keys=('du', 'dv', 'dudv'))

# VTK unstructured grid of cubic-Lagrange hexahedra
from HOMER.compat_functions.load_VTU import load_L3_vtu_as_HOMER

mesh = load_L3_vtu_as_HOMER('volume.vtu')
```

`load_ipmesh` names nodes by the file's own node and version numbers, so
elements reference nodes by id rather than by index, and nodes no element
uses are dropped.  `load_L3_vtu_as_HOMER` expects every cell to be a
64-point cubic-Lagrange hexahedron and raises `ValueError` otherwise.

Once read, save the result as JSON — that conversion only needs doing once.

---

## Notes

- An unknown basis name raises `KeyError` listing the bases that *are*
  registered.  A user-defined basis must be constructed (which registers it)
  in the loading process before the file is read.
- Values are written with `tolist()`, so a float32 mesh round-trips exactly;
  the file is larger than a binary dump but stays readable.
- `save_mesh` writes with `indent=4`, which diffs and merges sensibly under
  version control.
