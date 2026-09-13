```python exec="true" session="index"
import pyvista as pv

from HOMER.examples import wordmark

s = pv.Plotter(window_size=(2100, 900))
wordmark().plot(s, node_size=1)

# the letters lie in the z = 0 plane, so a camera down z reads them as text
s.view_xy()
s.camera.Dolly(2.0)
s.reset_camera_clipping_range()
s.show()
```

# HOMER – High Order Mesh Representations

HOMER is a Python library for constructing, fitting, evaluating, and visualising
**high-order finite-element meshes** using JAX for fast, differentiable
computations.  It supports arbitrary tensor-product basis functions
(cubic Hermite, linear/quadratic/cubic/quartic Lagrange) in 2-D and 3-D, and
provides tools for:

- Building manifold surface meshes and volume meshes
- Fitting meshes to point clouds via nonlinear least squares
- Embedding arbitrary points into the parametric coordinates of a mesh
- Refining mesh resolution while preserving the underlying geometry
- Converting between basis functions (rebasing)
- Storing and evaluating secondary vector/scalar fields over the mesh topology
- Computing Green-Lagrange strain tensors
- Saving and loading meshes to/from JSON

---

## Quick Example

```python exec="true" source="above" session="index"
import numpy as np
from HOMER import Mesh, MeshNode, MeshElement, H3

# 1. Create four corner nodes for a flat 2-D patch
node0 = MeshNode(loc=np.array([0., 0., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node1 = MeshNode(loc=np.array([1., 0., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node2 = MeshNode(loc=np.array([0., 1., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))
node3 = MeshNode(loc=np.array([1., 1., 0.]), du=np.zeros(3), dv=np.zeros(3), dudv=np.zeros(3))

# 2. Link the nodes through a bicubic-Hermite element
element = MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=H3 * 2)

# 3. Create the mesh
mesh = Mesh(nodes=[node0, node1, node2, node3], elements=element)

# 4. Evaluate the surface at a 10×10 grid
xis = mesh.xi_grid(10)                          # (100, 2)
pts = mesh.evaluate_embeddings_in_every_element(xis)  # (100, 3)

# 5. Visualise
mesh.plot()
```

While the hexagonal lattice is stylistically HOMER, it also shows the xi spacing of the created mesh.
As we created a Hermite mesh with 'degenerate' 0 node derivatives, the grid is not even over the surface of the mesh!
As a result, it is often better to use the `cube()` geometry object. 

---

## Getting Started

### Installation

A conda environment is recommended:

```bash
conda create --name HOMER "python=3.13"
conda activate HOMER
```

Install directly from the repository:

```bash
pip install git+https://github.com/abi-breast-biomechanics-group/HOMER.git
```

Or clone first for an editable install:

```bash
git clone https://github.com/abi-breast-biomechanics-group/HOMER
cd HOMER
pip install -e .
```

For the test and documentation extras:

```bash
pip install -e ".[dev]"    # pytest
pip install -e ".[docs]"   # mkdocs, mkdocstrings
```

### Troubleshooting

**JAX installs but runs on the CPU.** 

Nothing in HOMER is CPU-specific — evaluation, fitting and embedding are
ordinary JAX, and they run wherever JAX does — but the CPU build is what
the test suite, the benchmarks and every example in these guides are run
against, and what the defaults are tuned for.  Treat an accelerator
backend as untested rather than unsupported.

`pip install jax` gives you the CPU build. For GPU or TPU you need the matching
accelerator wheel from the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html);
HOMER does not pin one, because the right wheel depends on your CUDA version.
Check what you got with:

```python
import jax; print(jax.devices())
```

**PyVista cannot open a window.** Over SSH, in a container, or in CI there is
no display to open. Switch to off-screen rendering *before* PyVista is
imported:

```python
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")

import pyvista as pv
pv.OFF_SCREEN = True
```

`tests/conftest.py` does exactly this, which is why the suite runs headless.
See the [plotting guide](how-to/plotting.md#rendering-without-a-display).

**The first evaluation is slow.** That is XLA compiling. HOMER turns on JAX's
persistent compilation cache at import, in your platform's per-user cache
directory — `~/Library/Caches/HOMER` on macOS, `~/.cache/HOMER` on Linux,
`%LOCALAPPDATA%\HOMER\Cache` on Windows — so the cost is paid once per mesh
shape per machine rather than once per process.

That is only a default. Set `JAX_COMPILATION_CACHE_DIR` to put the cache
elsewhere, or to an empty string to turn it off, and HOMER leaves your choice
alone:

```bash
export JAX_COMPILATION_CACHE_DIR=""     # no cache
```

The setting is process-global, so it applies to all of your JAX work, not only
HOMER's.

**Numerical results differ slightly between runs or machines.** HOMER
evaluates in float32. Quantities reached by least squares or Newton-Raphson
agree to about `1e-3`, exact-arithmetic quantities to about `1e-5`; the test
suite uses those two tolerances throughout.

### Key Concepts

| Term | Description |
|---|---|
| `MeshNode` | Physical location + Hermite derivative vectors |
| `MeshElement` | Connects nodes via tensor-product basis functions |
| `MeshField` | Collection of nodes + elements; evaluates/fits a field |
| `Mesh` | Primary coordinate field that can carry secondary fields |
| Basis | 1-D interpolation building block: `H3`, `L1`–`L4` |
| xi | Parametric coordinates in [0, 1]ⁿ |

### Supported Basis Functions

| Class | Type | Nodes/dir | Continuity | Node fields |
|---|---|---|---|---|
| `H3` | Cubic Hermite | 2 | C¹ | `du`, `dv`, … |
| `L1` | Linear Lagrange | 2 | C⁰ | – |
| `L2` | Quadratic Lagrange | 3 | C⁰ | – |
| `L3` | Cubic Lagrange | 4 | C⁰ | – |
| `L4` | Quartic Lagrange | 5 | C⁰ | – |
| `B3` | Cubic B-spline | 4 control points | C² | – |

`B3` is not interpolatory: its parameters are control points shared with
the neighbouring elements, so they do not lie on the curve.

---

## Workflow

The core workflow demonstrated in the test suite is:

1. **Create nodes** – instantiate `MeshNode` objects with physical coordinates
   and (for Hermite bases) derivative vectors.
2. **Create elements** – combine nodes with a group of bases, e.g. `H3 * 3`.
3. **Build the mesh** – pass nodes and elements to `Mesh(...)`.
4. **Evaluate** – call `evaluate_embeddings()`, `evaluate_jacobians()`, etc.
5. **Fit** – use `linear_fit()` or `point_cloud_fit()` to update node parameters.
6. **Refine** – call `mesh.refine(2)` to subdivide elements.
7. **Save / load** – `mesh.save('path.json')` / `load_mesh('path.json')`.

See the [How-To Guides](how-to/3d-meshes.md) for detailed walk-throughs.

---

## Next Steps

- [Architecture overview](architecture.md) – understand the class hierarchy.
- [How-To Guides](how-to/3d-meshes.md) – feature-specific recipes.
- [API Reference](api/mesh.md) – full docstring reference.
