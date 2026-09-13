# Changelog

Notable changes to HOMER.

## Unreleased

### Added
- `HOMER.examples`, the example meshes the documentation and the test suite
  share: `bulged_patch`, `hermite_cube`, `unit_hex`, and `wordmark`, the
  library's own name written as a mesh.  The first three were private to the
  suite (`tests/_helpers.py`); making them public means a page that shows a
  mesh and the test that asserts on it cannot drift apart.
- The documentation examples are executed while the site is built and whatever
  they draw is embedded as an interactive vtk.js scene, so every example on
  every page carries the picture it produces.  A broken example now fails
  `mkdocs build --strict`, which makes the docs a second test suite.

### Changed
- `linear_fit` respects fixed parameters.  Parameters pinned with
  `MeshNode.fix_parameter` are held at their current values and moved to the
  right-hand side, so the solve runs over the free columns only and returns
  the constrained minimiser.  Fixing is per component, and the weight matrix
  is shared across them, so components that share a free set share a solve --
  one for the usual mesh, at most `fdim` when the constraints cut across
  components.  This changes the result for anyone who relied on the fit
  overwriting a constraint; the system also needs only as many points as it
  has free columns.
- `refine` and `rebase` transfer the constraints before they fit, rather than
  after.  A pinned location is now held *through* the least-squares solve, so
  the parameters around it take up the slack, instead of being fitted freely
  and having the pinned value written back over the answer.  It shows where
  the constraint binds -- rebasing to a basis that cannot represent the
  source -- and leaves refinement unchanged, which reproduces its parent
  exactly either way.  Constraints with no value to carry across (derivatives,
  and `loc` on a control net) still take the value the fit gives them.

### Fixed
- The hexagonal surface lattice is built as line cells rather than as
  two-point polygons.  VTK drew the degenerate polygons as edges anyway, so
  desktop rendering is unchanged, but any exporter stricter than VTK -- vtk.js
  in a browser among them -- discarded them and drew no lattice at all.
- `plot_mesh`'s docstring gave `field_artist` as `(plotter, locs, values)`;
  it is called with `(plotter, locs, values, field_xi)`.
- `MeshNode.fix_parameter(values=...)` no longer truncates the value it pins
  when the node's array is an integer one, as it is for any mesh whose
  coordinates were stated as whole numbers.

## 1.0.0 - 2026-09-12

Everything below is the first tagged release; HOMER carried a `0.2.2.x`
version through its whole pre-release life.

### Added
- Predictable node numbering after `refine` and `rebase`, with `'lattice'`,
  `'spatial'` and `'bandwidth'` strategies and a `reorder_nodes` argument on
  both operations.
- A custom JVP for `embed_points`, reusing the Jacobian the Newton solve
  converged to, so the embedding can sit inside a differentiable pipeline.
- `dim_mask` on `embed_points`, for embedding against a subset of field
  dimensions.
- Colouring utilities for sparse Jacobian evaluation.
- `matrix_free_jacobian`, for fits whose Jacobian is too large to form or
  too dense for a colouring: a `LinearOperator` over `jvp`/`vjp`, with the
  column equilibration `least_squares(tr_solver='lsmr')` needs and cannot
  compute for itself.
- Fixed parameters are preserved through serialisation, refinement and
  rebasing.
- Reference pages for `embedding`, `utils`, the compatibility readers and the
  internals; how-to guides for plotting and node indexing.

### Changed
- `mesher.py` split into `HOMER.mesh`, a module per concern, with the old
  import path kept as a re-export shim.
- Bases are singletons: interned by name, hashable, and equal across a
  deepcopy, a pickle or a JSON round trip.
- Point embedding is roughly 8x faster — its closures are built once per
  `generate_mesh()` instead of per call, and the iteration count is traced so
  changing it does not retrace. `benchmarks/bench_embedding.py` embeds a
  million points in 0.55s.
- `linear_fit` solves column-equilibrated, recovering several digits in
  float32 for Hermite and B-spline weight matrices.
- The test suite was rebuilt: the scripts that ended at `plotter.show()` are
  now assertions, and timing scripts moved to `benchmarks/`.
- Docstrings are reST throughout, rendered by mkdocstrings with a griffe
  extension that turns roles into cross-references.
- `load_exelem` renamed to `load_ipmesh`, after the format it reads.
- `refine` and `rebase` solve their fit sparse. The weight matrix they build
  is block-sparse by construction — a query point sees one element — and was
  being formed dense and handed to an SVD: 2.2GB, 99.84% zeros, 49 of the 56
  seconds a `refine(16)` took. `refine(16)` is now 2.5s, and the cost grows
  with the mesh rather than with its cube. The sparse solve runs in float64,
  so it is also several digits more accurate than the float32 dense one, and
  it differentiates in both directions, so a refit can sit inside a loss.
- `get_xi_surface_nodes` reads a face off the basis and the element node
  ordering instead of building a weight matrix over a tiled xi query. It no
  longer allocates a dense `(25 * n_elements) x n_parameters` array, and no
  longer retraces once per mesh size. Between 40x and 5000x faster depending
  on the basis, for identical results.
- JAX's persistent compilation cache now defaults to the platform's per-user
  cache directory instead of a shared `/tmp/jax_cache`, and only when the user
  has not chosen one themselves — `JAX_COMPILATION_CACHE_DIR` is no longer
  overridden on import. The size threshold is left at JAX's default, which is
  what stops the cache growing without bound.
- The cache's compile-time floor is lowered to 0.01s, again only when
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` says nothing. JAX's default of
  one second is tuned for a handful of large kernels; HOMER compiles many small
  ones, none of which reached the floor, so the cache stayed empty. The full
  test suite goes from 264s to 166s with it, and the cache converges at ~28MB.
- Bases are combined with `*` rather than `+`, and are named without the
  `Basis` suffix: `H3Basis * 2 + B3Basis` is now `H3**2 * B3`. `*` is the
  operator nearest the outer product an element actually takes, and `**` is
  its tensor power; against an `int`, `*` still repeats a direction, so
  `H3 * 3` and `H3 ** 3` are the same group. A `BasisGroup` reprs as the
  expression that builds it.
- `Basis.name` — the serialisation key — follows the rename, so new mesh
  files record `"H3"`. `basis_by_name` still resolves the pre-1.0 spellings,
  so files written as `"H3Basis"` keep loading.
- The raw basis evaluation functions are private (`H3` -> `_H3`), which is
  what frees the short names for the bases themselves. Nothing outside
  `basis_definitions.py` imported them.

### Deprecated
- The `H3Basis`, `L1Basis`, `L2Basis`, `L3Basis`, `L4Basis` and `B3Basis`
  names. They remain importable and are the same objects as `H3`, `L1`, `L2`,
  `L3`, `L4` and `B3`, but are absent from `HOMER.__all__`, so `import *` and
  tab-completion offer one name per basis.

### Removed
- `+` as a basis operator. `BasisGroup` subclasses `tuple`, so rather than
  inherit `tuple.__add__` and silently return a plain tuple, `+` raises a
  `TypeError` naming the `*` spelling to use instead.
- `compat_functions/dep_mesh.py`, which had never parsed.
- `compat_functions/convert_morphic.py`, which needed a dependency the
  project does not declare.
- `bspline.py`, which nothing imported; the B-spline basis lives in
  `basis_definitions.py`.
- `mesher.pyi`, alongside the `mesher.py` split; the generated stub now sits
  next to the class it describes, as `mesh/field.pyi`.
