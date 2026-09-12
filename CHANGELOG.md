# Changelog

Notable changes to HOMER. No releases have been tagged yet, so everything to
date sits under Unreleased.

## Unreleased

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
  changing it does not retrace.
- `linear_fit` solves column-equilibrated, recovering several digits in
  float32 for Hermite and B-spline weight matrices.
- The test suite was rebuilt: the scripts that ended at `plotter.show()` are
  now assertions, and timing scripts moved to `benchmarks/`.
- Docstrings are reST throughout, rendered by mkdocstrings with a griffe
  extension that turns roles into cross-references.
- `load_exelem` renamed to `load_ipmesh`, after the format it reads.

### Removed
- `compat_functions/dep_mesh.py`, which had never parsed.
- `compat_functions/convert_morphic.py`, which needed a dependency the
  project does not declare.
- `bspline.py`, which nothing imported; the B-spline basis lives in
  `basis_definitions.py`.
