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

### Removed
- `compat_functions/dep_mesh.py`, which had never parsed.
- `compat_functions/convert_morphic.py`, which needed a dependency the
  project does not declare.
- `bspline.py`, which nothing imported; the B-spline basis lives in
  `basis_definitions.py`.
