# HOMER test suite

```bash
pip install -e ".[dev]"
pytest                    # ~2 minutes, no windows, no prompts
pytest tests/test_embedding.py -q
```

## What changed and why

The suite used to be a folder of scripts that built a mesh, drew it with
PyVista, and stopped at `plotter.show()` waiting for a human. Several were
named `*_test.py`, so `pytest` collected them and blocked. Others could not
run at all any more: `get_colouring_dict` had grown a third return value,
`jacobian(sparsity=<callable>)` had become an explicit error.

Every one of those checks has been restated as something a machine can decide:

| the old script asked | the test now asserts |
| --- | --- |
| "does the refined mesh look the same?" | every surface sample of the original still lies on the refined mesh |
| "is the residual perpendicular?" | `abs(dot(normal, residual_direction))` is 1, and improves with iterations |
| "does the strain plot match the red line?" | the strain equals the analytic Green-Lagrange tensor, for all ten deformations |
| "did the fit converge?" | the fitted surface reaches the target when the target is representable |
| "does the mesh draw?" | the plotter's bounds equal the mesh bounds, and the off-screen render is not blank |

Rendering is still exercised — `conftest.py` forces PyVista off-screen and
matplotlib's Agg backend, and `test_plotting.py` takes a real screenshot — but
nothing waits for input.

## Layout

| file | covers |
| --- | --- |
| `_helpers.py` | shared mesh builders and the two tolerances (`EXACT`, `CLOSE`) |
| `conftest.py` | headless rendering setup, the `plotter` fixture |
| `test_basis_definitions.py` | partition of unity, interpolation, derivatives vs autodiff, polynomial reproduction |
| `test_utils.py` | rotations, nearest-neighbour searches, volumes, transforms, indexing helpers |
| `test_geometry.py` | the `cube` / `basic_surface` factories |
| `test_mesh_construction.py` | bases, mixed bases, collapsed elements, ids, quadrature, editing |
| `test_refine_rebase.py` | geometry preservation through refinement and basis conversion |
| `test_fixed_param_preservation.py` | constraint bookkeeping through the same two operations |
| `test_evaluation.py` | the three evaluator spellings, chunking, derivatives, normals, xi grids, drawable geometry |
| `test_embedding.py` | projecting points onto a mesh |
| `test_masked_multistate_embed.py` | `dim_mask` and multi-state fields in the embedding solve |
| `test_embedding_jvp.py` | differentiating through the embedding solve |
| `test_fitting.py` | `linear_fit`, `point_cloud_fit`, sparse Jacobians |
| `test_strain.py` | Green-Lagrange strain against ten analytic deformations |
| `test_topology.py` | element adjacency and `topomap` |
| `test_fields.py` | secondary fields |
| `test_io.py` | JSON round trips |
| `test_colouring.py` | Jacobian colouring |
| `test_plotting.py` | headless rendering |

## Conventions

- Assert against something knowable in advance: an analytic value, a
  conservation law, a round trip, or a second independent code path. If the
  only available reference is the current output, say so in the docstring and
  explain what a change would mean.
- HOMER evaluates in float32. Use `EXACT` (1e-5) for quantities that are exact
  in exact arithmetic and `CLOSE` (1e-3) for anything reached by least squares
  or Newton-Raphson. A looser tolerance than that needs a comment saying why.
- A few tolerances here are loose because the library is approximate in a way
  worth knowing about — non-uniform refinement of a Hermite or B-spline mesh
  cannot be exact (one shared nodal derivative, two children of different
  widths; a uniform knot vector, an interior knot). Those tests name the
  limit and pin its size rather than hiding it. Uniform refinement, and
  refinement of any Lagrange basis, *is* exact to float32 round-off.

Timing scripts live in `benchmarks/`, are not collected, and assert nothing.
