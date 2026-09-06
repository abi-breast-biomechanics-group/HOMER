# Benchmarks

Timing scripts. These are **not** tests: they have no pass/fail condition, they
allocate a lot of memory, and they take minutes to run. `pytest` does not
collect this directory.

Run one directly:

```bash
python benchmarks/bench_evaluation.py
python benchmarks/bench_embedding.py
python benchmarks/bench_jacobian.py
```

Each prints seconds per iteration. Compare against a previous run on the same
machine; absolute numbers mean nothing across machines or JAX versions.

They grew out of `tests/accel_test_embed.py`, `tests/test_eval_time.py` and the
timing half of `tests/point_to_plane_fit_test.py`, which lived in the test
directory but asserted nothing.
