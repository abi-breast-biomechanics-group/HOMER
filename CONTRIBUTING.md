# Contributing to HOMER

HOMER is developed at the Auckland Bioengineering Institute. Outside
contributions, bug reports and questions are welcome.

## Reporting a problem

Open an issue at
<https://github.com/abi-breast-biomechanics-group/HOMER/issues>. A report is
much easier to act on with:

- what you ran — ideally a mesh small enough to paste, or a saved `.json`;
- what happened, including the full traceback;
- what you expected instead;
- `python -c "import jax; print(jax.__version__, jax.devices())"` and your OS.

Numerical disagreements are worth reporting even when small: HOMER evaluates
in float32, so state the tolerance you expected and why.

## Asking a question

Use the issue tracker for those too, and label the issue `question`. There is
no separate mailing list or chat.

## Making a change

```bash
git clone https://github.com/abi-breast-biomechanics-group/HOMER
cd HOMER
pip install -e ".[dev]"
pytest
```

The suite takes about two minutes, runs headless, and needs no input. Please
run it before opening a pull request, from a branch off `development`.

A change is easier to merge when it:

- **comes with a test that would have failed before it.** The suite's
  convention is to assert against something knowable in advance — an analytic
  value, a conservation law, a round trip, or a second independent code path.
  `tests/README.md` describes the layout and the two tolerances (`EXACT`,
  `CLOSE`);
- **keeps the public surface honest.** `tests/test_public_api.py` pins what
  `HOMER` exports;
- **documents itself where the docs already are.** A new capability needs a
  how-to page under `docs/how-to/`; a new module needs an entry under
  `docs/api/`. Docstrings are reST (`:param x:`, `:class:`Foo``) and are
  rendered by mkdocstrings — `python -m mkdocs build --strict` must stay
  clean;
- **does not add a compatibility shim.** Obsolete paths are removed rather
  than deprecated.

## Adding a basis

A `Basis` validates itself on construction and registers itself by name, so a
user-defined basis round-trips through `HOMER.io` without the reader needing
to know about it. `tests/test_basis_definitions.py` checks partition of
unity, interpolation, derivatives against autodiff, and polynomial
reproduction — a new basis should pass the same checks.

## Code of conduct

Be decent to each other. Report unacceptable behaviour to the maintainers via
the issue tracker or directly.
