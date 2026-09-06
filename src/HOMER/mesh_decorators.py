"""
mesh_decorators.py – Class and method decorators for HOMER mesh evaluation.

Provides:

* :func:`wide_eval` – marks a method as "wide-evaluatable".
* :func:`expand_wide_evals` – class decorator that automatically generates
  ``*_in_every_element`` and ``*_ele_xi_pair`` variants for every
  ``@wide_eval`` method, plus backwards-compatibility aliases.
* :func:`_chunked_vmap` – the ``jax.lax.scan``-over-chunks batching used by
  those generated variants so that very wide evaluations don't OOM.
* :func:`depreciation` – wraps a function to emit a deprecation warning on
  call.
* Helper functions for generating and caching ``.pyi`` stub files alongside
  the decorated class (``_get_class_hash``, ``_write_pyi``, etc.).

The ``@expand_wide_evals`` decorator is applied to :class:`~HOMER.mesher.MeshField`
to automatically extend its API without boilerplate.
"""

import ast
import jax
import jax.numpy as jnp
import numpy as np
import hashlib
import inspect
from pathlib import Path
import logging
import traceback
from functools import partial

_PYI_FORMAT_VERSION = 2

#: Rows (i.e. individual point evaluations) pushed through a single vmapped
#: chunk by the generated ``*_in_every_element`` / ``*_ele_xi_pair`` wrappers.
#: Set ``mesh.eval_chunk_size`` (or pass ``chunk_size=``) to override per mesh
#: or per call; a falsy value restores the old unchunked ``jax.vmap``.
DEFAULT_EVAL_CHUNK_SIZE = 100_000

#: Whether chunked evaluations rematerialise their intermediates on the
#: backward pass instead of stacking them (see :func:`_chunked_vmap`).  Off by
#: default: it only pays under ``jax.grad``.  Override with ``mesh.eval_remat``
#: or a per-call ``remat=``.
DEFAULT_EVAL_REMAT = False

def wide_eval(fn):
    fn._is_derived = True
    return fn

def _resolve_chunk_size(obj, chunk_size):
    """``None`` means "use the mesh default"; anything falsy disables chunking."""
    if chunk_size is None:
        return getattr(obj, "eval_chunk_size", DEFAULT_EVAL_CHUNK_SIZE)
    return chunk_size

def _resolve_remat(obj, remat):
    """``None`` means "use the mesh default"."""
    if remat is None:
        return getattr(obj, "eval_remat", DEFAULT_EVAL_REMAT)
    return remat

def _n_xi(arg):
    """Best-effort count of xi points held by a positional argument."""
    shape = getattr(arg, "shape", None)
    if shape is None:
        shape = np.shape(arg)
    return int(shape[0]) if len(shape) >= 2 else 1

def _chunked_vmap(fn, mapped_args, chunk_size, remat=False):
    """``jax.vmap(fn)(*mapped_args)``, evaluated ``chunk_size`` rows at a time.

    Every entry of *mapped_args* is mapped over its leading axis (all of equal
    length).  Batches that already fit in one chunk take the plain
    :func:`jax.vmap` path; larger ones are padded up to a whole number of
    chunks and pushed through :func:`jax.lax.scan`, so peak memory scales with
    *chunk_size* instead of with the full batch.  The returned values are
    identical either way — only the memory profile changes.

    Under reverse-mode AD the scan still stacks one set of residuals per row,
    which is O(n_rows) memory.  Setting *remat* wraps the chunk body in
    :func:`jax.checkpoint`, so the backward pass re-runs each chunk forward to
    regenerate its intermediates rather than keeping them: residuals collapse
    to the (already stored) scan inputs at the cost of one extra forward
    evaluation per chunk.  It costs ~10% on a forward-only evaluation, so it is
    only worth setting when differentiating a wide evaluation.
    """
    mapped_args = tuple(jnp.asarray(a) for a in mapped_args)
    n = mapped_args[0].shape[0]
    body = jax.vmap(fn)
    if remat:
        # checkpoint the whole vmapped chunk, not fn: one rematerialised
        # region per chunk keeps the batched kernel fused.
        body = jax.checkpoint(body)
    if not chunk_size or n <= chunk_size:
        return body(*mapped_args)

    n_chunks = -(-n // chunk_size)
    pad = n_chunks * chunk_size - n
    if pad:  # repeat the last row: padding then never indexes out of range
        mapped_args = tuple(
            jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)], axis=0) for a in mapped_args
        )
    stacked = tuple(a.reshape(n_chunks, chunk_size, *a.shape[1:]) for a in mapped_args)

    def scan_body(carry, chunk_args):
        return carry, body(*chunk_args)

    _, out = jax.lax.scan(scan_body, None, stacked)
    return jax.tree_util.tree_map(
        lambda o: o.reshape(n_chunks * chunk_size, *o.shape[2:])[:n], out
    )

def depreciation(fn):
    def new_fn(*a, **kw):
        traceback.print_stack()
        logging.warning(f"This old naming order is depreciated, and may be removed in a future update")
        return(fn(*a, **kw))
    return new_fn

def make_iee(name):
    # @partial(jax.jit, static_argnames=['self' 'othr'])
    def iee(self, *a, fit_params=None, chunk_size=None, remat=None, **kw): 
        """Evaluates the base function in every element of the mesh

        ``chunk_size`` caps how many point evaluations are held in flight at
        once (``None`` uses ``mesh.eval_chunk_size``, ``0`` disables chunking).
        Because every element is evaluated at every xi, the elements are
        scanned over in groups of ``chunk_size // n_xi``.  ``remat`` trades
        recomputation for memory on the backward pass (``None`` uses
        ``mesh.eval_remat``).
        """

        if fit_params is None:
            fit_params = self.optimisable_param_array
        new_fn = getattr(self, name)
        chunk_size = _resolve_chunk_size(self, chunk_size)
        if chunk_size and a:
            chunk_size = max(1, chunk_size // _n_xi(a[0]))
        mapped = _chunked_vmap(
            lambda e: new_fn(e, *a, fit_params=fit_params, **kw),
            (jnp.arange(len(self.elements)),),
            chunk_size,
            _resolve_remat(self, remat),
        )
        return mapped.reshape(-1, *mapped.shape[2:])
    return iee

def make_ele_xi_pair(name):  
    # @partial(jax.jit, static_argnames=['self', 'othr'])
    def ele_xi_pair(self, eles, xis, *a, fit_params=None, chunk_size=None, remat=None, **kw):
        """
        Evaluates the base function in pairs of ele_xi_lists

        ``chunk_size`` pairs are evaluated per vmapped chunk (``None`` uses
        ``mesh.eval_chunk_size``, ``0`` disables chunking), which keeps peak
        memory flat for very long ele/xi lists.  ``remat`` trades recomputation
        for memory on the backward pass (``None`` uses ``mesh.eval_remat``).
        """
        if fit_params is None:
            fit_params = self.optimisable_param_array
        new_fn = getattr(self, name)
        eval_e = jnp.atleast_1d(jnp.array(eles))
        eval_xi = jnp.atleast_2d(jnp.array(xis))
        out_sorted = _chunked_vmap( #vmap original over every (element, xi) pair in sorted order
            lambda single_e, single_xi: new_fn(single_e, single_xi, *a, fit_params=fit_params, **kw),
            (eval_e, eval_xi),
            _resolve_chunk_size(self, chunk_size),
            _resolve_remat(self, remat),
        ).squeeze()
        return out_sorted

    return ele_xi_pair

def _get_class_hash(cls) -> str:
    sources = [f"pyi_format:{_PYI_FORMAT_VERSION}"]
    for name, val in sorted(vars(cls).items()):
        if callable(val):
            try:
                sources.append(inspect.getsource(val))
            except OSError:
                pass
    return hashlib.md5("\n".join(sources).encode()).hexdigest()

def _extract_hash(pyi_path: Path) -> str | None:
    """Extract the hash comment from an existing .pyi file."""
    if not pyi_path.exists():
        return None
    first_line = pyi_path.read_text().splitlines()[0]
    if first_line.startswith("# hash:"):
        return first_line.removeprefix("# hash:").strip()
    return None

def _annotation_str(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    # handles generics like list[int], int | None, etc.
    return str(annotation)

def _extract_import_lines(source_file: Path) -> list[str]:
    """Extract top-level import statements from a Python source file."""
    try:
        source = source_file.read_text()
        tree = ast.parse(source)
        stmts = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                stmt = ast.get_source_segment(source, node)
                if stmt:
                    stmts.append(stmt)
        return stmts
    except (SyntaxError, OSError):
        return []

def _extract_instance_annotations(class_node: ast.ClassDef) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith('_'):
                annotations[node.target.id] = ast.unparse(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.AnnAssign):
                    target = stmt.target
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        if not target.attr.startswith('_'):
                            annotations.setdefault(target.attr, ast.unparse(stmt.annotation))
    return annotations

def _write_pyi(cls, pyi_path: Path, current_hash: str):
    """Write a .pyi stub file for the decorated class and all other top-level module symbols."""
    source_file = Path(inspect.getfile(cls))
    import_lines = _extract_import_lines(source_file)

    lines = [
        f"# hash: {current_hash}",
        "# Auto-generated by @expand_wide_evals — do not edit manually",
        "",
    ] + import_lines + [""]

    try:
        source = source_file.read_text()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        tree = None

    if tree is not None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == cls.__name__:
                    # Use runtime introspection for the decorated class to capture generated methods
                    lines.append(f"class {cls.__name__}:")
                    annotations = _extract_instance_annotations(node)
                    for attr_name, ann in sorted(annotations.items()):
                        lines.append(f"    {attr_name}: {ann}")
                    if annotations:
                        lines.append("")
                    for name, val in sorted(vars(cls).items()):
                        if callable(val):
                            try:
                                sig = inspect.signature(val)
                                lines.append(f"    def {name}{sig}: ...")
                            except (ValueError, TypeError):
                                lines.append(f"    def {name}(self, *args, **kwargs): ...")
                else:
                    # Use AST for all other classes
                    bases = [ast.unparse(b) for b in node.bases]
                    base_str = f"({', '.join(bases)})" if bases else ""
                    lines.append(f"class {node.name}{base_str}:")
                    methods = [n for n in ast.iter_child_nodes(node)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    if not methods:
                        lines.append("    ...")
                    else:
                        for method in methods:
                            args_str = ast.unparse(method.args)
                            ret = f" -> {ast.unparse(method.returns)}" if method.returns else ""
                            lines.append(f"    def {method.name}({args_str}){ret}: ...")
                lines.append("")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args_str = ast.unparse(node.args)
                ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                lines.append(f"def {node.name}({args_str}){ret}: ...")
                lines.append("")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('_'):
                        lines.append(f"{target.id}: ...")
                        lines.append("")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if not node.target.id.startswith('_'):
                    ann_str = ast.unparse(node.annotation)
                    lines.append(f"{node.target.id}: {ann_str}")
                    lines.append("")
    else:
        # Fallback: write only the decorated class
        lines.append(f"class {cls.__name__}:")
        for name, val in sorted(vars(cls).items()):
            if callable(val):
                try:
                    sig = inspect.signature(val)
                    lines.append(f"    def {name}{sig}: ...")
                except (ValueError, TypeError):
                    lines.append(f"    def {name}(self, *args, **kwargs): ...")

    contents = "\n".join(lines).rstrip("\n") + "\n"
    with pyi_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(contents)
    print(f"[expand_wide_evals] Updated {pyi_path}")

def expand_wide_evals(cls:"type[MeshField]"):
    """Iterates through the class definition, defining a range of methods to evaluate functions."""
    for name, val in list(vars(cls).items()):
        if callable(val) and getattr(val, '_is_derived', False):
            setattr(cls, f"{name}_in_every_element", make_iee(name))
            setattr(cls, f"{name}_ele_xi_pair", make_ele_xi_pair(name))

    # backwards compatibility for naming.
    cls.evaluate_ele_xi_pair_deriv_embeddings = depreciation(cls.evaluate_deriv_embeddings_ele_xi_pair)
    cls.evaluate_ele_xi_pair_embeddings = depreciation(cls.evaluate_embeddings_ele_xi_pair)
    cls.evaluate_ele_xi_pair_normals = depreciation(cls.evaluate_normals_ele_xi_pair)
    # cls.strain_tensor_in_ele_xi_pairs = depreciation(cls.strain_tensor_ele_xi_pair)
    # cls.strain_tensor_iee = depreciation(cls.strain_tensor_in_every_element)

    # --- Stub generation ---
    try:
        source_file = Path(inspect.getfile(cls))
        pyi_path = source_file.with_suffix(".pyi")  # inline stub, alongside the .py file
        current_hash = _get_class_hash(cls)
        stored_hash = _extract_hash(pyi_path)
        if current_hash != stored_hash:
            _write_pyi(cls, pyi_path, current_hash)
    except (TypeError, OSError):
        pass  # skip if source is unavailable (e.g. REPL, frozen)
    return cls
