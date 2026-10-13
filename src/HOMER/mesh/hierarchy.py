"""
hierarchy.py - truncated hierarchical (THB) spaces over locally refined elements.

A locally refined mesh is a set of leaf elements, each a cell of some level of
the lattice the mesh was refined from (:attr:`MeshElement.box`).  Level *k* has
its own uniform space of the element's basis, and the spaces are nested: a
level-*k* function is a combination of level-*k + 1* ones.  The leaves fix the
region refined to each level, Omega^k, and with it the THB space:

* a level-*k* function is *active* -- a DOF -- when its support lies in Omega^k
  but not wholly in the part refined further;
* an active function is *truncated*: written at each finer level, it drops the
  part carried by the finer functions whose support lies in Omega^(k+1).

The active functions sum to one (for Hermite, the value functions do), are
linearly independent, and span a space containing every coarser one.  Restricted
to a leaf, each is a combination of the leaf's own slots, which is the leaf's
parameter map: :meth:`Hierarchy.extraction` computes it.

For a nodal basis a truncated coarse function is the fine nodal function where
the two meet, so this reproduces the hanging-node construction; for ``B3`` it is
THB-splines.  Cells are ``(component, level, index)``, positions are lattice
coordinates at the function's own level, and a node is ``(component, level,
position)``.
"""

import functools
import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from HOMER.basis_definitions import BasisGroup
from HOMER.mesh.element_eval import make_weight_eval

MAX_DENOMINATOR = 10**6


def _snap(values: np.ndarray) -> np.ndarray:
    """Snap to the small rationals a basis evaluated at rational points gives."""
    values = np.where(np.abs(values) < 1e-12, 0.0, values)
    nonzero = values != 0
    values[nonzero] = [float(Fraction(v).limit_denominator(MAX_DENOMINATOR)) for v in values[nonzero]]
    return values


@functools.lru_cache
def prolongation(bases: BasisGroup, bp_inds: tuple, offset: tuple) -> np.ndarray:
    """``(n_slots, n_slots)``: a child's slot coefficients from its parent's.

    The child is the half of the parent at *offset*, one 0 or 1 per direction.
    Found by collocation -- the child's basis solved against the parent's at
    the same points -- which is exact for any basis whose refined space holds
    the coarse one, and needs nothing from the basis but its functions.

    :raises ValueError: If the basis is not nested under halving.
    """
    evaluate = make_weight_eval(bases, bp_inds)
    grid = [np.linspace(0.05, 0.95, len(b.weights) + 2) for b in bases]
    xi = np.stack(np.meshgrid(*grid, indexing='ij'), -1).reshape(-1, len(bases))
    with jax.enable_x64(True):
        child = np.asarray(evaluate(jnp.asarray(xi))).T
        parent = np.asarray(evaluate(jnp.asarray((np.asarray(offset) + xi) / 2))).T
    rows, *_ = np.linalg.lstsq(child, parent, rcond=None)
    if np.abs(child @ rows - parent).max() > 1e-9:
        raise ValueError(f"{bases!r} is not nested under halving, so it cannot be refined locally")
    return _snap(rows)


class Hierarchy:
    """The THB space a set of leaves defines.

    :param element: Any element of the mesh; every element shares its basis.
    :param leaves: Every leaf's box, ``(component, origin, size)``.
    :param widths: ``{(component, root index): widths}``, the derivative
        scaling of each level-0 cell, per direction.
    """

    def __init__(self, element, leaves, widths):
        self.element = element
        self.bases = element.basis_functions
        self.bp_inds = tuple(map(tuple, element.BasisProductInds))
        self.orders = element.derivative_orders
        self.fields = element.slot_fields
        self.widths = widths
        self.locs = [[Fraction(float(x)).limit_denominator(MAX_DENOMINATOR) for x in b.node_locs]
                     for b in self.bases]
        shape = [len(b.node_locs) for b in self.bases]
        lattice = [np.unravel_index(j, shape, order='F') for j in range(element.num_nodes)]
        #slot s sits at this offset into its cell, and is this field there
        self.slot_offset = [tuple(self.locs[d][i] for d, i in enumerate(lattice[s // len(self.fields)]))
                            for s in range(len(element.slots))]
        self.slot_field = [self.fields[s % len(self.fields)] for s in range(len(element.slots))]

        self._seen = {}
        self.leaves = [self.cell(box) for box in leaves]
        self.omega = {}
        for c, level, index in self.leaves:
            for k in range(level + 1):
                self.omega.setdefault(k, set()).add((c, tuple(i >> (level - k) for i in index)))
        self.roots = self.omega[0]
        #the cells refined further than their own level
        self.deeper = {k: {(c, tuple(i >> 1 for i in index)) for c, index in self.omega.get(k + 1, ())}
                       for k in self.omega}

    @staticmethod
    def cell(box) -> tuple:
        c, origin, size = box
        level = int(round(np.log2(1 / float(size))))
        return c, level, tuple(int(x * 2 ** level) for x in origin)

    def _support(self, c, level, position) -> set:
        per_direction = [[p - x for x in locs if (p - x).denominator == 1]
                         for p, locs in zip(position, self.locs)]
        cells = {(c, tuple(int(i) for i in idx)) for idx in itertools.product(*per_direction)}
        return {cell for cell in cells if (cell[0], tuple(i >> level for i in cell[1])) in self.roots}

    def _inside(self, c, level, position, deeper: bool) -> bool:
        key = (c, level, position, deeper)
        if key not in self._seen:
            region = (self.deeper if deeper else self.omega).get(level, set())
            self._seen[key] = self._support(c, level, position) <= region
        return self._seen[key]

    def active(self, c, level, position) -> bool:
        return (self._inside(c, level, position, False)
                and not self._inside(c, level, position, True))

    def _scale(self, c, level, index, s) -> float:
        root = tuple(i >> level for i in index)
        return float(np.prod(np.where(self.orders[s], self.widths[(c, root)] / 2 ** level, 1.0)))

    def extraction(self, leaf) -> tuple[np.ndarray, list]:
        """``(table, functions)``: the leaf's slots as combinations of THB functions.

        ``table[s, t]`` is the weight of function ``functions[t]`` --
        ``((component, level, position), field)`` -- in slot *s*.  Built down
        the leaf's ancestors: at each level the coarser functions are carried
        into the child, the ones whose support lies in that level's refined
        region are truncated away, and the level's own active functions join.
        """
        c, level, index = leaf
        table = np.zeros((len(self.slot_offset), 0))
        functions = []
        for k in range(level + 1):
            here = tuple(i >> (level - k) for i in index)
            keys = [tuple(Fraction(i) + x for i, x in zip(here, offset)) for offset in self.slot_offset]
            if k:
                offset = tuple(i & 1 for i in here)
                table = prolongation(self.bases, self.bp_inds, offset) @ table
                truncated = [self._inside(c, k, key, False) for key in keys]
                table[truncated] = 0.0
            joining = [s for s, key in enumerate(keys) if self.active(c, k, key)]
            column = np.zeros((len(keys), len(joining)))
            column[joining, range(len(joining))] = [self._scale(c, k, here, s) for s in joining]
            table = np.hstack([table, column])
            functions += [((c, k, keys[s]), self.slot_field[s]) for s in joining]

        keep = np.abs(table).max(0) > 1e-12
        return table[:, keep], [f for f, kept in zip(functions, keep) if kept]
