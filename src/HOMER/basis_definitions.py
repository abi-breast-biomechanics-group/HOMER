"""
basis_definitions.py – 1-D basis function definitions for HOMER mesh elements.

This module provides the building blocks for constructing high-order mesh
elements.  Every mesh element in HOMER is defined by a *tensor product* of
1-D basis functions – one per parametric direction.

Available 1-D bases (all :class:`Basis` instances):

* :data:`H3Basis` – cubic Hermite (C¹ continuity, 2 nodes per direction)
* :data:`L1Basis` – linear Lagrange (2 nodes per direction)
* :data:`L2Basis` – quadratic Lagrange (3 nodes per direction)
* :data:`L3Basis` – cubic Lagrange (4 nodes per direction)
* :data:`L4Basis` – quartic Lagrange (5 nodes per direction)
* :data:`B3Basis` – cubic B-spline (C² continuity, shared control points)

A basis is a *value*, not a type: each of the names above is a frozen
:class:`Basis` instance, and the directions of an element are combined with
arithmetic – ``*`` repeats a basis across directions, ``+`` concatenates
directions, and the result is a :class:`BasisGroup` (a ``tuple`` subclass, so
plain lists and tuples of bases remain valid everywhere)::

    from HOMER.basis_definitions import H3Basis, L1Basis, B3Basis

    H3Basis * 3               # tricubic-Hermite volume
    H3Basis * 2 + L1Basis     # Hermite surface extruded linearly
    2 * H3Basis + B3Basis     # the same shape, written the other way round
    H3Basis ** 3              # tensor power, a spelling of H3Basis * 3

Each basis carries:

* ``name`` – the serialisation key written into mesh JSON
* ``fn`` – the basis evaluation function ``fn(x) -> (n_pts, n_basis)``
* ``deriv`` – derivative functions ``(fn, d1_fn, d2_fn, …)``
* ``weights`` – ordered weight names, e.g. ``('x0', 'dx0', 'x1', 'dx1')``
* ``order`` – polynomial order
* ``node_locs`` – canonical node positions in [0, 1]
* ``node_fields`` – a :class:`DerivativeField` describing which derivative
  quantities each node must carry (``None`` for Lagrange bases)
* ``interpolatory`` – whether nodal parameters are field values at the nodes

Bases are interned by name in a registry (:func:`basis_by_name`,
:func:`registered_bases`), which is what lets :mod:`HOMER.io` round-trip them
through JSON – including any basis a user defines.

Typical usage::

    from HOMER.basis_definitions import H3Basis, L1Basis
    from HOMER.mesh import MeshElement

    # 2-D cubic-Hermite surface element
    elem = MeshElement(node_indexes=[0,1,2,3], basis_functions=H3Basis * 2)

    # 3-D trilinear volume element
    elem3d = MeshElement(node_indexes=list(range(8)), basis_functions=L1Basis * 3)
"""

from typing import Callable, Optional
import jax.numpy as jnp
import jax
import numpy as np
from itertools import combinations_with_replacement, product

from dataclasses import dataclass, field


deriv_fields = (
    (),
    ('du'),
    ('du', 'dv', 'dudv'),
    ('du', 'dv', 'dw', 'dudv', 'dudw', 'dvdw', 'dudvdw'),
)

DERIV_ORDER = {
        (0,):1, (1,):2, (2,):3,
        (0, 1):4, (0, 2):5, (1, 2):6,
        (0, 1, 2):7,
}

EVAL_PATTERN = {
    0:[],
    1:[(1,)],
    3:[(1, 0), (0, 1), (1,1)],
    7:[(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0, 1, 1), (1,1,1)],
    # 7:[(0,0,1), (0,1,0), (1,0,0), (0,1,1), (1,0,1), (1, 1, 0), (1,1,1)],
}


@dataclass
class AbstractField:
    """Base class for node-field descriptors.

    Tracks how many derivative *fields* a node must carry and maps that count
    to a tuple of required field names via ``_field_scaling``.
    """

    n_field: int
    _field_scaling: tuple[tuple[str]]

    def __add__(self, other: type["AbstractField"]) -> "AbstractField":
        if not isinstance(other, self.__class__):
            raise ValueError("Can only add the same fields")
        new_class = self.__class__.__new__(self.__class__)
        new_class.n_field=self.n_field + other.n_field
        return new_class
    
    def get_needed_fields(self):
        """Return the tuple of required field names for the current count."""
        return self._field_scaling[self.n_field]

@dataclass
class DerivativeField(AbstractField):
    """Descriptor for Hermite-style derivative fields on a node.

    When a mesh element uses :class:`H3Basis` in *n* parametric directions,
    each node needs an increasing set of mixed-derivative vectors:

    * 1 Hermite direction → ``('du',)``
    * 2 Hermite directions → ``('du', 'dv', 'dudv')``
    * 3 Hermite directions → ``('du', 'dv', 'dw', 'dudv', 'dudw', 'dvdw', 'dudvdw')``
    """

    n_field:int = field(default=1)
    _field_scaling:tuple[tuple[str]] = field(default=deriv_fields)

#: name -> basis, populated by :class:`Basis` construction.  This is the live
#: mapping :mod:`HOMER.io` resolves against, so a user-defined basis becomes
#: loadable the moment it is defined.
BASIS_REGISTRY: dict[str, "Basis"] = {}


def basis_by_name(name: str) -> "Basis":
    """Return the registered basis called ``name``.

    The name is the serialisation key: :mod:`HOMER.io` writes ``basis.name``
    into the mesh JSON and reads it back through here, so any basis a user
    defines round-trips as soon as it has been constructed.
    """
    try:
        return BASIS_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown basis {name!r}; registered bases are "
                       f"{sorted(BASIS_REGISTRY)}") from None


def registered_bases() -> dict[str, "Basis"]:
    """A copy of the ``name -> basis`` registry."""
    return dict(BASIS_REGISTRY)


def _register(basis: "Basis") -> None:
    """Add ``basis`` to the registry, rejecting a clashing redefinition."""
    existing = BASIS_REGISTRY.get(basis.name)
    if existing is None:
        BASIS_REGISTRY[basis.name] = basis
    elif existing._identity() != basis._identity():
        raise ValueError(
            f"A different basis is already registered as {basis.name!r}. "
            "Names are the serialisation key, so they must be unique.")


@dataclass(frozen=True, eq=False)
class Basis:
    """A single 1-D basis function definition.

    A basis is a *value*, not a type: the module-level :data:`H3Basis`,
    :data:`L1Basis`, ... are frozen instances of this class, and a mesh element
    is a tensor product of them - one per parametric direction.  Directions are
    combined with the arithmetic operators::

        H3Basis * 2 + B3Basis   # -> BasisGroup(H3Basis, H3Basis, B3Basis)
        2 * H3Basis + L1Basis   # the same
        (H3Basis + L1Basis) * 2 # -> H3, L1, H3, L1
        L3Basis ** 3            # tensor power, a spelling of L3Basis * 3

    ``*`` repeats a basis across directions and ``+`` concatenates directions;
    neither is a pointwise operation on the basis functions themselves.

    Equality and hashing are by :attr:`name`, so a basis compares equal to
    itself across a deepcopy, a pickle, and a JSON round-trip.

    Attributes
    ----------
    name : str
        Serialisation key and repr, e.g. ``'H3Basis'``.
    fn : Callable
        Basis evaluation function ``fn(x) -> ndarray (n_pts, n_basis)``.
    weights : tuple[str, ...]
        Ordered weight names, e.g. ``('x0', 'dx0', 'x1', 'dx1')``.
        Names starting with ``'dx'`` indicate derivative entries.
    deriv : tuple[Callable, ...]
        Derivative evaluation functions, ``(fn, d1_fn, d2_fn, ...)``.
    order : int
        Polynomial order of the basis.
    node_locs : tuple[float, ...]
        Canonical node positions in [0, 1].
    node_fields : AbstractField or None
        Describes the derivative quantities each node must carry.
        ``None`` for pure Lagrange bases.
    interpolatory : bool
        ``True`` when the nodal parameters *are* the field values at
        ``node_locs`` (Lagrange and Hermite bases).  ``False`` for control-net
        bases such as :data:`B3Basis`, whose parameters are control points that
        do not equal the geometry at the node location.  Used when refining or
        rebasing to decide whether a fixed nodal value may be carried across
        verbatim.

    Construction validates the definition - ``deriv[0]`` must be ``fn``, and
    ``fn`` must return one column per entry of ``weights`` - so a malformed
    basis fails where it is defined rather than as a wrong-looking mesh.
    """

    name: str
    fn: Callable
    weights: tuple[str, ...]
    deriv: tuple[Callable, ...]
    order: int
    node_locs: tuple[float, ...]
    node_fields: Optional[AbstractField] = None
    interpolatory: bool = True

    def __post_init__(self):
        #frozen, so the list -> tuple coercion has to go around __setattr__
        object.__setattr__(self, 'weights', tuple(self.weights))
        object.__setattr__(self, 'deriv', tuple(self.deriv))
        object.__setattr__(self, 'node_locs', tuple(float(l) for l in self.node_locs))

        if not self.weights:
            raise ValueError(f"{self.name}: a basis needs at least one weight")
        if not self.node_locs:
            raise ValueError(f"{self.name}: a basis needs at least one node location")
        if not self.deriv or self.deriv[0] is not self.fn:
            raise ValueError(f"{self.name}: deriv[0] must be fn itself, so that "
                             "the 0th derivative evaluates the basis")
        try:
            n_cols = self.fn(jnp.zeros(1)).shape[-1]
        except Exception as exc:
            raise ValueError(f"{self.name}: fn must accept a 1-D array of xi values "
                             f"and return (n_pts, n_basis); calling it raised {exc!r}") from exc
        if n_cols != len(self.weights):
            raise ValueError(f"{self.name}: fn returns {n_cols} columns but "
                             f"{len(self.weights)} weight names were given")
        _register(self)

    def _identity(self):
        """The content that makes two same-named bases the same definition."""
        return (self.fn, self.weights, self.deriv, self.order,
                self.node_locs, type(self.node_fields), self.interpolatory)

    # --- identity -------------------------------------------------------
    def __eq__(self, other):
        return isinstance(other, Basis) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    @property
    def __name__(self):
        """The name, so ``basis.__name__`` keeps working now that a basis is
        an instance rather than a class."""
        return self.name

    #bases are interned singletons: copying one would break identity for no gain
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (basis_by_name, (self.name,))

    # --- direction algebra ----------------------------------------------
    def __mul__(self, n: int) -> "BasisGroup":
        """``H3Basis * 3`` - the same basis in ``n`` parametric directions."""
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            return NotImplemented
        return BasisGroup((self,) * int(n))

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "BasisGroup":
        """``H3Basis ** 3`` - tensor power, a spelling of ``H3Basis * 3``."""
        return self.__mul__(n)

    def __add__(self, other) -> "BasisGroup":
        """``H3Basis + L1Basis`` - one direction of each, in order."""
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup((self,) + tuple(other))

    def __radd__(self, other) -> "BasisGroup":
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(other) + (self,))


def _as_group(obj) -> Optional["BasisGroup"]:
    """Coerce a basis, or any sequence of bases, to a group; ``None`` if it is
    not something that can sit in a tensor product."""
    if isinstance(obj, Basis):
        return BasisGroup((obj,))
    if isinstance(obj, (BasisGroup, tuple, list)):
        try:
            return BasisGroup(obj)
        except TypeError:
            return None
    return None


class BasisGroup(tuple):
    """The ordered bases of a tensor-product element, one per direction.

    A ``tuple`` subclass, so anything that already iterates, indexes, takes the
    ``len`` of, or compares a list of bases keeps working unchanged.  What it
    adds is the algebra::

        H3Basis * 2 + B3Basis     # BasisGroup(H3Basis, H3Basis, B3Basis)
        (H3Basis + L1Basis) * 2   # BasisGroup(H3Basis, L1Basis, H3Basis, L1Basis)

    Constructing one from a list, a tuple, or a bare :class:`Basis` normalises
    all three, which is how the mesh entry points accept every spelling.  The
    group itself does not cap the number of directions - a partial expression
    is free to be any length - :class:`~HOMER.mesh.element.MeshElement` is what
    requires 1, 2 or 3.
    """

    def __new__(cls, items=()):
        if isinstance(items, Basis):
            items = (items,)
        items = tuple(items)
        bad = [b for b in items if not isinstance(b, Basis)]
        if bad:
            raise TypeError(
                f"A BasisGroup holds Basis values; got {bad[0]!r}. "
                "Bases are now instances - pass H3Basis, not H3Basis().")
        return super().__new__(cls, items)

    def __add__(self, other) -> "BasisGroup":
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(self) + tuple(other))

    def __radd__(self, other) -> "BasisGroup":
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(other) + tuple(self))

    def __mul__(self, n: int) -> "BasisGroup":
        """``(H3Basis + L1Basis) * 2`` repeats the whole pattern."""
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            return NotImplemented
        return BasisGroup(tuple(self) * int(n))

    __rmul__ = __mul__

    @property
    def ndim(self) -> int:
        """Number of parametric directions."""
        return len(self)

    @property
    def interpolatory(self) -> bool:
        """``True`` only when every direction is interpolatory."""
        return all(b.interpolatory for b in self)

    def __repr__(self):
        if not self:
            return "BasisGroup()"
        parts, run = [], 1
        for prev, cur in zip(self, self[1:] + (None,)):
            if cur is not None and cur == prev:
                run += 1
                continue
            parts.append(f"{prev.name}*{run}" if run > 1 else prev.name)
            run = 1
        return " + ".join(parts)


#: Retained so ``type[AbstractBasis]`` annotations and imports keep resolving.
AbstractBasis = Basis


@jax.jit
def N2_weights(w0, w1, bp_inds):
    bp_inds = jnp.asarray(bp_inds, dtype=jnp.int32)  # (B, 2)
    def one_pair(ind):
        i, j = ind[0], ind[1]
        return w0[:, i] * w1[:, j]   # (n_pts,)

    return jax.vmap(one_pair, in_axes=0)(bp_inds)

@jax.jit
def N3_weights(w0, w1, w2, bp_inds):
    bp_inds = jnp.asarray(bp_inds, dtype=jnp.int32)  # (B, 3)
    def one_triplet(ind):
        i, j, k = ind[0], ind[1], ind[2]
        return w0[:, i] * w1[:, j] * w2[:, k]  # (n_pts,)
    return jax.vmap(one_triplet, in_axes=0)(bp_inds)

# def N2_weights(w0, w1, bp_inds) -> jnp.ndarray:
#     # BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
#     #          [2, 0], [3, 0], [2, 1], [3, 1],
#     #          [0, 2], [1, 2], [0, 3], [1, 3],
#     #          [2, 2], [3, 2], [2, 3], [3, 3]]
#     BPInd = bp_inds
#     w_list = [w0[:, ii[0]] * w1[:, ii[1]] for ii in BPInd]
#     weights = jnp.vstack(w_list)
#     return weights
#
# def N3_weights(w0, w1, w2, bp_inds) -> jnp.ndarray:
#     BPInd = bp_inds
#     w_list = [w0[:, ii[0]] * w1[:, ii[1]] * w2[:, ii[2]] for ii in BPInd]
#     weights = jnp.vstack(w_list)
#     return weights

######################################## BASIS FUNCS

def B3(x) -> jnp.ndarray:
    """
    Cubic bezier basis function.
    :param x: points to interpolaet

    :param x: points to interpolate
    :return: basis weights
    """

    return jnp.column_stack((
        (1 - x)**3/6,
        (3*x**3 - 6*x**2 + 4)/6,
        (-3*(x**3) + 3*x**2 + 3 * x + 1)/6,
        (x**3)/6,
    ))
def B3d1(x) -> jnp.ndarray:
    return jnp.column_stack((
        -(1 - x)**2 / 2,
        (3*x**2 - 4*x) / 2,
        (-3*x**2 + 2*x + 1) / 2,
        (x**2) / 2,
    ))
def B3d1d1(x) -> jnp.ndarray:
    return jnp.column_stack((
        1 - x,
        3 * x - 2,
        -3 * x + 1,
        x,
    ))


def L1(x) -> jnp.ndarray:
    """
    Linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    return jnp.array([1. - x, x]).T

def L1d1(x) -> jnp.ndarray:
    """
    First derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    W = jnp.ones((x.shape[0], 2))
    W = W.at[:,0].add(-2)
    return jnp.array(W)

def L1d1d1(x) -> jnp.ndarray:
    """
    Second derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    return jnp.zeros((x.shape[0], 2))

def H3(x:jnp.ndarray) -> jnp.ndarray:
    """
    The cubic-Hermite basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    x2 = x*x
    Phi = jnp.column_stack([
        1-3*x2+2*x*x2,
        x*(x-1)*(x-1),
        x2*(3-2*x),
        x2*(x-1)
    ])
    return Phi

def H3d1(x: jnp.ndarray) -> jnp.ndarray:
    """
    First derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    x2 = x*x
    Phi = jnp.column_stack([ \
        6*x*(x-1),
        3*x2-4*x+1,
        6*x*(1-x),
        x*(3*x-2)])
    return Phi

def H3d1d1(x) -> jnp.ndarray:
    """
    Second derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    Phi = jnp.column_stack([ \
        12*x-6,
        6*x-4,
        6-12*x,
        6*x-2]) 
    return Phi

def L2(x):
    """
    Quadratic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 3)
    """
    L1, L2 = 1-x, x
    Phi = jnp.array([
        L1 * (2.0 * L1 - 1),
        4.0 * L1 * L2,
        L2 * (2.0 * L2 - 1)])
    return Phi.T

def L2d1(x):
    """
    First derivative of the quadratic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 3)
    """
    L1 = 1-x
    return jnp.array([
        1.0 - 4.0 * L1,
        4.0 * L1 - 4.0 * x,
        4.0 * x - 1.]).T

# .. todo: L2dxdx

def L3(x):
    """
    Cubic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    L1, L2 = 1-x, x
    sc = 9./2.
    return jnp.array([
        0.5*L1*(3*L1-1)*(3*L1-2),
        sc*L1*L2*(3*L1-1),
        sc*L1*L2*(3*L2-1),
        0.5*L2*(3*L2-1)*(3*L2-2)]).T

def L3d1(x):
    """
    First derivative of the cubic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 4)
    """
    L1 = x*x
    return jnp.array([
        -(27.*L1-36.*x+11.)/2.,
        (81.*L1-90.*x+18.)/2.,
        -(81.*L1-72.*x+9.)/2.,
        (27.*L1-18.*x+2.)/2.]).T

# .. todo: L3dxdx

def L4(x):
    """
    Quartic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 5)
    """
    sc = 1/3.
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    return jnp.array([
        sc*(32*x4-80*x3+70*x2-25*x+3),
        sc*(-128*x4+288*x3-208*x2+48*x),
        sc*(192*x4-384*x3+228*x2-36*x),
        sc*(-128*x4+224*x3-112*x2+16*x),
        sc*(32*x4-48*x3+22*x2-3*x)]).T

def L4d1(x):
    """
    First derivative of the quartic lagrange basis function.
    
    :param x: points to interpolate
    :type x: numpy array (npoints)
    :return: basis weights
    :rtype: numpy array(npoints, 5)
    """
    sc = 1/3.
    x2 = x*x
    x3 = x2*x
    return jnp.array([ \
        sc*(128*x3-240*x2+140*x-25), \
        sc*(-512*x3+864*x2-416*x+48), \
        sc*(768*x3-1152*x2+456*x-36), \
        sc*(-512*x3+672*x2-224*x+16), \
        sc*(128*x3-144*x2+44*x-3)]).T

H3Basis = Basis(
    name='H3Basis',
    fn=H3,
    weights=('x0', 'dx0', 'x1', 'dx1'), #then this records the derivatives
    deriv=(H3, H3d1, H3d1d1),
    order=3,
    node_locs=(0, 1),
    node_fields=DerivativeField(),
)
"""Cubic Hermite basis (C¹ continuity, 2 nodes, 4 weights per direction).

Each node contributes a *position* and a *tangent derivative*:
``('x0', 'dx0', 'x1', 'dx1')``.  Requires each :class:`~HOMER.mesh.node.MeshNode`
to carry Hermite derivative fields (``du``, ``dv``, … depending on the element
dimensionality).

Best choice for smooth geometry where derivative continuity across element
boundaries is important.
"""

L1Basis = Basis(
    name='L1Basis',
    fn=L1,
    weights=('x0', 'x1'),
    deriv=(L1, L1d1, L1d1d1),
    order=1,
    node_locs=(0, 1),
)
"""Linear Lagrange basis (C⁰ continuity, 2 nodes per direction).

Each node contributes only a *position* weight.  No derivative fields are
required on the associated :class:`~HOMER.mesh.node.MeshNode` objects.

Useful for coarse linear meshes that are subsequently
:meth:`~HOMER.mesh.field.MeshField.rebase`-d to a higher-order basis.
"""

L2Basis = Basis(
    name='L2Basis',
    fn=L2,
    weights=('x0', 'x1', 'x2'),
    deriv=(L2, L2d1),
    order=2,
    node_locs=(0, 1/2, 2/2),
)
"""Quadratic Lagrange basis (C⁰ continuity, 3 nodes per direction).

Provides second-order accuracy with 3 nodes per direction and no derivative
fields on nodes.
"""

L3Basis = Basis(
    name='L3Basis',
    fn=L3,
    weights=('x0', 'x1', 'x2', 'x3'),
    deriv=(L3, L3d1),
    order=3,
    node_locs=(0/3, 1/3, 2/3, 3/3),
)
"""Cubic Lagrange basis (C⁰ continuity, 4 nodes per direction).

Third-order accuracy with uniformly-spaced node positions at 0, 1/3, 2/3, 1.
No derivative fields required on nodes.
"""

L4Basis = Basis(
    name='L4Basis',
    fn=L4,
    weights=('x0', 'x1', 'x2', 'x3', 'x4'),
    deriv=(L4, L4d1),
    order=4,
    node_locs=(0/4, 1/4, 2/4, 3/4, 4/4),
)
"""Quartic Lagrange basis (C⁰ continuity, 5 nodes per direction).

Fourth-order accuracy with uniformly-spaced node positions at
0, 1/4, 2/4, 3/4, 1.  No derivative fields required on nodes.
"""

B3Basis = Basis(
    name='B3Basis',
    fn=B3,
    weights=('x0', 'x1', 'x2', 'x3'),
    deriv=(B3, B3d1, B3d1d1),
    order=3,
    node_locs=(-1, 0, 1, 2), #hat t do this # yeah buddy get down with this.
    interpolatory=False, #shared control points, not interpolated nodal values
)
"""Cubic B-spline basis (C² continuity, 4 control points per element per
direction, each shared across neighbouring elements).
"""

LAGRANGE_BASES = {b.order: b for b in (L1Basis, L2Basis, L3Basis, L4Basis)}


def Lagrange(order: int) -> Basis:
    """The Lagrange basis of the requested order.

    ``Lagrange(3) is L3Basis``.  Useful where the order is a variable::

        mesh.rebase(Lagrange(order) * 3)
    """
    try:
        return LAGRANGE_BASES[order]
    except KeyError:
        raise ValueError(f"No Lagrange basis of order {order}; HOMER defines "
                         f"orders {sorted(LAGRANGE_BASES)}") from None
