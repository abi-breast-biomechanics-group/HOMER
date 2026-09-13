"""
basis_definitions.py – 1-D basis function definitions for HOMER mesh elements.

This module provides the building blocks for constructing high-order mesh
elements.  Every mesh element in HOMER is defined by a *tensor product* of
1-D basis functions – one per parametric direction.

Available 1-D bases (all :class:`Basis` instances):

* :data:`H3` – cubic Hermite (C¹ continuity, 2 nodes per direction)
* :data:`L1` – linear Lagrange (2 nodes per direction)
* :data:`L2` – quadratic Lagrange (3 nodes per direction)
* :data:`L3` – cubic Lagrange (4 nodes per direction)
* :data:`L4` – quartic Lagrange (5 nodes per direction)
* :data:`B3` – cubic B-spline (C² continuity, shared control points)

A basis is a *value*, not a type: each of the names above is a frozen
:class:`Basis` instance, and the directions of an element are combined with
arithmetic – ``*`` joins directions the way the element takes their outer
product, ``**`` is the tensor power, and against an ``int`` ``*`` repeats a
direction.  The result is a :class:`BasisGroup` (a ``tuple`` subclass, so
plain lists and tuples of bases remain valid everywhere)::

    from HOMER.basis_definitions import H3, L1, B3

    H3 ** 3              # tricubic-Hermite volume
    H3**2 * L1           # Hermite surface extruded linearly
    H3 * H3 * L1         # the same shape, written out
    H3 * 3               # a spelling of H3 ** 3

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

    from HOMER.basis_definitions import H3, L1
    from HOMER.mesh import MeshElement

    # 2-D cubic-Hermite surface element
    elem = MeshElement(node_indexes=[0,1,2,3], basis_functions=H3 * 2)

    # 3-D trilinear volume element
    elem3d = MeshElement(node_indexes=list(range(8)), basis_functions=L1 * 3)
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

    When a mesh element uses :class:`H3` in *n* parametric directions,
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

#: The pre-1.0 serialisation keys.  Mesh JSON written before the bases were
#: renamed carries ``'H3Basis'`` where it now carries ``'H3'``, so those files
#: keep loading.  Deliberately not in :data:`BASIS_REGISTRY`: these are not
#: names a basis may be registered under, only names a file may be read by.
_LEGACY_NAMES = {f'{short}Basis': short
                 for short in ('H3', 'L1', 'L2', 'L3', 'L4', 'B3')}


def basis_by_name(name: str) -> "Basis":
    """Return the registered basis called ``name``.

    The name is the serialisation key: :mod:`HOMER.io` writes ``basis.name``
    into the mesh JSON and reads it back through here, so any basis a user
    defines round-trips as soon as it has been constructed.

    The pre-1.0 spellings (``'H3Basis'`` for ``'H3'``, and so on) still
    resolve, so mesh files written before the rename keep loading.

    :param name:
        The registered name, e.g. ``'H3'``.

    :returns:
        The basis instance.

    :raises KeyError:
        If no basis is registered under that name; the message lists the ones
        that are.
    """
    try:
        return BASIS_REGISTRY[name]
    except KeyError:
        pass
    try:
        return BASIS_REGISTRY[_LEGACY_NAMES[name]]
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

    A basis is a *value*, not a type: the module-level :data:`H3`,
    :data:`L1`, ... are frozen instances of this class, and a mesh element
    is a tensor product of them - one per parametric direction.  Directions are
    combined with ``*``, the operator nearest the outer product that the
    element actually takes::

        H3**2 * B3    # -> BasisGroup(H3, H3, B3)
        H3 * 2 * B3   # the same; an int repeats a direction
        (H3 * L1)**2  # -> H3, L1, H3, L1
        L3 ** 3            # tensor power, a spelling of L3 * 3

    ``*`` joins directions, and against an ``int`` it repeats them; ``**`` is
    the tensor power.  Neither is a pointwise operation on the basis functions
    themselves.

    Equality and hashing are by :attr:`name`, so a basis compares equal to
    itself across a deepcopy, a pickle, and a JSON round-trip.

    :ivar name: str
        Serialisation key and repr, e.g. ``'H3'``.
    :ivar fn: Callable
        Basis evaluation function ``fn(x) -> ndarray (n_pts, n_basis)``.
    :ivar weights: tuple[str, ...]
        Ordered weight names, e.g. ``('x0', 'dx0', 'x1', 'dx1')``.
        Names starting with ``'dx'`` indicate derivative entries.
    :ivar deriv: tuple[Callable, ...]
        Derivative evaluation functions, ``(fn, d1_fn, d2_fn, ...)``.
    :ivar order: int
        Polynomial order of the basis.
    :ivar node_locs: tuple[float, ...]
        Canonical node positions in [0, 1].
    :ivar node_fields: AbstractField or None
        Describes the derivative quantities each node must carry.
        ``None`` for pure Lagrange bases.
    :ivar interpolatory: bool
        ``True`` when the nodal parameters *are* the field values at
        ``node_locs`` (Lagrange and Hermite bases).  ``False`` for control-net
        bases such as :data:`B3`, whose parameters are control points that
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
    def __mul__(self, other) -> "BasisGroup":
        """``H3 * L1`` - one direction of each, in order.

        Against an ``int``, ``H3 * 3`` repeats the basis across three
        directions instead.
        """
        if _is_count(other):
            return BasisGroup((self,) * int(other))
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup((self,) + tuple(other))

    def __rmul__(self, other) -> "BasisGroup":
        """``3 * H3``, and the reflected case where a plain list or tuple
        sits on the left - which prepends, so the order is not the mirror of
        :meth:`__mul__`."""
        if _is_count(other):
            return BasisGroup((self,) * int(other))
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(other) + (self,))

    def __pow__(self, n: int) -> "BasisGroup":
        """``H3 ** 3`` - tensor power, a spelling of ``H3 * 3``."""
        if not _is_count(n):
            return NotImplemented
        return BasisGroup((self,) * int(n))


def _is_count(obj) -> bool:
    """``True`` for a plain integer repeat count.  ``bool`` is not one: it is
    an ``int`` subclass, and ``H3 * True`` is a mistake, not a repeat."""
    return isinstance(obj, (int, np.integer)) and not isinstance(obj, bool)


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

        H3**2 * B3      # BasisGroup(H3, H3, B3)
        (H3 * L1)**2    # BasisGroup(H3, L1, H3, L1)

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
                "Bases are now instances - pass H3, not H3().")
        return super().__new__(cls, items)

    def __mul__(self, other) -> "BasisGroup":
        """``H3**2 * B3`` appends directions; against an ``int``,
        ``(H3 * L1) * 2`` repeats the whole pattern."""
        if _is_count(other):
            return BasisGroup(tuple(self) * int(other))
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(self) + tuple(other))

    def __rmul__(self, other) -> "BasisGroup":
        if _is_count(other):
            return BasisGroup(tuple(self) * int(other))
        other = _as_group(other)
        if other is None:
            return NotImplemented
        return BasisGroup(tuple(other) + tuple(self))

    def __pow__(self, n: int) -> "BasisGroup":
        """``(H3 * L1) ** 2`` - the whole pattern repeated, a
        spelling of ``(H3 * L1) * 2``."""
        if not _is_count(n):
            return NotImplemented
        return BasisGroup(tuple(self) * int(n))

    #inherited tuple.__add__ would silently return a plain tuple and drop the
    #group type, so the removed operator has to be blocked rather than deleted
    def __add__(self, other):
        raise TypeError(
            "'+' no longer combines bases; '*' joins directions. "
            "Write H3**2 * B3, not H3 * 2 + B3.")

    __radd__ = __add__

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
            parts.append(f"{prev.name}**{run}" if run > 1 else prev.name)
            run = 1
        return " * ".join(parts)


#: Retained so ``type[AbstractBasis]`` annotations and imports keep resolving.
AbstractBasis = Basis


@jax.jit
def N2_weights(w0, w1, bp_inds):
    """Tensor-product weights for a 2-D element.

    :param w0:
        1-D basis values in the first direction, shape
        ``(n_pts, n_basis_0)``.
    :param w1:
        The same for the second direction.
    :param bp_inds:
        ``(B, 2)`` pairs picking which 1-D weight to multiply for each of the
        element's ``B`` tensor-product weights.

    :returns:
        Weights of shape ``(B, n_pts)``.
    """
    bp_inds = jnp.asarray(bp_inds, dtype=jnp.int32)  # (B, 2)
    def one_pair(ind):
        i, j = ind[0], ind[1]
        return w0[:, i] * w1[:, j]   # (n_pts,)

    return jax.vmap(one_pair, in_axes=0)(bp_inds)

@jax.jit
def N3_weights(w0, w1, w2, bp_inds):
    """Tensor-product weights for a 3-D element.

    :param w0:
        1-D basis values in the first direction, shape
        ``(n_pts, n_basis_0)``.
    :param w1:
        The same for the second direction.
    :param w2:
        The same for the third.
    :param bp_inds:
        ``(B, 3)`` triples picking which 1-D weights to multiply for each of
        the element's ``B`` tensor-product weights.

    :returns:
        Weights of shape ``(B, n_pts)``.
    """
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

def _B3(x) -> jnp.ndarray:
    """
    Cubic bezier basis function.

    :param x: points to interpolate
    :return: basis weights
    """

    return jnp.column_stack((
        (1 - x)**3/6,
        (3*x**3 - 6*x**2 + 4)/6,
        (-3*(x**3) + 3*x**2 + 3 * x + 1)/6,
        (x**3)/6,
    ))
def _B3d1(x) -> jnp.ndarray:
    """First derivative of the cubic B-spline basis.

    :param x: points to interpolate
    :return: basis weight derivatives
    """
    return jnp.column_stack((
        -(1 - x)**2 / 2,
        (3*x**2 - 4*x) / 2,
        (-3*x**2 + 2*x + 1) / 2,
        (x**2) / 2,
    ))
def _B3d1d1(x) -> jnp.ndarray:
    """Second derivative of the cubic B-spline basis.

    :param x: points to interpolate
    :return: basis weight second derivatives
    """
    return jnp.column_stack((
        1 - x,
        3 * x - 2,
        -3 * x + 1,
        x,
    ))


def _L1(x) -> jnp.ndarray:
    """
    Linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    return jnp.array([1. - x, x]).T

def _L1d1(x) -> jnp.ndarray:
    """
    First derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    W = jnp.ones((x.shape[0], 2))
    W = W.at[:,0].add(-2)
    return jnp.array(W)

def _L1d1d1(x) -> jnp.ndarray:
    """
    Second derivative for the linear lagrange basis function.
    
    :param x: points to interpolate
    :return: basis weights
    """
    return jnp.zeros((x.shape[0], 2))

def _H3(x:jnp.ndarray) -> jnp.ndarray:
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

def _H3d1(x: jnp.ndarray) -> jnp.ndarray:
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

def _H3d1d1(x) -> jnp.ndarray:
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

def _L2(x):
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

def _L2d1(x):
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

def _L3(x):
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

def _L3d1(x):
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

def _L4(x):
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

def _L4d1(x):
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

H3 = Basis(
    name='H3',
    fn=_H3,
    weights=('x0', 'dx0', 'x1', 'dx1'), #then this records the derivatives
    deriv=(_H3, _H3d1, _H3d1d1),
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

L1 = Basis(
    name='L1',
    fn=_L1,
    weights=('x0', 'x1'),
    deriv=(_L1, _L1d1, _L1d1d1),
    order=1,
    node_locs=(0, 1),
)
"""Linear Lagrange basis (C⁰ continuity, 2 nodes per direction).

Each node contributes only a *position* weight.  No derivative fields are
required on the associated :class:`~HOMER.mesh.node.MeshNode` objects.

Useful for coarse linear meshes that are subsequently
:meth:`~HOMER.mesh.refinement.rebase`-d to a higher-order basis.
"""

L2 = Basis(
    name='L2',
    fn=_L2,
    weights=('x0', 'x1', 'x2'),
    deriv=(_L2, _L2d1),
    order=2,
    node_locs=(0, 1/2, 2/2),
)
"""Quadratic Lagrange basis (C⁰ continuity, 3 nodes per direction).

Provides second-order accuracy with 3 nodes per direction and no derivative
fields on nodes.
"""

L3 = Basis(
    name='L3',
    fn=_L3,
    weights=('x0', 'x1', 'x2', 'x3'),
    deriv=(_L3, _L3d1),
    order=3,
    node_locs=(0/3, 1/3, 2/3, 3/3),
)
"""Cubic Lagrange basis (C⁰ continuity, 4 nodes per direction).

Third-order accuracy with uniformly-spaced node positions at 0, 1/3, 2/3, 1.
No derivative fields required on nodes.
"""

L4 = Basis(
    name='L4',
    fn=_L4,
    weights=('x0', 'x1', 'x2', 'x3', 'x4'),
    deriv=(_L4, _L4d1),
    order=4,
    node_locs=(0/4, 1/4, 2/4, 3/4, 4/4),
)
"""Quartic Lagrange basis (C⁰ continuity, 5 nodes per direction).

Fourth-order accuracy with uniformly-spaced node positions at
0, 1/4, 2/4, 3/4, 1.  No derivative fields required on nodes.
"""

B3 = Basis(
    name='B3',
    fn=_B3,
    weights=('x0', 'x1', 'x2', 'x3'),
    deriv=(_B3, _B3d1, _B3d1d1),
    order=3,
    node_locs=(-1, 0, 1, 2), #hat t do this # yeah buddy get down with this.
    interpolatory=False, #shared control points, not interpolated nodal values
)
"""Cubic B-spline basis (C² continuity, 4 control points per element per
direction, each shared across neighbouring elements).
"""

#: The pre-1.0 names.  ``H3Basis is H3``, so they compare, hash, serialise and
#: combine identically - only the spelling is deprecated.
H3Basis, L1Basis, L2Basis, L3Basis, L4Basis, B3Basis = H3, L1, L2, L3, L4, B3

LAGRANGE_BASES = {b.order: b for b in (L1, L2, L3, L4)}


def Lagrange(order: int) -> Basis:
    """The Lagrange basis of the requested order.

    ``Lagrange(3) is L3``.  Useful where the order is a variable::

        mesh.rebase(Lagrange(order) * 3)

    :param order:
        Polynomial order, 1 to 4.

    :returns:
        The corresponding registered basis.

    :raises ValueError:
        If HOMER defines no Lagrange basis of that order.
    """
    try:
        return LAGRANGE_BASES[order]
    except KeyError:
        raise ValueError(f"No Lagrange basis of order {order}; HOMER defines "
                         f"orders {sorted(LAGRANGE_BASES)}") from None
