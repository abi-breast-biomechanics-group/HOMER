"""
basis_definitions.py – 1-D basis function definitions for HOMER mesh elements.

This module provides the building blocks for constructing high-order mesh
elements.  Every mesh element in HOMER is defined by a *tensor product* of
1-D basis functions – one per parametric direction.

Available 1-D basis classes (all subclass :class:`AbstractBasis`):

* :class:`H3Basis` – cubic Hermite (C¹ continuity, 2 nodes per direction)
* :class:`L1Basis` – linear Lagrange (2 nodes per direction)
* :class:`L2Basis` – quadratic Lagrange (3 nodes per direction)
* :class:`L3Basis` – cubic Lagrange (4 nodes per direction)
* :class:`L4Basis` – quartic Lagrange (5 nodes per direction)

Each basis class is a frozen :class:`dataclasses.dataclass` with the following
class attributes:

* ``fn`` – the basis evaluation function ``fn(x) -> (n_pts, n_basis)``
* ``deriv`` – list of derivative functions ``[fn, d1_fn, d2_fn, …]``
* ``weights`` – ordered weight names, e.g. ``['x0', 'dx0', 'x1', 'dx1']``
* ``order`` – polynomial order
* ``node_locs`` – canonical node positions in [0, 1]
* ``node_fields`` – a :class:`DerivativeField` instance describing which
  derivative quantities each node must carry (``None`` for Lagrange bases)

Typical usage::

    from HOMER.basis_definitions import H3Basis, L1Basis

    # 2-D cubic-Hermite surface element
    from HOMER.mesher import MeshElement
    elem = MeshElement(node_indexes=[0,1,2,3], basis_functions=(H3Basis, H3Basis))

    # 3-D trilinear volume element
    elem3d = MeshElement(node_indexes=[0,1,2,3,4,5,6,7],
                         basis_functions=(L1Basis, L1Basis, L1Basis))
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

@dataclass
class AbstractBasis:
    """Abstract base class for all 1-D basis function definitions.

    Concrete subclasses (e.g. :class:`H3Basis`, :class:`L2Basis`) are frozen
    dataclasses that define the class attributes below.  They are passed as
    classes (not instances) to :class:`~HOMER.mesher.MeshElement`.

    Attributes
    ----------
    fn : Callable
        Basis evaluation function ``fn(x) -> ndarray (n_pts, n_basis)``.
    node_fields : AbstractField or None
        Describes the derivative quantities each node must carry.
        ``None`` for pure Lagrange bases.
    weights : list[str]
        Ordered weight names, e.g. ``['x0', 'dx0', 'x1', 'dx1']``.
        Names starting with ``'dx'`` indicate derivative entries.
    deriv : list[Callable]
        Derivative evaluation functions: ``[fn, d1_fn, d2_fn, …]``.
    order : int
        Polynomial order of the basis.
    node_locs : list[float]
        Canonical node positions in [0, 1].
    """

    fn:Optional[Callable] = None
    node_fields: Optional[type[AbstractField]] = None
    weights: Optional[list[str]] = None
    deriv:Optional[list[Callable]] = None
    order:Optional[int] = None
    node_locs: Optional[list[float]] = None

BasisGroup = tuple[type[AbstractBasis], type[AbstractBasis]] | tuple[type[AbstractBasis], type[AbstractBasis], type[AbstractBasis]] | list[type[AbstractBasis], type[AbstractBasis]] | tuple[type[AbstractBasis], type[AbstractBasis], type[AbstractBasis]]


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

    return jnp.concatenate((
        (1 - x)**3,
        3*x*((1 - x)**2),
        3*(x**2) * (1 - x),
        x**3,
    ))

def B3d1(x) -> jnp.ndarray:
    """
    First derivative with respect to x of the Cubic Bezier basis function.
    :param x: points to interpolaet

    :param x: points to interpolate
    :return: basis weights
    """

    return jnp.concatenate((
        (1 - x)**3,
        3*x*((1 - x)**2),
        3*(x**2) * (1 - x),
        3*x**2,
    ))

def B3d1d1(x) -> jnp.ndarray:
    """
    Second derivative with respect to x of the Cubic Bezier basis function.
    :param x: points to interpolaet

    :param x: points to interpolate
    :return: basis weights
    """

    return jnp.concatenate((
        (1 - x)**3,
        3*x*((1 - x)**2),
        3*(x**2) * (1 - x),
        6*x,
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

@dataclass
class H3Basis(AbstractBasis):
    """Cubic Hermite basis (C¹ continuity, 2 nodes, 4 weights per direction).

    Each node contributes a *position* and a *tangent derivative*:
    ``['x0', 'dx0', 'x1', 'dx1']``.  Requires each :class:`~HOMER.mesher.MeshNode`
    to carry Hermite derivative fields (``du``, ``dv``, … depending on the
    element dimensionality).

    Best choice for smooth geometry where derivative continuity across element
    boundaries is important.
    """

    fn = H3
    node_fields = DerivativeField()
    weights = ['x0', 'dx0', 'x1', 'dx1'] #then this records the derivatives
    deriv = [H3, H3d1, H3d1d1]
    order = 3
    node_locs = [0, 1]
    
@dataclass
class L1Basis(AbstractBasis):
    """Linear Lagrange basis (C⁰ continuity, 2 nodes per direction).

    Each node contributes only a *position* weight.  No derivative fields are
    required on the associated :class:`~HOMER.mesher.MeshNode` objects.

    Useful for coarse linear meshes that are subsequently :meth:`~HOMER.mesher.MeshField.rebase`-d
    to a higher-order basis.
    """

    fn = L1
    weights = ['x0', 'x1']
    deriv = [L1, L1d1, L1d1d1]
    order = 1
    node_locs = [0, 1]

@dataclass
class L2Basis(AbstractBasis):
    """Quadratic Lagrange basis (C⁰ continuity, 3 nodes per direction).

    Provides second-order accuracy with 3 nodes per direction and no
    derivative fields on nodes.
    """

    fn = L2
    weights = ['x0', 'x1', 'x2']
    deriv =[L2, L2d1]
    order = 2
    node_locs = [0, 1/2, 2/2]

@dataclass
class L3Basis(AbstractBasis):
    """Cubic Lagrange basis (C⁰ continuity, 4 nodes per direction).

    Third-order accuracy with uniformly-spaced node positions at
    0, 1/3, 2/3, 1.  No derivative fields required on nodes.
    """

    fn = L3
    weights = ['x0', 'x1', 'x2', 'x3']
    deriv = [L3, L3d1]
    order = 3
    node_locs = [0/3, 1/3, 2/3, 3/3]
    
@dataclass
class L4Basis(AbstractBasis):
    """Quartic Lagrange basis (C⁰ continuity, 5 nodes per direction).

    Fourth-order accuracy with uniformly-spaced node positions at
    0, 1/4, 2/4, 3/4, 1.  No derivative fields required on nodes.
    """

    fn = L4
    weights = ['x0', 'x1', 'x2', 'x3', 'x4']
    deriv = [L4, L4d1]
    order = 4
    node_locs = [0/4, 1/4, 2/4, 3/4, 4/4]
