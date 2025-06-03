from typing import Callable, Optional
import jax.numpy as jnp
import jax
import numpy as np
from itertools import combinations_with_replacement, product

from dataclasses import dataclass, field


deriv_fields = (
    ('du'),
    ('du', 'dv', 'dudv'),
    ('du', 'dv', 'dw', 'dudv', 'dudw', 'dwdv', 'dudvdqw'),
)

@dataclass
class AbstractField:
    n_field: int
    _field_scaling: tuple[tuple[str]]

    def __add__(self, other: type["AbstractField"]) -> "AbstractField":
        if not isinstance(other, self.__class__):
            raise ValueError("Can only add the same fields")
        return self.__class__.__new__(self.__class__, n_field=self.n_field + other.n_field, _field_scaling=self._field_scaling)
    
    def get_needed_fields(self):
        return self._field_scaling[self.n_field]

@dataclass
class DerivativeField(AbstractField): 
    n_field:int = field(default=1)
    _field_scaling:tuple[tuple[str]] = field(default=deriv_fields)

@dataclass
class AbstractBasis:
    fn:Optional[Callable] = None
    node_fields: Optional[type[AbstractField]] = None
    num_weights: Optional[int] = None
    d1:Optional[Callable] = None
    d2:Optional[Callable] = None
    d3:Optional[Callable] = None

BasisGroup = tuple[type[AbstractBasis], type[AbstractBasis]] | tuple[type[AbstractBasis], type[AbstractBasis], type[AbstractBasis]]

def BasisOuterProductInds(basis_nums:list[int]):
    #the order of the combination here is dependant on the results
    return np.arange(32) 


#we can refactor these function
def N2_weights(w0, w1, bp_inds) -> jnp.ndarray:
    BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
             [2, 0], [3, 0], [2, 1], [3, 1],
             [0, 2], [1, 2], [0, 3], [1, 3],
             [2, 2], [3, 2], [2, 3], [3, 3]]
    # BPInd = product([0,1,2,3], repeat=2)
    w_list = []
    for ii in BPInd:
        w_list.append(w0[:, ii[0]] * w1[:, ii[1]])
    weights = jnp.vstack(w_list)
    return weights

def N3_weights(w0, w1, w2, bp_inds) -> jnp.ndarray:
    # BPInd = product([0,1,2,3], repeat=2)
    BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
             [2, 0], [3, 0], [2, 1], [3, 1],
             [0, 2], [1, 2], [0, 3], [1, 3],
             [2, 2], [3, 2], [2, 3], [3, 3]]
    w_list = []
    for ii in BPInd:
        w_list.append(w0[:, ii[0]] * w1[:, ii[1]] * w2[:, 0])
    for ii in BPInd:
        w_list.append(w0[:, ii[0]] * w1[:, ii[1]] * w2[:, 1])
    weights = jnp.vstack(w_list)
    return weights

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
    return jnp.array([W])

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

@dataclass
class H3Basis(AbstractBasis):
    fn = H3
    node_fields = DerivativeField()
    num_weights = 4
    d1 = H3d1
    d2 = H3d1d1
    
@dataclass
class L1Basis(AbstractBasis):
    fn = L1
    num_weights = 2
    d1 = L1d1
    d2 = L1d1d1
