"""
HOMER.mesh - the mesh data structures and everything they do.

The four classes are the package's subject:

* :class:`~HOMER.mesh.node.MeshNode` - a single node: a physical location plus
  any Hermite derivative vectors the chosen bases require.
* :class:`~HOMER.mesh.element.MeshElement` - connects nodes through a product
  of 1-D bases, one per parametric direction.
* :class:`~HOMER.mesh.field.MeshField` - nodes and elements together, able to
  evaluate and optimise any vector-valued field over that topology.
* :class:`~HOMER.mesh.mesh.Mesh` - the world-space geometry field, owning a
  dictionary of named secondary fields (``mesh['fibres']``).

:class:`MeshField` is deliberately thin: it owns the state and the lifecycle,
while what a field *does* lives in a module per concern - :mod:`evaluation`,
:mod:`parameters`, :mod:`topology`, :mod:`refinement`, :mod:`plotting` - whose
functions take the field as their first argument and are bound into the class
by :mod:`HOMER.mesh.field`.

Typical import::

    from HOMER import Mesh, MeshNode, MeshElement
    from HOMER.basis_definitions import H3Basis, L1Basis
"""

from HOMER.mesh.node import MeshNode
from HOMER.mesh.element import MeshElement
from HOMER.mesh.field import MeshField
from HOMER.mesh.mesh import Mesh
from HOMER.mesh.element_eval import (make_eval, make_deriv_eval, make_weight_eval,
                                     volume_quadrature_order, GAUSS)
from HOMER.mesh.parameters import column_equilibrated_lstsq
from HOMER.mesh.refinement import MAX_XI_DENOMINATOR

__all__ = ['MeshNode', 'MeshElement', 'MeshField', 'Mesh',
           'make_eval', 'make_deriv_eval', 'make_weight_eval',
           'volume_quadrature_order', 'GAUSS', 'column_equilibrated_lstsq',
           'MAX_XI_DENOMINATOR']
