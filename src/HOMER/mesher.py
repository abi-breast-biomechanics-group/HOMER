"""
mesher.py – the mesh classes, now defined in :mod:`HOMER.mesh`.

This module is kept so that ``from HOMER.mesher import Mesh`` and friends keep
working; the definitions moved into a module per concern when the original
3,000-line file was split:

* :mod:`HOMER.mesh.node`, :mod:`HOMER.mesh.element` – the leaf structures
* :mod:`HOMER.mesh.field`, :mod:`HOMER.mesh.mesh` – the field classes
* :mod:`HOMER.mesh.evaluation`, :mod:`HOMER.mesh.parameters`,
  :mod:`HOMER.mesh.topology`, :mod:`HOMER.mesh.refinement`,
  :mod:`HOMER.mesh.plotting` – what a field does

New code should import from :mod:`HOMER` or :mod:`HOMER.mesh` directly.
"""

from HOMER.mesh import (MeshNode, MeshElement, MeshField, Mesh, make_eval,
                        make_deriv_eval, make_weight_eval, volume_quadrature_order,
                        GAUSS, column_equilibrated_lstsq, MAX_XI_DENOMINATOR,
                        reorder_nodes)

__all__ = ['MeshNode', 'MeshElement', 'MeshField', 'Mesh', 'make_eval',
           'make_deriv_eval', 'make_weight_eval', 'volume_quadrature_order',
           'GAUSS', 'column_equilibrated_lstsq', 'MAX_XI_DENOMINATOR',
           'reorder_nodes']
