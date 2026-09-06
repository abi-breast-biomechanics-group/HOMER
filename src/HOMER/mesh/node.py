"""
node.py - :class:`MeshNode`, a single point of a HOMER mesh.

A node stores a physical location plus whatever Hermite derivative vectors the
element bases require, and tracks which of those parameters are pinned by the
user (see :meth:`MeshNode.fix_parameter`).
"""

from typing import Optional

import numpy as np
import jax.numpy as jnp
import pyvista as pv


class MeshNode(dict):
    """A mesh node that stores a physical location and associated derivative data.

    :class:`MeshNode` subclasses :class:`dict` so that derivative quantities
    (``du``, ``dv``, ``dw``, ``dudv``, …) required by higher-order basis
    functions can be stored as named entries.  All values must be
    :class:`numpy.ndarray` objects of the same length as ``loc``.

    For a 2-D manifold mesh with cubic-Hermite basis in both directions
    (``H3Basis``, ``H3Basis``), each node must carry ``du``, ``dv``, and
    ``dudv`` derivatives::

        node = MeshNode(
            loc=np.array([0.0, 0.0, 1.0]),
            du=np.zeros(3),
            dv=np.zeros(3),
            dudv=np.zeros(3),
        )

    For a 3-D volume mesh with ``H3Basis`` in all three directions, the
    additional derivatives ``dw``, ``dudw``, ``dvdw``, and ``dudvdw`` are
    also required::

        node = MeshNode(
            loc=np.array([0.0, 0.0, 1.0]),
            du=np.zeros(3), dv=np.zeros(3), dw=np.zeros(3),
            dudv=np.zeros(3), dudw=np.zeros(3), dvdw=np.zeros(3),
            dudvdw=np.zeros(3),
        )

    Parameters
    ----------
    loc:
        Physical-space coordinates of the node, shape ``(fdim,)``.
    id:
        Optional unique identifier.  When provided, nodes can be referenced
        by ID rather than list index in a :class:`MeshElement`.
    **kwargs:
        Named derivative arrays, e.g. ``du``, ``dv``, ``dw``, ``dudv``, …
        All values must be ``numpy.ndarray`` (or list / JAX array, which
        are automatically converted).

    Attributes
    ----------
    loc : numpy.ndarray
        Physical-space coordinates, shape ``(fdim,)``.
    id :
        The node identifier (or ``None``).
    fixed_params : dict
        Maps parameter name → array of fixed component indices.  Populated
        by :meth:`fix_parameter`.
    """

    def __init__(self, loc, id=None, **kwargs):
        """Initialise a :class:`MeshNode`.

        Parameters
        ----------
        loc:
            Physical-space coordinates, shape ``(fdim,)``.
        id:
            Optional unique identifier.
        **kwargs:
            Named derivative arrays (``du``, ``dv``, ``dw``, …).
            Each value must be an array of the same length as ``loc``.

        Raises
        ------
        ValueError
            If any keyword-argument value is not an array-like type.
        """
        self.loc = np.asarray(loc)
        self.id = id
        self.update(kwargs)
        self.fixed_params = {}

        for key, value in kwargs.items():
            if isinstance(value, list):
                self[key] = np.asarray(value).copy()
            elif isinstance(value, jnp.ndarray):
                self[key] = np.asarray(value).copy()
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"Only np.ndarray are valid additional data, but found key: {key}, value: {value} pair")
            else:
                self[key] = np.array(value).copy()

    def fix_parameter(self, param_names: list | str, values: Optional[list[np.ndarray]|np.ndarray]=None, inds: Optional[list[int]] = None) -> None:
        """Mark one or more node parameters as fixed (non-optimisable).

        Fixed parameters are excluded from the optimisable parameter vector
        exposed by :class:`MeshField`.  Optionally, the parameter can also be
        set to a specified value at the same time.

        Parameters
        ----------
        param_names:
            Name or list of names of the parameters to fix, e.g.
            ``'loc'``, ``'du'``, ``['loc', 'dv']``.
        values:
            Optional value(s) to assign at the time of fixing.  Must match
            the shape implied by ``inds`` (or the full parameter dimension
            when ``inds`` is ``None``).
        inds:
            Component indices to fix within the parameter array (e.g.
            ``[0, 2]`` to fix the *x* and *z* components of ``loc``).
            When ``None``, all components are fixed.
        """
        l_dim = self.loc.shape[0]

        if inds is not None:
            inds = np.array(inds).astype(int)
        if isinstance(param_names, str):
            param_names = [param_names]
        if not isinstance(values, list):
            values = [values] * len(param_names)

        for idp, param in enumerate(param_names):
            if inds is None:
                inds = np.arange(l_dim).astype(int)
            if param in self.fixed_params:
                self.fixed_params[param] = np.union1d(self.fixed_params[param], inds)
            else:
                self.fixed_params[param] = inds

            if values[idp] is not None:
                if param == 'loc':
                    self.loc[inds] = values[idp]
                else:
                    self[param][inds] = values[idp]

    def get_optimisability_arr(self):
        """
        Returns the optimisable status of all data stored on the node.
        """
        l_dim = self.loc.shape[0]
        free_loc = np.ones(l_dim) 
        free_loc[self.fixed_params.get('loc', [])] = 0
        list_data = [free_loc]
        for key in self.keys():
            free_key = np.ones(l_dim)
            free_key[self.fixed_params.get(key, [])] = 0
            list_data.append(free_key)
        return np.concatenate(list_data, axis=0)


    def plot(self, scene: Optional[pv.Plotter] = None) -> pv.Plotter | None:
        """
        Draws the node, and any quantities, to a pyvista plotter.
        :param scene: An existing pyvista scene to draw too - if given will not draw the plot.
        """
        s = pv.Plotter() if scene is None else scene
        
        s.add_mesh(pv.PolyData(self.loc), point_size=5, render_points_as_spheres=True)
        label_locs = []
        label_names = []
        for key, value in self.items():
            s.add_mesh(pv.lines_from_points(np.array((self.loc, self.loc + value))))
            label_locs.append(self.loc + value)
            label_names.append(key)
        s.add_point_labels(label_locs, labels=label_names)
        if scene is None:
            s.show()
            return None
        return s

    def unfix_params(self):
        self.fixed_params = {}
