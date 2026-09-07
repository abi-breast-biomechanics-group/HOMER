"""
mesh.py - :class:`Mesh`, a geometry field that owns named secondary fields.

A :class:`~HOMER.mesh.field.MeshField` is one vector-valued field over a
topology.  A :class:`Mesh` is the world-space geometry field plus a dictionary
of further fields sharing that topology - fibre directions, velocities,
stresses - reached as ``mesh['name']``.
"""

from os import PathLike
from copy import deepcopy
from typing import Optional

import numpy as np

from HOMER.mesh import plotting
from HOMER.basis_definitions import BasisGroup
from HOMER.mesh.field import MeshField
from HOMER.mesh.node import MeshNode
from HOMER.mesh.element import MeshElement


class Mesh(MeshField):
    """A coordinate mesh that can also carry named secondary fields.

    :class:`Mesh` extends :class:`MeshField` by adding a dictionary
    ``fields`` that stores secondary :class:`MeshField` objects.  The
    primary geometry (XYZ world-space coordinates) is stored in the parent
    :class:`MeshField`, while any secondary quantities such as fibre
    directions, velocities, or material properties are stored as named
    entries in ``fields``.

    Secondary fields are created with :meth:`new_field` and accessed via
    dictionary-style indexing::

        mesh = Mesh(nodes=[...], elements=[...])
        mesh.new_field('fibre', field_dimension=3,
                       field_locs=data_pts, field_values=fibre_vectors,
                       new_basis=[H3Basis]*3)
        fibre_field = mesh['fibre']   # MeshField

    Parameters
    ----------
    nodes:
        Nodes defining the primary geometry.
    elements:
        Elements of the mesh.
    jax_compile:
        Pre-compile JAX functions at construction time.

    Attributes
    ----------
    fields : dict[str, MeshField]
        Named secondary fields.
    """

    #--- methods defined in sibling modules
    plot = plotting.plot_mesh

    def __init__(self, nodes:Optional[list[MeshNode]] = None, elements: Optional[list[MeshElement]|MeshElement]=None, jax_compile:bool = False) -> None:
        """Initialise a :class:`Mesh`.

        Parameters
        ----------
        nodes:
            Node list (may be ``None`` for incremental construction).
        elements:
            Element or element list.
        jax_compile:
            If ``True``, JIT-compile internal functions at construction.
        """
        super().__init__(nodes, elements, jax_compile)
        self.fields = {}

    def __getitem__(self, input: str) -> MeshField:
        return self.fields[input]

    def __setitem__(self, key: str, value: MeshField):
        assert len(self.elements) == len(value.elements), 'Fields must have the same number of elements'
        assert self.elements[0].ndim == value.elements[0].ndim, 'Feilds must share the same dimensionality of basis components'
        self.fields[key] = value

    def refine(self, refinement_factor: Optional[int] = None, by_xi_refinement: Optional[tuple[np.ndarray]] = None, clean_nodes=True, plot=False, preserve_fixed_params=True, reorder_nodes=True):
        """Refine the primary geometry *and* all secondary fields simultaneously.

        Calls :meth:`MeshField.refine` on the coordinate mesh and on every
        field in :attr:`fields`.

        Parameters
        ----------
        refinement_factor:
            Uniform refinement multiplier (≥ 2).
        by_xi_refinement:
            Per-direction xi breakpoint arrays.
        clean_nodes:
            Remove unreferenced nodes after refinement.
        preserve_fixed_params:
            Carry each node's :attr:`MeshNode.fixed_params` across to the
            coincident nodes of the refined geometry, and of every field.
        reorder_nodes:
            Renumber the refined nodes into a predictable order - see
            :func:`~HOMER.mesh.reordering.reorder_nodes`.  Each field is
            ordered from its own topology, so a field and the geometry stay
            consistent without sharing a node numbering.
        """
        super().refine(refinement_factor, by_xi_refinement, clean_nodes, plot, preserve_fixed_params,
                       reorder_nodes)
        for field in self.fields.values():
            field.refine(refinement_factor, by_xi_refinement, clean_nodes, plot, preserve_fixed_params,
                         reorder_nodes)

    def rebase(self, new_basis: BasisGroup, in_place=False, res=10, preserve_fixed_params=True,
               reorder_nodes=True) -> 'Mesh':
        """ Rebases self, capturing the rebase of the underlying mesh field.

        *reorder_nodes* is passed straight through to
        :meth:`MeshField.rebase`; the secondary fields are not rebased here,
        so their numbering is untouched.
        """
        temp_meshField = super().rebase(new_basis, in_place=False, res=res,
                                        preserve_fixed_params=preserve_fixed_params,
                                        reorder_nodes=reorder_nodes)
        mesh_field_backup = dict(self.fields) #a copy: the two meshes must not share a dict
        new_mesh = Mesh(elements=temp_meshField.elements, nodes=temp_meshField.nodes)
        new_mesh.fields = mesh_field_backup
        if in_place:
            self.nodes = new_mesh.nodes
            self.elements = new_mesh.elements
            self.fields = mesh_field_backup
            self.generate_mesh()
            return self
        return new_mesh

    def new_field(self, field_name: str, field_dimension: int, new_basis: Optional[BasisGroup]=None, field_locs: Optional[np.ndarray]=None, field_values: Optional[np.ndarray]=None, field_params=None, res=10) -> None:
        """Create a secondary field and optionally fit it to sample data.

        A secondary field is a :class:`MeshField` with its own basis
        functions and node topology that is *co-located* with the primary
        coordinate mesh.  It can represent any spatially varying quantity
        – fibre directions, velocity vectors, pressures, stresses, etc.

        The three-step construction algorithm is:

        1. Determine the new field node locations by evaluating the primary
           mesh at the node positions of *new_basis*.
        2. If *field_locs* and *field_values* are provided, embed the sample
           points into the mesh with :meth:`embed_points`.
        3. Build the linear weight matrix and solve for nodal parameters with
           :meth:`linear_fit`.

        After this call, the field is accessible as ``mesh[field_name]``.

        If possible, this will preserve the node and element level topology.
        This allows, as an example, repeating or subsampling the field parameters.

        Parameters
        ----------
        field_name:
            Key used to store and retrieve the new field, e.g.
            ``'fibre_direction'``.
        field_dimension:
            Dimensionality of the field values:

            * ``1`` – scalar field (e.g. pressure, temperature, Z-coordinate)
            * ``3`` – 3-D vector field (e.g. fibre direction, velocity)
        new_basis:
            The 1-D bases for the new field, one per
            parametric direction.  May differ from the primary mesh basis.
            For example, use ``[H3Basis]*3`` for a smooth vector field or
            ``[L1Basis]*3`` for a piecewise-linear scalar field.
        field_locs:
            Physical-space sample locations where field values are known,
            shape ``(n_samples, fdim)``.  When ``None``, an empty field is
            created without fitting.
        field_values:
            Target field values at *field_locs*, shape
            ``(n_samples,)`` for scalars or ``(n_samples, field_dimension)``
            for vectors.  Required if *field_locs* is provided.
        res:
            Unused (reserved for future use).

        Examples
        --------
        Fit a unit-normal vector field and a scalar height field::

            mesh.new_field(
                'normals',
                field_dimension=3,
                field_locs=sample_pts,       # shape (N, 3)
                field_values=normal_vectors, # shape (N, 3)
                new_basis=H3Basis * 3,
            )
            mesh.new_field(
                'height',
                field_dimension=1,
                field_locs=sample_pts,       # shape (N, 3)
                field_values=sample_pts[:, 2],  # scalar Z values
                new_basis=L1Basis * 3,
            )

            # Retrieve and evaluate
            normal_field = mesh['normals']
            values_at_xis = normal_field.evaluate_embeddings(elem_ids, xis)
        """

        if new_basis is None:
            new_basis = self.elements[0].basis_functions


        #only the topology and basis of the rebase are kept -- every node is
        #replaced below -- and the result must be a plain MeshField: a Mesh
        #carries a `fields` dict of its own, and storing one inside this mesh's
        #`fields` would make the mesh a member of itself.
        rebased = self.rebase(new_basis, preserve_fixed_params=False)
        new_field = MeshField(nodes=rebased.nodes, elements=rebased.elements)
        new_field.fdim = field_dimension
        used_fields = new_field.elements[0].used_node_fields

        for idn, node in enumerate(new_field.nodes):
            new_field.nodes[idn] = MeshNode(loc=[0] * field_dimension, **{uf:np.zeros(field_dimension) for uf in used_fields})


        n_vals_per_node = (len(new_field.elements[0].used_node_fields) + 1) * field_dimension #plus 1 is for the spatial field
        total_vals = n_vals_per_node * len(new_field.nodes)

        new_field.true_param_array = np.zeros(total_vals) #instantiate a null parameter array
        new_field.optimisable_param_bool = np.ones(total_vals, dtype=bool)
        new_field.optimisable_param_array = np.zeros(total_vals)

        new_field.generate_mesh()

        if field_params is not None:
            #written through the nodes: generate_mesh rebuilds the parameter
            #arrays from them, so assigning the arrays directly would be undone
            field_params = np.asarray(field_params).ravel()
            if field_params.shape[0] != total_vals:
                raise ValueError(
                    f"field_params had {field_params.shape[0]} entries, but a "
                    f"{field_dimension}-dimensional field on this mesh needs {total_vals}"
                )
            new_field.update_from_params(field_params)

        self[field_name] = new_field

        #certain things should be inhereted, as they will be calculated "wrong"
        self[field_name].faces = self.faces
        self[field_name].bmap = self.bmap
        self[field_name]._topo_lookup = self._topo_lookup
        self[field_name].topomap = self.topomap

        if field_locs is None or field_values is None:
            return

        locs = self.embed_points(field_locs)
        w_mat = self[field_name].get_xi_weight_mat(*locs)
        self[field_name].linear_fit(weight_mat=w_mat, targets=field_values)
        return

    
    def __deepcopy__(self, memo):
        """
        Deep copy a Mesh (including any secondary fields).
        """
        from HOMER.io import dump_mesh_to_dict, parse_mesh_from_dict
        dict_rep = deepcopy(dump_mesh_to_dict(self))
        return parse_mesh_from_dict(dict_rep)

    @classmethod
    def from_dict(cls, dict_rep: dict) -> "Mesh":
        """
        Build a Mesh from a dictionary representation.
        """
        from HOMER.io import parse_mesh_from_dict
        return parse_mesh_from_dict(dict_rep)

    @classmethod
    def load(cls, loc: PathLike) -> "Mesh":
        """
        Load a Mesh (including fields) from a JSON file.
        """
        from HOMER.io import load_mesh
        return load_mesh(loc)

    def to_dict(self) -> dict:
        """
        Returns a dict structure representing the mesh object, including fields.
        """
        from HOMER.io import dump_mesh_to_dict
        return dump_mesh_to_dict(self)

    def save(self, loc: PathLike):
        """
        Saves the mesh (including fields) to a .json formated file in the given location
        """
        from HOMER.io import save_mesh #avoid the circular import here
        save_mesh(self, loc)

    def dump_to_dict(self):
        """
        Returns a dict structure representing the mesh object, for ease of saving
        """
        return self.to_dict()
