"""
evaluation.py - evaluating a field, its derivatives, and the quantities built
on them.

Every function here is a method of :class:`~HOMER.mesh.field.MeshField` that
lives in its own module: the first argument is the field itself, and
:mod:`HOMER.mesh.field` binds them into the class.  The two ``@wide_eval``
placeholders are replaced per-instance by the compiled closures that
:meth:`MeshField._generate_eval_function` builds; they stay here so that
``@expand_wide_evals`` still finds them and generates the
``*_in_every_element`` / ``*_ele_xi_pair`` variants.
"""

import typing
from typing import Optional, Callable, TYPE_CHECKING
from itertools import product

import numpy as np
import jax.numpy as jnp
import pyvista as pv

from HOMER.embedding import build_embedding_fn
from HOMER.mesh.element_eval import GAUSS, volume_quadrature_order
from HOMER.mesh_decorators import wide_eval
from HOMER.utils import make_tiling

if TYPE_CHECKING:
    from HOMER.mesh.mesh import Mesh


@wide_eval
def evaluate_embeddings(self, *a, **kw): #placeholder for later func definition
    """Evaluate the field at parametric coordinates within one or more elements.

    This is a placeholder that is replaced by a compiled JAX function when
    :meth:`generate_mesh` is called.  The full signature after
    initialisation is::

        evaluate_embeddings(element_ids, xis, fit_params=None) -> jnp.ndarray

    Parameters
    ----------
    element_ids:
        1-D array of integer element indices, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.
    fit_params:
        Override of the current :attr:`optimisable_param_array`.
        When ``None`` the stored parameter values are used.

    Returns
    -------
    jnp.ndarray
        Field values at the requested locations, shape ``(n_pts, fdim)``.

    Notes
    -----
    The ``@expand_wide_evals`` class decorator automatically creates two
    additional variants:

    * ``evaluate_embeddings_in_every_element(xis)`` – evaluates the same
      grid of xi points in *every* element and stacks the results.
    * ``evaluate_embeddings_ele_xi_pair(element_ids, xis)`` – evaluates
      each ``(element, xi)`` pair independently (equivalent signature to
      the base function but without batching).
    """
    if not typing:
        raise RuntimeError('Called evaluate_embeddings before initialisation')
    return


@wide_eval
def evaluate_deriv_embeddings(self, *a, **kw): #placeholder for later func definition
    """Evaluate a specified partial derivative of the field.

    This is a placeholder replaced at :meth:`generate_mesh` time.  The
    full signature is::

        evaluate_deriv_embeddings(element_ids, xis, derivs, fit_params=None)
            -> jnp.ndarray

    Parameters
    ----------
    element_ids:
        1-D integer array, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.
    derivs:
        Derivative order per parametric direction, e.g. ``[1, 0]`` for
        ∂/∂u in a 2-D element or ``[0, 0, 1]`` for ∂/∂w in a 3-D one.
    fit_params:
        Optional override of :attr:`optimisable_param_array`.

    Returns
    -------
    jnp.ndarray
        Derivative field values, shape ``(n_pts, fdim)``.
    """
    if not typing:
        raise RuntimeError('Called evaluate_deriv_embeddings before initialisation')
    return


def evaluate_element_embeddings(self, element_id, xis, fit_params=None):
    """Evaluate the embedding for a single element identified by its ID.

    Parameters
    ----------
    element_id:
        The user-assigned element ID (not the list index).
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.
    fit_params:
        Optional parameter override.

    Returns
    -------
    jnp.ndarray
        Field values, shape ``(n_pts, fdim)``.
    """
    if fit_params is None:
        fit_params = self.optimisable_param_array
    return self.evaluate_embeddings([self.element_id_to_ind[element_id]], xis, fit_params=fit_params)


@wide_eval
def evaluate_normals(self, element_ids: np.ndarray, xis: np.ndarray, fit_params=None) -> np.ndarray:
    """Return the surface normal vectors at parametric coordinates.

    Only valid for 2-D manifold meshes (``ndim == 2``).  The normal is
    computed as the cross product of the two surface tangent vectors.

    Parameters
    ----------
    element_ids:
        1-D integer array of element indices, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, 2)``.
    fit_params:
        Optional override of :attr:`optimisable_param_array`.

    Returns
    -------
    jnp.ndarray
        Normal vectors (not normalised), shape ``(n_pts, 3)``.

    Raises
    ------
    ValueError
        If called on a 3-D volume mesh.
    """

    if self.ndim == 3: 
        raise ValueError("Normals aren't defined on a volume mesh")
    if fit_params is None:
        fit_params = self.optimisable_param_array

    d0 = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1], fit_params) 
    d1 = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0], fit_params)
    return jnp.cross(d0, d1)


@wide_eval
def eval_numeric_jac(self, element_ids, xis, locals=None, step=2e-1, fit_params=None):
    """ 
    Evaluates the jacobian at a set of xis within an element
    Uses numeric derivatives, useful when the underlying mesh field has zero derivative boundaries.
    """
    if fit_params is None:
        fit_params = self.optimisable_param_array

    if locals is None:
        locals = self.evaluate_embeddings(element_ids, xis)

    flip = jnp.where(jnp.atleast_2d(xis) > 0.5, -1, 1)

    if self.ndim == 2:
        du = (self.evaluate_embeddings(element_ids, xis + jnp.array([step, 0])[None] * flip[:, 0][:, None], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 0][:, None, None]
        dv = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, step])[None] * flip[:, 1][:, None], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 1][:, None, None]
        jmats = jnp.concatenate((du, dv), axis=1) / step
    if self.ndim == 3:
        du = (self.evaluate_embeddings(element_ids, xis + jnp.array([step, 0, 0])[None] * flip[:, 0][:, None], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 0][:, None, None]
        dv = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, step, 0])[None] * flip[:, 1][:, None], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 1][:, None, None]
        dw = (self.evaluate_embeddings(element_ids, xis + jnp.array([0, 0, step])[None] * flip[:, 2][:, None], fit_params=fit_params) - locals).reshape(-1, 1, self.fdim) * flip[:, 2][:, None, None]

        jmats = jnp.concatenate((du, dv, dw), axis=1) / step
    # return jmats
    return jnp.swapaxes(jmats, -1,-2) #differing jacobin implementation.


@wide_eval
def evaluate_jacobians(self, element_ids, xis, fit_params=None):
    """Evaluate the Jacobian matrix of the embedding at parametric coordinates.

    Returns ∂x/∂ξ, the matrix mapping parametric-space tangent vectors to
    physical-space tangent vectors.  Rows correspond to physical directions
    (x, y, z) and columns to parametric directions (u, v[, w]).

    Parameters
    ----------
    element_ids:
        1-D integer array, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.
    fit_params:
        Optional override of :attr:`optimisable_param_array`.

    Returns
    -------
    jnp.ndarray
        Jacobian matrices, shape ``(n_pts, fdim, ndim)``.
    """
    if fit_params is None:
        fit_params = self.optimisable_param_array

    if self.ndim == 2:
        du = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
        dv = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1], fit_params=fit_params).reshape(-1, 1, self.fdim)
        jmats = jnp.concatenate((du, dv), axis=1)
    if self.ndim == 3:

        du = self.evaluate_deriv_embeddings(element_ids, xis, [1, 0, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
        dv = self.evaluate_deriv_embeddings(element_ids, xis, [0, 1, 0], fit_params=fit_params).reshape(-1, 1, self.fdim)
        dw = self.evaluate_deriv_embeddings(element_ids, xis, [0, 0, 1], fit_params=fit_params).reshape(-1, 1, self.fdim)
        jmats = jnp.concatenate((du, dv, dw), axis=1)
    # return jmats
    return jnp.swapaxes(jmats, -1,-2) #differing jacobin implementation.


def xi_grid(self, res: int, dim=None, surface=False, boundary_points=True, lattice=None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return a regular grid of parametric (xi) coordinates.

    Creates a uniform Cartesian grid of xi points for use with
    :meth:`evaluate_embeddings_in_every_element` or for passing to
    :meth:`get_xi_weight_mat`.

    Parameters
    ----------
    res:
        Number of grid points along each parametric direction.  The
        total number of points is ``res ** ndim`` (or ``res ** 2`` when
        returning surface faces of a volume mesh).
    dim:
        Dimensionality of the grid (2 or 3).  Defaults to
        :attr:`ndim`.
    surface:
        For a 3-D mesh, return only points on the six element faces
        rather than the full interior grid.
    boundary_points:
        When ``False``, exclude xi = 0 and xi = 1 from the grid (useful
        to avoid double-counting shared element boundaries).
    lattice:
        Optional ``(xn, yn)`` tiling definition for hexagonal surface
        patterns.

    Returns
    -------
    numpy.ndarray
        Grid points, shape ``(res**ndim, ndim)`` (or, when *lattice* is
        provided and *surface* is ``True``, a ``(pts, connectivity)``
        tuple).
    """
    dim = self.ndim if dim is None else dim

    b_off = 0 if boundary_points else 1
    if not boundary_points:
        res = res + 1 #boundary points drops a res
    if dim == 2:
        if lattice is None:
            X,Y = (np.mgrid[
                0:res - b_off,
                0:res - b_off,
            ] + b_off * 0.5)/(res - 1)
            return np.column_stack((X.flatten(), Y.flatten()))
        else:
            return make_tiling(*lattice)
    else:
        if not surface:
            X,Y,Z = (np.mgrid[
                0:res - b_off,
                0:res - b_off,
                0:res - b_off,
            ] + b_off * 0.5 )/(res - 1)
            return np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        else:
            if lattice is None:
                raw_x = np.array([x.flatten() for x in np.mgrid[:res, :res]/(res-1)])
            else:
                raw_x, connectivity = make_tiling(*lattice) 
                raw_x = raw_x.T

            zero_r = np.zeros(shape=raw_x[0].shape[0])
            ones_r = np.ones(shape=raw_x[0].shape[0])

            arrays = [
                    np.column_stack((zero_r, raw_x[0], raw_x[1])),
                    np.column_stack((ones_r, raw_x[0], raw_x[1])),
                    np.column_stack((raw_x[0], zero_r, raw_x[1])),
                    np.column_stack((raw_x[0], ones_r, raw_x[1])),
                    np.column_stack((raw_x[0], raw_x[1], zero_r)),
                    np.column_stack((raw_x[0], raw_x[1], ones_r)),
            ]
            if lattice is None:
                return np.concatenate(arrays) 
            return np.concatenate(arrays), connectivity


def gauss_grid(self, ng):
    """Return a tensor-product Gauss quadrature grid.

    Parameters
    ----------
    ng:
        * **int** – return 1-D Gauss points for a single direction.
        * **list[int]** – tensor-product grid; e.g. ``[3, 3]`` for a
          2-D surface integration or ``[3, 3, 3]`` for a 3-D volume.

    Returns
    -------
    Xi : numpy.ndarray
        Gauss point locations, shape ``(n_gauss, ndim)`` or ``(n_gauss,)``
        for the 1-D case.
    W : numpy.ndarray
        Corresponding quadrature weights, shape ``(n_gauss,)``.

    Raises
    ------
    ValueError
        If ``ng`` has more than 3 entries or is of an unsupported type.
    """

    if isinstance(ng, int):
        if ng not in GAUSS:
            raise ValueError(
                f"No {ng}-point Gauss rule is tabulated; available orders are "
                f"{sorted(GAUSS)}"
            )
        return GAUSS[ng]
    elif isinstance(ng, list):
        if len(ng) > 3:
            raise ValueError('Gauss points for 4 dimensions and above not supported')
        if len(ng) == 2:
            Xi1, W1 = self.gauss_grid(ng[0])
            Xi2, W2 = self.gauss_grid(ng[1])
            Xi1g, Xi2g = np.meshgrid(Xi1.flatten(), Xi2.flatten())
            Xi1 = np.array([Xi1g.flatten(), Xi2g.flatten()]).T
            W1g, W2g = np.meshgrid(W1.flatten(), W2.flatten())
            W1 = W1g.flatten() * W2g.flatten()
            return Xi1, W1
        elif len(ng) == 3:
            Xi1, W1 = self.gauss_grid(ng[0])
            Xi2, W2 = self.gauss_grid(ng[1])
            Xi3, W3 = self.gauss_grid(ng[2])
            gindex = np.mgrid[0:ng[0], 0:ng[1], 0:ng[2]]
            gindex = np.array([gindex[n].flatten() for n in [0,1,2]]).T #doesn't seem to work as default
            Xi = np.array([
                Xi1[gindex[:, 0]], Xi2[gindex[:, 1]], Xi3[gindex[:, 2]]])[:, :, 0].T
            W = np.array([
                W1[gindex[:, 0]], W2[gindex[:, 1]], W3[gindex[:, 2]]]).T.prod(1)
            return Xi, W

    raise ValueError('Invalid number of gauss points')


def eval_surface(self, res, boundary_points=False):
    faces = self.faces
    face_pts = []
    elem_pts = []
    xi_pts = []
    xi3grid = self.xi_grid(res=res, dim=3, surface=True, boundary_points=boundary_points).reshape(3,2,-1,3)
    for face in faces:
        grid_def = xi3grid[face[1], face[2]]
        elem_pts.append(np.ones(grid_def.shape[0]) * face[0])
        xi_pts.append(grid_def)
        face_pts.append(self.evaluate_embeddings(jnp.array([face[0]]),grid_def))
    coarse_pts = jnp.concatenate(face_pts, axis=0)
    return coarse_pts


def embed_points(self, points, verbose=0, init_elexi=None, fit_params=None, return_residual=False,
                 surface_embed=False, iterations=15, max_c=None, grid_res=10,
                 vis_max_norm=None, scene: Optional[pv.Plotter]=None,
                 dim_mask=None, robust_init_est=False,
                 approx_jac=False,
                 chunk_size=None, window_size=16, tol=None,
                 ):
    """Find the parametric coordinates (element, xi) for a set of physical-space points.

    Uses an approximate nearest-neighbour search on a coarse xi grid to
    obtain initial estimates, then refines with a JAX-accelerated
    Newton–Raphson descent.  Topology mapping (:meth:`topomap`) is
    applied at each iteration so that points near element boundaries are
    correctly assigned to neighbouring elements.

    The core computation is performed by a JIT-compiled function
    built once in :meth:`generate_mesh` (see :mod:`HOMER.embedding`),
    which eliminates redundant XLA retracing across calls.

    Parameters
    ----------
    points:
        Physical-space query points, shape ``(n_pts, fdim)``.
    verbose:
        Verbosity level.  ``0`` → silent; ``2`` → print mean/max
        residual; ``3`` → also render an error visualisation with
        PyVista.
    init_elexi:
        Pre-computed initial ``(elem_num, xis)`` tuple.  When supplied,
        the coarse nearest-neighbour search is skipped.
    fit_params:
        Optional parameter override for the mesh geometry.
    return_residual:
        When ``True``, returns a ``((elem_num, embedded), residual)``
        tuple instead of just ``(elem_num, embedded)``.
    surface_embed:
        Restrict the coarse search to the surface faces of a 3-D mesh.
    iterations:
        Number of refinement iterations.
    scene:
        pyvista plotter.
    dim_mask:
        a vector used to project against the distance in a subset of dimensions.
        Either ``(fdim,)`` — applied to every point — or ``(n_pts, fdim)``
        for a per-point mask; ``None`` keeps every dimension.  It is a
        *static* statement about which residual components exist, so it is
        coerced to ``bool`` and carries no tangent: differentiating
        ``embed_points`` never produces a derivative with respect to it.
        Masked components come back as exactly zero in both the residual
        and its Jacobian.  A row that masks out more dimensions than the
        mesh has parametric directions leaves the embedding
        under-determined; the derivative then takes the minimum-norm
        solution, and an all-``False`` row yields a zero residual and a
        zero Jacobian row.

        A non-trivial mask also switches the coarse search from the
        Morton Z-curve lookup to an exact one
        (:func:`~HOMER.utils.masked_closest_indices`).  A Z-curve code
        describes a whole coordinate vector, so it cannot express "these
        components only"; seeding from it meant the masked components
        still steered the match.  For a multi-state field that picks a
        different local minimum of the cross-state compromise, not merely
        a less precise seed, so the exact search is used whenever the
        mask makes one necessary.
    approx_jac:
        drops the calculation of the sliding term from the residual gradient estimation for embedding.
        Is less accurate, but recovers seperable derivatives by dimension, allowing further compression of the Jacobian.
        The jac estimate will^* have the right sign.
    chunk_size:
        When set, query points are processed in batches of at most
        this size.  Bounds peak memory to ``O(chunk_size)`` instead
        of ``O(n_pts)``, preventing swap on large inputs.
    window_size:
        Window width for the Morton-code nearest-neighbour coarse
        search (default 16).  Larger values improve coarse-search
        accuracy at the cost of memory and time; the Newton–Raphson
        refinement corrects for coarse-search misses.  Ignored when
        *dim_mask* masks anything off, since that case uses an exact
        search instead (see below).
    tol:
        Residual norm at which the refinement stops, making
        *iterations* an upper bound rather than a fixed count.
        ``None`` (the default) uses
        :data:`~HOMER.embedding.DEFAULT_EMBED_TOL` times the mesh's
        extent — float32 round-off, which the residual reaches within
        two or three iterations and then never leaves, so the remaining
        iterations were recomputing the same bits.  Pass ``0`` to
        iterate unconditionally.

        The refinement is vectorised over the query points, so the batch
        runs until its *slowest* point finishes: a point that cannot
        converge (one lying well off the mesh, say) keeps the whole batch
        going to *iterations*.  That costs nothing but the saving, and it
        does not perturb the others — a converged point's state is frozen,
        so every result is identical to the one it would get on its own.

    Returns
    -------
    elem_num : jnp.ndarray
        Element index for each query point, shape ``(n_pts,)``.
    embedded : jnp.ndarray
        Parametric coordinates, shape ``(n_pts, ndim)``.
    residual : jnp.ndarray
        (Only when *return_residual* is ``True``) Embedding error
        vectors, shape ``(n_pts, fdim)``.
    """
    if fit_params is None:
        fit_params = self.optimisable_param_array

    points = jnp.atleast_2d(points)

    # Rebuild the compiled embedding function when approx_jac or
    # robust_init_est differ from the defaults baked into the
    # cached function.  This is rare — most callers leave them
    # False and reuse the function built in generate_mesh().  The
    # variants are cached per flag pair: building one costs a full XLA
    # retrace, and a caller that wants a non-default flag almost always
    # wants it on every call (a fitting loop embedding the same points
    # against moving parameters), which used to retrace every time.
    embed_fn = self._mesh_embed_points
    if approx_jac or robust_init_est:
        key = (bool(approx_jac), bool(robust_init_est))
        cache = self.__dict__.setdefault("_embed_fn_variants", {})
        embed_fn = cache.get(key)
        if embed_fn is None:
            embed_fn = build_embedding_fn(self, approx_jac=approx_jac,
                                          robust_init_est=robust_init_est)
            cache[key] = embed_fn

    (elem_num, embedded), residual = embed_fn(
        points, fit_params, dim_mask,
        init_elexi, surface_embed, grid_res, iterations,
        chunk_size=chunk_size, window_size=window_size, tol=tol,
    )

    if verbose > 0:
        final_mean_dist = np.mean(np.linalg.norm(np.asarray(residual), axis=-1))
        final_max_dist  = np.max(np.linalg.norm(np.asarray(residual), axis=-1))
        print(f"final mean error of {final_mean_dist} units, max error of {final_max_dist}")

    if verbose >= 1:
        pass

    if verbose == 3: #three as an artifact of old scipy logging behaviour.
        locs = self.evaluate_embeddings_ele_xi_pair(elem_num, embedded, fit_params=fit_params)
        vec_errors = points - locs
        errors = np.linalg.norm(vec_errors, axis=-1)

        if vis_max_norm is not None:
            mask = errors < vis_max_norm
            locs = locs[mask]
            points = points[mask]
            errors = errors[mask]

        line_segs = np.concatenate(
            (np.atleast_2d(locs)[:, None], np.atleast_2d(points)[:, None]), axis=1
        )

        if scene is not None:
            s = scene
        else:
            s = pv.Plotter()
        self.plot(s, fit_params=fit_params, node_size=0.001, line_opacity=0.1, mesh_opacity=0.05)
        data = pv.PolyData(np.asarray(points))
        data['err'] = errors
        # data['constraint'] = constraints
        lines = pv.line_segments_from_points(line_segs.reshape(-1, 3))
        lines['err'] = np.repeat(errors, 2).copy()
        if max_c is None:
            max_c = np.percentile(errors, 99) * 1.1
        s.add_mesh(lines, line_width=4, clim=[0, max_c], render_lines_as_tubes=True)
        s.add_mesh(data, render_points_as_spheres=True, point_size=20, clim=[0, max_c])
        if scene is None:
            s.show()

    if return_residual:
        return (elem_num, embedded), residual

    return elem_num, embedded


def evaluate_sobolev(self, weights=None, fit_params=None,flatten=True):
    """
    Works out and defines the Sobolev values associated with the derivatives of the input elements.
    Then calculates the appropriate gauss points, and returns the elements assessed with the appropriate weighting. 
    """

    n_derivs = [len(b.deriv) for b in self.elements[0].basis_functions]
    d_order = [b.order for b in self.elements[0].basis_functions]
    if fit_params is None:
        fit_params = self.true_param_array

    gp, w = self.gauss_grid(d_order)
    deriv_combos = list(product(*[range(d) for d in n_derivs]))[1:] # skip the no deriv case
    n_eles = len(self.elements)

    if weights is None:
        weights = np.ones(len(deriv_combos))
    else:
        if not len(weights) == len(deriv_combos):
            raise ValueError("The length of the provided weights did not match the number of sobolev terms associated with this element")

    out_data = []
    for d, sw in zip(deriv_combos, weights):
        data = self.evaluate_deriv_embeddings_in_every_element(gp, d, fit_params=fit_params)

        weighted = (data.reshape(n_eles, -1, 3) * w[None, :, None])
        if not flatten:
            fshape = weighted.shape
        weighted = weighted.ravel() * sw
        if not flatten:
            weighted = weighted.reshape(fshape)
        out_data.append(weighted)

    return jnp.concatenate(out_data)


def get_volume(self, fit_params = None, element_wise=False):
    """
    Calculates the mesh volume by Gauss quadrature of the Jacobian determinant.

    The quadrature order is chosen by :func:`volume_quadrature_order` so
    that det(J) is integrated *exactly*, which makes the result exact (to
    float round-off) for any element the basis can describe -- not just
    affine ones.

    :param fit_params: an overide of the standard mesh parameters to use for fitting.
    :returns vol: The volume of the mesh, signed by element orientation.
    """
    if self.ndim != 3:
        raise ValueError(f"Volume is only defined on a 3-D mesh, this one is {self.ndim}-D")
    gauss_points, weights = self.gauss_grid(volume_quadrature_order(self.elements[0].basis_functions))
    Jmats = self.evaluate_jacobians_in_every_element(gauss_points, fit_params=fit_params)
    dets = jnp.linalg.det(Jmats).reshape(len(self.elements), -1)
    vols = dets * weights[None]
    if not element_wise:
        return jnp.sum(vols)
    return jnp.sum(vols, axis=-1)


@wide_eval
def evaluate_strain(self, element_ids, xis, othr: "Mesh", coord_function: Optional[Callable] = None, return_F=False, fit_params=None):
    """Evaluate the Green-Lagrange strain tensor between two mesh states.

    Computes the deformation gradient **F** = J_ref⁻¹ · J_def where J_ref
    is the Jacobian of *self* (reference configuration) and J_def is the
    Jacobian of *othr* (deformed configuration), then returns the strain
    tensor **E** = (Fᵀ F − I) / 2.

    Parameters
    ----------
    element_ids:
        1-D integer array, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.
    othr:
        A :class:`MeshField` representing the *deformed* configuration of
        the same topology.
    coord_function:
        Optional callable ``(mesh, eles, xis, Jmats) → Jmats`` that
        re-maps the Jacobian into a local coordinate frame (required for
        2-D manifold meshes).
    return_F:
        When ``True``, return the deformation gradient **F** instead of
        the strain tensor **E**.
    fit_params:
        Optional parameter override for *self*.

    Returns
    -------
    jnp.ndarray
        Green-Lagrange strain tensor ``E``, shape
        ``(n_pts, ndim, ndim)``, or the deformation gradient **F** if
        *return_F* is ``True``.

    Raises
    ------
    ValueError
        If called on a 2-D manifold mesh without supplying *coord_function*.
    """

    if self.ndim == 2 and coord_function is None:
        raise ValueError("Strain tensor on manifold mesh requires a coord function to provide a meaninful basis")

    deriv_self = self.evaluate_jacobians(element_ids, xis, fit_params=fit_params)
    deriv_othr = othr.evaluate_jacobians(element_ids, xis, fit_params=None)

    if coord_function is not None:
        deriv_self = coord_function(self, element_ids, xis, deriv_self)
        deriv_othr = coord_function(othr, element_ids, xis, deriv_othr)

    F = jnp.matrix_transpose(jnp.linalg.solve(jnp.matrix_transpose(deriv_self), jnp.matrix_transpose(deriv_othr)))
    if return_F:
        return F

    strain = (F.transpose(0,2,1) @ F - np.eye(3)[None])/2 #srtaings always 3D here
    return strain.reshape(-1, 3,3) #self.ndim, self.ndim)
