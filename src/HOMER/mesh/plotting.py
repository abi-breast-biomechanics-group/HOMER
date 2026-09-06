"""
plotting.py - drawing a mesh, and the surface extraction that feeds it.

Every function here is a method of :class:`~HOMER.mesh.field.MeshField` that
lives in its own module: the first argument is the field itself, and
:mod:`HOMER.mesh.field` binds them into the class.  They read the mesh and its
parameters, never mutate them, and are the only place PyVista is touched.
"""

import logging
from typing import Optional, Callable
from itertools import product

import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv

from HOMER.utils import spheres_to_polydata, jax_aknn

pv.global_theme.allow_empty_mesh = True


def get_surface(self, element_ids: Optional[np.ndarray] = None, res:int = 20, just_faces=False, tiling=None, fit_params=None) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
    """
    Returns a set of points evaluated over the mesh surface.
    """
    ele_iter  = [element_ids] if not isinstance(element_ids, list) else element_ids
    elements_to_iter = self.elements if element_ids is None else ele_iter
    if not just_faces:
        grid = self.xi_grid(res=res, dim=self.ndim, surface=True)
        if element_ids is not None:
            all_points = []
            for ne, e in enumerate(elements_to_iter):
                all_points.append(self.evaluate_embeddings(np.array([ne]), grid, fit_params=fit_params))
            return np.concatenate(all_points, axis=0) 
        else:
            return self.evaluate_embeddings_in_every_element(grid)
    else:
        face_pts = []

        if self.ndim == 3:
            faces = self.get_faces()
            if tiling is None:
                xi3grid = self.xi_grid(res=res, dim=3, surface=True).reshape(3,2,-1,3)
                for face in faces:
                    grid_def = xi3grid[face[1], face[2]]
                    face_pts.append(self.evaluate_embeddings(np.array([face[0]]),grid_def, fit_params=fit_params))
                return np.concatenate(face_pts, axis=0)

            c = []
            xi3grid, connectivity = self.xi_grid(res=res, dim=3, surface=True, lattice=tiling)
            connectivity = connectivity.reshape(-1, 3)
            xi3grid = xi3grid.reshape(3,2,-1,3)
            l_xi = xi3grid.shape[2]

            for idf, face in enumerate(faces):
                grid_def = xi3grid[face[1], face[2]]
                face_pts.append(self.evaluate_embeddings(np.array([face[0]]),grid_def, fit_params=fit_params))
                c.append([[0, idf * l_xi, idf * l_xi]] + connectivity)
            return np.concatenate(face_pts, axis=0), np.concatenate(c, axis=0)
        else:
            if tiling is None:
                xi2grid = self.xi_grid(res=res, dim=2)
                return np.asarray(self.evaluate_embeddings_in_every_element(xi2grid, fit_params=fit_params))
            xi2grid, connectivity = self.xi_grid(res=res, dim=2, lattice=tiling)
            lc = len(xi2grid)
            c = np.concatenate([connectivity.reshape(-1, 3) + [[0, idc * lc, idc * lc]] for idc in range(len(self.elements))], axis=0)
            return np.asarray(self.evaluate_embeddings_in_every_element(xi2grid, fit_params=fit_params)), c


def get_hex_surface(self, element_ids, tiling = (10, 6), fit_params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns lines evaluating a hexagon tiling of the element surface

    :params tiling: the repetitions of the underlying unit surface (5/3 ratio "looks good")
    """
    surface_points, single_face_connectivity = self.get_surface(element_ids, just_faces=True, tiling=tiling, fit_params=fit_params)
    return surface_points, single_face_connectivity.astype(int)


def get_triangle_surface(self, element_ids: Optional[np.ndarray] = None, res:int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a set of points evaluated over the mesh surface, and triangles to create the surface.

    :returns surface pts: Surface points evaluated over the mesh.
    :returns tris: the triangles creatign the mesh surface.
    """
    base_0 = np.array([0, 1, res])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(res - 1) * res)[:, None, None] 
    base_1 = np.array([res, 1, res + 1])[None, None] + np.arange(res - 1)[None, :, None] + (np.arange(res - 1) * res)[:, None, None] 
    surface_pts = self.get_surface(element_ids, just_faces=True, res=res)
    n_surfaces = surface_pts.shape[0]/(res**2)
    tris = (np.concatenate((base_0.reshape((-1,3)), base_1.reshape((-1,3))))[None] + np.arange(n_surfaces)[:, None, None] * res**2).reshape(-1,3)

    return surface_pts, tris


def get_lines(self, element_ids: Optional[list[int]|int|np.ndarray] = None, res=20, fit_params=None) -> pv.PolyData:
    """
    Returns a pv.PolyData object containing lines defining the edges of the mesh surface.
    """

    line_points = np.empty((0, 3))
    connectivity = np.empty((0, 3))
    blank_connectivity = np.column_stack((
        2 * np.ones(res - 1),
        np.arange(0, res - 1),
        np.arange(1, res)
    ))

    ele_iter  = [element_ids] if not isinstance(element_ids, list) else element_ids
    elements_to_iter = self.elements if element_ids is None else ele_iter #if we assume that all elements must be the same because it's easier.

    n_dim = self.elements[0].ndim
    residual_size = n_dim - 1 
    vals = [0, 1]
    combs = list(product(vals, repeat=residual_size)) #the combinations 
    all_xis = []

    total_ls = 0
    for i in range(n_dim):
        d = list(range(n_dim))
        d.pop(i)
        for comb in combs:
            xi_list = [0] * n_dim
            for cs, ind in zip(comb, d):
                xi_list[ind] = cs * np.ones(res)
            xi_list[i] = np.linspace(0, 1, res)
            xis = np.column_stack(xi_list)
            all_xis.append(xis)

            l_pts = total_ls
            connectivity = np.concatenate((
                connectivity,
                blank_connectivity + [0, l_pts, l_pts],
            ))
            total_ls += xis.shape[0]

    flat_xis = np.array(all_xis).reshape(-1, n_dim)

    lc = flat_xis.shape[0]
    n_ele = len(self.elements) 
    ele_up = lc * np.arange(n_ele)[None, :, None] * [0, 1, 1]
    long_connectivity = (connectivity[:, None] + ele_up).reshape(-1, 3)
    line_points = np.asarray(self.evaluate_embeddings_in_every_element(flat_xis, fit_params=fit_params)) #.reshape(n_ele, -1 , 3)[:2].reshape(-1, 3)

    mesh = pv.PolyData(
        line_points, 
        lines=long_connectivity.astype(int),
    )

    return mesh


def plot(self, scene:Optional[pv.Plotter] = None,
         node_colour: str | np.ndarray ='r', node_col_scalar_name='Field', node_size=10,
         labels = False, tiling=(10, 6), 
         mesh_colour: str | np.ndarray ='gray', mesh_opacity=0.1, mesh_width = 2, mesh_col_scalar_name="Field",
         line_colour: str | np.ndarray ='black', line_opacity=1, line_width=2, line_col_scalar_name="Field",
         elem_labels=False,
         render_name:Optional[str] = None,
         fit_params=None,
         ):
    """
    Draws the mesh as a pyvista scene.

    :param scene: A pyvista scene, if provided will not call .show().
    :param node_colour: The colour to draw the node values.
    :param node_size: The size of the node points.
    :param labels: Whether to label the node numbers.
    :param res: The resolution of the surface mesh.
    :param mesh_color: The mesh surface colour.
    :param mesh_opacity: The mesh surface opacity.
    :param elem_labels: Whether to label the mesh elements.

    """

    if labels:
        if not node_size == 10:
            logging.warning("Requested non-default node size, but setting node_size to 0 to allow labels to be visualised")
        node_size = 0

    is_tag = render_name is not None
    render_name = "" if render_name is None else render_name
    l_tag = render_name + "_lines" if is_tag else None 
    n_tag = render_name + "_nodes" if is_tag else None 
    h_tag = render_name + "_hexes" if is_tag else None 
    v_tag = render_name + "_nnums" if is_tag else None 
    e_tag = render_name + "_enums" if is_tag else None 
    render_name = None if is_tag else render_name


    #evaluate the mesh surface and evaluate all of the elements
    lines = self.get_lines(fit_params=fit_params)

    if fit_params is None:
        node_dots = np.array([n.loc for n in self.nodes])
    else:
        #the node markers are nodal parameters, not surface samples, so an
        #override has to be read out of the parameter vector directly
        params = np.asarray(self.true_param_array).copy()
        override = np.asarray(fit_params).ravel()
        if override.shape[0] == params.shape[0]:
            params = override
        else:
            params[self.optimisable_param_bool] = override
        #generate_mesh lays each node out as loc first, then its derivative fields
        node_dots = params.reshape(len(self.nodes), -1)[:, :self.fdim]


    s=pv.Plotter() if scene is None else scene


    if isinstance(line_colour, np.ndarray):
        lines[line_col_scalar_name] = line_colour

    s.add_mesh(lines, line_width=line_width, color=line_colour if not isinstance(line_colour, np.ndarray) else None, name=l_tag, opacity=line_opacity)
    node_dots_m = pv.PolyData(node_dots)
    if isinstance(node_colour, np.ndarray):
        node_dots_m[node_col_scalar_name] = node_colour

    s.add_mesh(node_dots_m, render_points_as_spheres=True, color=node_colour if not isinstance(node_colour, np.ndarray) else None, point_size=node_size, name=n_tag, cmap='tab10')

    # tri_surf, tris = self.get_triangle_surface(res=res)
    hex_surf, lines = self.get_hex_surface(list(range(len(self.elements))), tiling, fit_params=fit_params)
    surf_mesh = pv.PolyData(hex_surf, lines)

    if isinstance(mesh_colour, np.ndarray):
        surf_mesh[mesh_col_scalar_name] = mesh_colour
    # surf_mesh.faces = np.concatenate((3 * np.ones((tris.shape[0], 1)), tris), axis=1).astype(int)
    s.add_mesh(surf_mesh, style='wireframe', color=None if isinstance(mesh_colour, np.ndarray) else mesh_colour, opacity=mesh_opacity, name=h_tag, line_width=mesh_width, render_lines_as_tubes=True)
    if labels:
        s.add_point_labels(points = node_dots, labels=[str(i) for i in range(node_dots.shape[0])], name=v_tag)
    if elem_labels:
        elem_locs= np.ones((1, self.elements[0].ndim)) * 0.5
        pts = np.array(self.evaluate_embeddings_in_every_element(elem_locs))
        elem_labels = [f"elem: {i}" if self.elements[0].id is None else f"elem: ind {i}, id {self.elements[i].id}" for i in range(pts.shape[0])] 
        s.add_point_labels(points = pts, labels=elem_labels, name=e_tag)

    if scene is not None:
        return
    s.show()


def plot_strains(self, eles, xis, strains, scene:Optional[pv.Plotter]=None, cmap='coolwarm', spacer=4, show_max=False):
    """
    Given ele, xi locations, and the strain tensors evaluated at those locations, evaluates local strain ellipsoids, and plots them.
    """
    def get_batch_stretch_tensors(strains):
        m = strains.shape[0]
        I_batch = jnp.tile(jnp.eye(3), (m, 1, 1))
        C = 2 * strains + I_batch
        evals, evecs = jnp.linalg.eigh(C)
        safe_evals = jnp.maximum(evals, 0.0)
        sqrt_lambdas = jnp.sqrt(safe_evals)
        def reconstruct_single_U(v, s_lambdas):
            return v @ jnp.diag(s_lambdas) @ v.T
        U = jax.vmap(reconstruct_single_U)(evecs, sqrt_lambdas)
        return U
    locs = self.evaluate_embeddings_ele_xi_pair(eles, xis)
    sphere_base = pv.Sphere(radius = 1, theta_resolution=15, phi_resolution=15)

    test_pts = sphere_base.points[None, ..., None]
    U = get_batch_stretch_tensors(strains)

    def_pts = U[:, None] @ test_pts
    r_mag = np.linalg.norm(def_pts[..., 0], axis=-1) - 1

    if show_max:
        r_mag = np.broadcast_to(np.max(r_mag, axis=1, keepdims=True), r_mag.shape)

    pts = jax_aknn(locs, locs, k=2)[0]
    scale_to_use = np.median(pts[:, 1])/spacer

    sphere_pts = def_pts[..., 0] * scale_to_use + locs[:, None]

    sphere_arr = spheres_to_polydata(np.asarray(sphere_pts), sphere_base.faces)
    if show_max:
        sphere_arr['max length change'] = r_mag.flatten()
    else:
        sphere_arr['relative length change'] = r_mag.flatten()


    mean_c = np.nanmedian(np.abs(r_mag))
    mad_c = np.nanmedian(np.abs(np.abs(r_mag) - mean_c))
    max_c = mean_c + 4 * mad_c

    draw_flag = False
    if scene is None:
        scene = pv.Plotter()
        draw_flag= True

    self.plot(scene)
    scene.add_mesh(sphere_arr, smooth_shading=True, cmap=cmap,
                   # clim=[-max_c, max_c]
                   )
    if draw_flag:
        scene.show()


def plot_mesh(self, scene: Optional[pv.Plotter] = None, node_colour: str | np.ndarray ='r', node_col_scalar_name="Field", node_size=10, labels=False, tiling=(10, 6), 
         mesh_colour: str | np.ndarray = 'gray', mesh_opacity=0.1, mesh_width=2, mesh_col_scalar_name="Field", 
         line_colour: str | np.ndarray = 'black', line_opacity=1, line_width=2, line_col_scalar_name="Field",
         elem_labels=False, render_name: Optional[str] = None, 
         field_to_draw = None, field_xi = None, draw_xyz_field = True, field_artist: Optional[Callable[[pv.Plotter, np.ndarray, np.ndarray], None]] = None,
         default_field_point_size=25, default_xi_res=4, fit_params=None):
    """Draw the mesh and optionally overlay a secondary field.

    Parameters
    ----------
    scene:
        Existing :class:`pyvista.Plotter`.  When ``None``, a new plotter
        is created and shown.
    node_colour:
        Colour for node spheres.
    node_size:
        Node sphere size.
    labels:
        When ``True``, add node index labels (forces *node_size* = 0).
    tiling:
        ``(xn, yn)`` tiling for the hexagonal surface overlay.
    mesh_colour:
        Surface mesh colour.  Pass a :class:`numpy.ndarray` to colour-map
        by scalar values.
    mesh_opacity:
        Surface opacity (0–1).
    mesh_width:
        Line width for the hex wireframe.
    mesh_col_scalar_name:
        Scalar array name used when *mesh_colour* is an array.
    line_colour:
        Colour for the structural edge lines.
    line_opacity:
        Edge line opacity.
    line_width:
        Edge line width.
    line_col_scalar_name:
        Scalar name for colour-mapped edges.
    elem_labels:
        When ``True``, label element centres.
    render_name:
        Prefix for named actors (allows individual actor replacement in
        an interactive scene).
    field_to_draw:
        Name of a secondary field to visualise.  When ``None`` only the
        geometry is drawn.
    field_xi:
        Custom xi grid at which to evaluate the secondary field.
        Defaults to a uniform grid at *default_xi_res*.
    draw_xyz_field:
        When ``False``, suppress drawing of the primary geometry.
    field_artist:
        Custom callable ``(plotter, locs, values) → None`` for rendering
        the secondary field.  Defaults to line segments for 3-D fields
        and coloured spheres for 1-D scalar fields.
    default_field_point_size:
        Point size used by the default scalar field artist.
    default_xi_res:
        Xi grid resolution for the secondary field visualisation.
    """
    s_flag = False
    if scene is None:
        scene = pv.Plotter()
        s_flag = True

    #then you evaluate the field with the surface values throughout the mesh.
    if draw_xyz_field:
        #`MeshField.plot` *is* this module's `plot`; a zero-arg super() would need
        #the __class__ cell that only exists for a def inside a class body.
        plot(self, scene, node_colour, node_col_scalar_name, node_size, labels, tiling,
             mesh_colour, mesh_opacity, mesh_width, mesh_col_scalar_name,
             line_colour, line_opacity, line_width, line_col_scalar_name,
             elem_labels, render_name, fit_params)

    if field_to_draw == None:
        if s_flag:
            scene.show()
        return

    if field_xi is None:
        field_xi = self.xi_grid(res=default_xi_res, boundary_points=False)

    f_locs = self.evaluate_embeddings_in_every_element(field_xi)

    f_values = self[field_to_draw].evaluate_embeddings_in_every_element(field_xi)

    if field_artist is None:
        def field_artist(lscene, locs, values, field_xi):
            if self[field_to_draw].fdim == 3:
                #rather than arrows, create a line object.
                ldata = np.concatenate((locs[:, None], (locs + values)[:, None]), axis=1).reshape(-1, 3)
                lines = pv.line_segments_from_points(ldata)
                lines[field_to_draw] = np.linalg.norm(values, axis=-1)
                lscene.add_mesh(lines, render_lines_as_tubes=True, line_width=5)
            elif self[field_to_draw].fdim == 1:
                f = pv.PolyData(np.asarray(locs))
                f[field_to_draw] = np.asarray(values)
                lscene.add_mesh(f, render_points_as_spheres=True, point_size=default_field_point_size)
            else:
                raise ValueError(f"Default field artist doesn't support {self[field_to_draw].fdim} dimension fields, create a custom artist")

    field_artist(scene, f_locs, f_values, field_xi)

    if s_flag:
        scene.show()
    return
