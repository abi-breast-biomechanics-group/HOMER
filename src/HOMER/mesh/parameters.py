"""
parameters.py - the nodal parameter vector, and fitting it to data.

A field keeps its degrees of freedom in one flat array; these are the methods
of :class:`~HOMER.mesh.field.MeshField` that read it, write it, and solve for
it.  The first argument is the field itself, and :mod:`HOMER.mesh.field` binds
them into the class.
"""

import numpy as np
import jax.numpy as jnp 


def get_element_params(self, ele_num: int) -> np.ndarray:
    """
    returns the flat vector of node parameters associated with this element.
    """
    return self.true_param_array[self.ele_map[ele_num].astype(int)]


def update_from_params(self, inp_params, generate=True):
    """
        Updates all nodes with data from an input param array.

        :param inp_params: the input params to update the mesh with
        :param generate: whether to rebuild the mesh after updating.
    """

    if len(inp_params) == len(self.optimisable_param_array):
        params = self.true_param_array.copy()
        params[self.optimisable_param_bool] = inp_params 
    elif len(inp_params) == len(self.true_param_array):
        params = inp_params
        # self.true_param_array = inp_params
    else:
        raise ValueError("Input param array was provided that did not match either that set of parameters, or the optimisable subset of parameters")

    for node in self.nodes:
        node.loc, params = params[:self.fdim], params[self.fdim:]
        for key, value in node.items():
            l_val = value.flatten().shape[0]
            flat_node = node[key].ravel()
            flat_node[:], params = params[:l_val], params[l_val:] 
    if generate:
        self.generate_mesh()


def unfix_mesh(self):
    """
    Removes all fixed parameters in the mesh, and regenerates the mesh structure.
    """
    for node in self.nodes:
        node.unfix_params()
    self.generate_mesh()


def get_xi_weight_mat(self, eles, xis):
    """Build the linear weight matrix for least-squares fitting.

    For each query point ``(eles[i], xis[i])``, evaluates the basis
    function values and places them in the appropriate column positions of
    a global weight matrix **W**, where ``W[i, j]`` is the contribution
    of the *j*-th nodal degree of freedom to the *i*-th query point.

    This matrix is used by :meth:`linear_fit`::

        W * node_params = target_values   (solved in a least-squares sense)

    Parameters
    ----------
    eles:
        1-D integer array of element indices, shape ``(n_pts,)``.
    xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.

    Returns
    -------
    numpy.ndarray
        Weight matrix, shape ``(n_pts, n_nodes)``.
    """
    # out_weight = np.zeros((len(eles), len(self.true_param_array)//self.fdim)) #
    # unique_elem, inv = jnp.unique_inverse(eles)
    # for ide, e in enumerate(unique_elem):
    #     mask = ide == inv
    #     weight_mat = self.generate_weight_matrix(xis[mask]).T #weights associated with each of the parameters for the input matrix.
    #     relevant_weight_locs = (jnp.atleast_2d(self.ele_map)[e, ::self.fdim]//self.fdim).astype(int)
    #     out_weight[np.ix_(mask, relevant_weight_locs)] = weight_mat
    # return out_weight
    num_rows = eles.shape[0]
    num_cols = self.true_param_array.shape[0] // self.fdim
    weight_mat_all = self.generate_weight_matrix(xis).T
    K = weight_mat_all.shape[1]
    ele_map_2d = jnp.atleast_2d(self.ele_map)
    all_relevant_weight_locs = (ele_map_2d[eles, ::self.fdim] // self.fdim).astype(int)  # Shape: (num_rows, K)
    rows = jnp.repeat(jnp.arange(num_rows), K)
    cols = all_relevant_weight_locs.ravel()
    indices = jnp.column_stack([rows, cols])
    data = jnp.squeeze(weight_mat_all.ravel())

    # 4. Instantiate the sparse BCOO matrix
    out_weight = jnp.zeros((num_rows, num_cols))
    # jax.experimental.sparse.BCOO((data, indices), shape=(num_rows, num_cols))       
    return out_weight.at[(rows, cols)].add(data)


def linear_fit(self, targets, weight_mat, target_empty=-1, return_params=False, skip_bool=False):
    """Fit nodal parameters by solving a linear least-squares problem.

    Solves ``weight_mat @ params ≈ targets`` via :func:`jax.numpy.linalg.lstsq`
    and updates the mesh's nodal parameters with the solution.  This is
    the fastest fitting approach when the xi embeddings are fixed (i.e. the
    mesh topology does not change during fitting).

    The solve is column-equilibrated (see
    :func:`column_equilibrated_lstsq`), which costs nothing in exact
    arithmetic but recovers several digits in float32 for the bases whose
    weight matrices are badly scaled -- Hermite and B-spline especially.

    Parameters
    ----------
    targets:
        Target field values, shape ``(n_pts,)`` or ``(n_pts, fdim)``.
        Rows equal to *target_empty* (default ``-1``) are excluded from
        the fit.
    weight_mat:
        Weight matrix from :meth:`get_xi_weight_mat`,
        shape ``(n_pts, n_nodes)``.
    target_empty:
        Sentinel value used to mask out unused target rows.

    Notes
    -----
    Fixed parameters (set via :meth:`MeshNode.fix_parameter`) are
    currently **not** respected by this method.  Use the nonlinear
    optimisation pathway (``fitting.point_cloud_fit``) if constraints
    are required.
    """
    if not skip_bool: #just to make jax easier
        if targets.ndim > 1:
            target_mask = np.any(targets != target_empty, axis=-1)
        else:
            target_mask = targets != target_empty
        A = weight_mat[target_mask]
        b = targets[target_mask]
        assert A.shape[0] >= A.shape[1], "Attempted to solve an undertederimined system, more datapoints are needed"
    else:
        A = weight_mat
        b = targets

    new_params, residual, rank, s = column_equilibrated_lstsq(A, b)
    # if not skip_bool:
    #     if rank < A.shape[1]:
    #         logging.warning("Problem matrix was rank deficient. Try fitting (i) more datapoints, or (ii) a lower order field")
    #         pass

    # print('residual error:', residual)
    if return_params:
        return new_params.flatten()
    self.true_param_array = np.array(new_params).flatten()
    self.optimisable_param_array = self.true_param_array[self.optimisable_param_bool]
    self.update_from_params(new_params.flatten(), generate=False)
    self.generate_mesh()


def column_equilibrated_lstsq(A, b):
    """``jnp.linalg.lstsq`` with Jacobi column preconditioning.

    Each column of *A* is scaled to unit norm before the solve and the answer
    is scaled back afterwards.  For a system of full column rank this cannot
    change the minimiser -- it is unique and invariant under column scaling --
    so in exact arithmetic the result is identical to a plain ``lstsq``
    (measured: 3e-14 relative agreement under ``jax_enable_x64``, at a
    condition number of 2.1e7).  What it changes is the float32 error path.

    That matters because the weight matrices HOMER builds are badly scaled for
    the bases whose nodal parameters are not all the same kind of quantity.  A
    refined tricubic Hermite mesh has ``cond(W) = 4.4e4`` -- the derivative
    weights are an order of magnitude smaller than the value weights, cubed
    over three directions -- and B-spline control nets reach ``1.6e5``.  In
    float32 that costs four to five digits of the fitted geometry; equilibrated,
    those condition numbers fall to 4.1e2 and 2.5e3.

    Stays jit-able and differentiable: ``jnp.linalg.norm`` has a NaN gradient
    at an all-zero column, so the norm is formed as ``sqrt(sq + tiny)``, and a
    dead column is then left unscaled rather than divided by ~0 (which would
    otherwise amplify its round-off by ``1 / tiny``).

    Notes
    -----
    ``rank`` and the singular values come back from the *scaled* system.  For a
    rank-deficient system the minimum-norm tie-break is also taken in scaled
    coordinates, so the returned parameters differ from a plain ``lstsq`` --
    the fit itself does not.

    Parameters
    ----------
    A:
        Design matrix, shape ``(n_pts, n_params)``.
    b:
        Targets, shape ``(n_pts,)`` or ``(n_pts, fdim)``.

    Returns
    -------
    tuple
        ``(params, residual, rank, singular_values)``, as ``jnp.linalg.lstsq``.
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)

    sq = jnp.sum(A * A, axis=0)
    scale = jnp.where(sq > 0, jnp.sqrt(sq + jnp.finfo(A.dtype).tiny), 1.0)

    params, residual, rank, singular_values = jnp.linalg.lstsq(A / scale, b)
    return (params / scale.reshape((-1,) + (1,) * (params.ndim - 1)),
            residual, rank, singular_values)


def _pseudoinverse_matvec(J: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    Jt_v = J.T @ v          # (d,) — project v onto tangent space
    JtJ = J.T @ J           # (d, d) Gram matrix
    dxi, _, _, _ = jnp.linalg.lstsq(JtJ, Jt_v, rcond=None)
    return dxi
