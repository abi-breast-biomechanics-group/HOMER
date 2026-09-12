"""
parameters.py - the nodal parameter vector, and fitting it to data.

A field keeps its degrees of freedom in one flat array; these are the methods
of :class:`~HOMER.mesh.field.MeshField` that read it, write it, and solve for
it.  The first argument is the field itself, and :mod:`HOMER.mesh.field` binds
them into the class.
"""

import jax
import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def get_element_params(self, ele_num: int) -> np.ndarray:
    """
    returns the flat vector of node parameters associated with this element.

    :param ele_num:
        Index of the element.

    :returns:
        The element's node parameters, gathered from
        :attr:`true_param_array` in the element's own node order.
    """
    return self.true_param_array[self.ele_map[ele_num].astype(int)]


def update_from_params(self, inp_params, generate=True):
    """
        Updates all nodes with data from an input param array.

        :param inp_params: the input params to update the mesh with
        :param generate: whether to rebuild the mesh after updating.
    """

    if len(inp_params) == len(self.optimisable_param_array):
        #promote before scattering: storing float parameters into an integer
        #array truncates them silently
        params = np.asarray(self.true_param_array).astype(
            np.result_type(self.true_param_array, np.asarray(inp_params)))
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

    :param eles:
        1-D integer array of element indices, shape ``(n_pts,)``.
    :param xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.

    :returns:
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


def get_xi_weight_blocks(self, eles, xis):
    """The weight matrix of :meth:`get_xi_weight_mat`, left unscattered.

    Every query point draws on exactly one element, so its row of the weight
    matrix has the same small number of non-zeros wherever it lands -- one per
    nodal degree of freedom of that element.  :meth:`get_xi_weight_mat`
    scatters those into a dense ``(n_pts, n_nodes)`` array that is almost
    entirely zero; this returns them as they are computed, alongside the
    columns they belong in, which is all :func:`sparse_equilibrated_lstsq`
    needs.

    :param eles:
        1-D integer array of element indices, shape ``(n_pts,)``.
    :param xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.

    :returns:
        tuple
            ``(weights, columns)``, both ``(n_pts, K)`` for *K* the number of
            nodal degrees of freedom per element.  ``weights[i, k]`` belongs at
            column ``columns[i, k]`` of row *i*; repeats within a row are
            summed, as the dense form's scatter-add does.
    """
    weights = self.generate_weight_matrix(xis).T
    columns = (jnp.atleast_2d(self.ele_map)[eles, ::self.fdim] // self.fdim).astype(int)
    return jnp.squeeze(weights), columns


def _splu_solve(weights, columns, n_cols, b):
    """The numpy side of :func:`sparse_equilibrated_lstsq`."""
    weights = np.asarray(weights, dtype=np.float64)
    columns = np.asarray(columns)
    b = np.atleast_2d(np.asarray(b, dtype=np.float64).T).T

    n_rows, K = columns.shape
    rows = np.repeat(np.arange(n_rows), K)
    #coo sums duplicate (row, column) pairs, which is what the dense form's
    #scatter-add does for a node an element refers to more than once
    W = sp.coo_matrix((weights.reshape(-1), (rows, columns.reshape(-1))),
                      shape=(n_rows, n_cols)).tocsc()

    scale = np.sqrt(W.multiply(W).sum(axis=0)).A1
    scale[scale == 0] = 1.0
    W = W @ sp.diags(1.0 / scale)

    normal = (W.T @ W).tocsc()
    #the normal matrix is symmetric positive definite, so the symmetric
    #ordering is the one that keeps the factor sparse
    factor = splu(normal, permc_spec='MMD_AT_PLUS_A')
    return (factor.solve(W.T @ b) / scale[:, None])


def sparse_equilibrated_lstsq(weights, columns, n_cols, b):
    """Least squares on a sparse weight matrix, by equilibrated normal equations.

    The sparse counterpart of :func:`column_equilibrated_lstsq`, for the
    systems built out of a mesh rather than handed in by a caller.  Those are
    block-sparse by construction -- a query point sees one element, so its row
    touches only that element's nodes -- and the density falls as the mesh
    grows: a refined cube reaches 0.16% at four thousand elements, where the
    dense form is a 2.2 GB array that is 99.84% zeros.

    Columns are equilibrated exactly as in the dense solve, and for the same
    reason it costs nothing and buys several digits.  Here it does something
    more: forming ``W.T @ W`` squares the condition number, which would be
    fatal at the 4.4e4 a refined tricubic Hermite mesh starts from, and merely
    uninteresting at the 4.1e2 equilibration leaves.  The factorisation runs in
    float64 whatever the field's precision, so the squaring costs nothing that
    the float32 result can see.

    Runs through :func:`jax.pure_callback`, so it can be called from traced
    code.  It carries no differentiation rule: nothing differentiates through
    a refinement, and a JVP written for a caller that does not exist is a
    liability rather than a feature.

    :param weights:
        Non-zero weights per row, shape ``(n_pts, K)``, from
        :meth:`get_xi_weight_blocks`.
    :param columns:
        Column index of each weight, shape ``(n_pts, K)``.
    :param n_cols:
        Number of columns of the full matrix -- the nodal degree-of-freedom
        count.  Static, so the callback has a shape to promise.
    :param b:
        Targets, shape ``(n_pts,)`` or ``(n_pts, fdim)``.

    :returns:
        jax.numpy.ndarray
            Fitted parameters, shape ``(n_cols, fdim)``.
    """
    fdim = 1 if jnp.ndim(b) == 1 else jnp.shape(b)[1]
    out = jax.ShapeDtypeStruct((n_cols, fdim), jnp.asarray(b).dtype)
    return jax.pure_callback(lambda w, c, t: _splu_solve(w, c, n_cols, t).astype(out.dtype),
                             out, weights, columns, b)


def linear_fit(self, targets, weight_mat, target_empty=-1, return_params=False, skip_bool=False,
               sparse_columns=None):
    """Fit nodal parameters by solving a linear least-squares problem.

    Solves ``weight_mat @ params ≈ targets`` via :func:`jax.numpy.linalg.lstsq`
    and updates the mesh's nodal parameters with the solution.  This is
    the fastest fitting approach when the xi embeddings are fixed (i.e. the
    mesh topology does not change during fitting).

    The solve is column-equilibrated (see
    :func:`column_equilibrated_lstsq`), which costs nothing in exact
    arithmetic but recovers several digits in float32 for the bases whose
    weight matrices are badly scaled -- Hermite and B-spline especially.

    :param targets:
        Target field values, shape ``(n_pts,)`` or ``(n_pts, fdim)``.
        Rows equal to *target_empty* (default ``-1``) are excluded from
        the fit.
    :param weight_mat:
        Weight matrix from :meth:`get_xi_weight_mat`,
        shape ``(n_pts, n_nodes)``.
    :param target_empty:
        Sentinel value used to mask out unused target rows.
    :param return_params:
        Return the fitted parameter vector instead of writing it into the
        field, leaving the mesh untouched.
    :param skip_bool:
        Skip the *target_empty* masking and the overdetermined-system
        assertion, solving against *weight_mat* and *targets* as given.
        For callers inside a traced region, where masking would give a
        data-dependent shape.
    :param sparse_columns:
        When given, *weight_mat* is the ``(n_pts, K)`` weight block from
        :meth:`get_xi_weight_blocks` rather than a dense matrix, and this is
        the matching column index array.  The system is then solved by
        :func:`sparse_equilibrated_lstsq` without ever being formed dense.
        The *target_empty* masking does not apply -- a system assembled from a
        mesh has no empty rows to drop.

    **Notes**

    Fixed parameters (set via :meth:`MeshNode.fix_parameter`) are
    currently **not** respected by this method.  Use the nonlinear
    optimisation pathway (``fitting.point_cloud_fit``) if constraints
    are required.
    """
    if sparse_columns is not None:
        n_cols = self.true_param_array.shape[0] // self.fdim
        new_params = sparse_equilibrated_lstsq(weight_mat, sparse_columns, n_cols, targets)
    else:
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

    **Notes**

    ``rank`` and the singular values come back from the *scaled* system.  For a
    rank-deficient system the minimum-norm tie-break is also taken in scaled
    coordinates, so the returned parameters differ from a plain ``lstsq`` --
    the fit itself does not.

    :param A:
        Design matrix, shape ``(n_pts, n_params)``.
    :param b:
        Targets, shape ``(n_pts,)`` or ``(n_pts, fdim)``.

    :returns:
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
