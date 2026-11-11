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


def assemble_param_maps(self) -> None:
    """Assemble the elements' parameter maps into the field's arrays.

    The parameter vector is a sequence of *DOFs*, one per node field, each
    :attr:`fdim` entries long.  Every element slot is a weighted sum of DOFs
    (:attr:`MeshElement.param_map`), padded to the longest such sum in the
    mesh with zero-weight terms on DOF 0, which is what lets the evaluators
    apply every element's map as one gather:

    * ``ele_map`` -- ``(n_elements, n_slots, K)`` DOF indices,
    * ``ele_weights`` -- ``(n_elements, n_slots, K)`` their weights,
    * ``dof_node`` -- ``(n_dofs,)`` the node each DOF belongs to.

    :raises ValueError:
        If a node field is not :attr:`fdim` long, or an element draws on a
        field its node does not carry.
    """
    node_dofs, dof_node = [], []
    for idn, node in enumerate(self.nodes):
        fields = {'loc': node.loc, **node}
        if any(np.size(v) != self.fdim for v in fields.values()):
            raise ValueError(f"Every field of node {idn} must hold {self.fdim} values, "
                             "one per field component")
        node_dofs.append({f: len(dof_node) + i for i, f in enumerate(fields)})
        dof_node += [idn] * len(fields)
    self.dof_node = np.array(dof_node, dtype=int)

    rows = []
    for ide, element in enumerate(self.elements):
        pool = (element.nodes if element.used_index
                else [self.node_id_to_ind[n] for n in element.nodes])
        try:
            rows.append([[(node_dofs[pool[k]][f], w) for k, f, w in element.terms(slot)]
                         for slot in element.slots])
        except KeyError as missing:
            raise ValueError(f"Element {ide} draws on a field its node does not have: "
                             f"it did not have the required field {missing}") from None

    width = max(len(terms) for row in rows for terms in row)
    shape = (len(rows), len(rows[0]), width)
    self.ele_map = np.zeros(shape, dtype=int)
    self.ele_weights = np.zeros(shape)
    for ide, row in enumerate(rows):
        for slot, terms in enumerate(row):
            if terms:
                self.ele_map[ide, slot, :len(terms)], self.ele_weights[ide, slot, :len(terms)] = zip(*terms)


def get_element_params(self, ele_num: int) -> np.ndarray:
    """
    returns the flat vector of parameters of this element, its map applied.

    :param ele_num:
        Index of the element.

    :returns:
        The element's parameters drawn from :attr:`true_param_array`, slot
        by slot, each slot's components together.
    """
    dofs = np.asarray(self.true_param_array).reshape(-1, self.fdim)
    return (dofs[self.ele_map[ele_num]] * self.ele_weights[ele_num][..., None]).sum(-2).ravel()


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
            Weight matrix, shape ``(n_pts, n_dofs)``.
    """
    weights, columns = self.get_xi_weight_blocks(eles, xis)
    rows = jnp.repeat(jnp.arange(columns.shape[0]), columns.shape[1])
    out_weight = jnp.zeros((columns.shape[0], self.true_param_array.shape[0] // self.fdim))
    return out_weight.at[(rows, columns.ravel())].add(weights.ravel())


def get_xi_weight_blocks(self, eles, xis):
    """The weight matrix of :meth:`get_xi_weight_mat`, left unscattered.

    Every query point draws on exactly one element, so its row of the weight
    matrix has the same small number of non-zeros wherever it lands -- one per
    term of that element's parameter map.  :meth:`get_xi_weight_mat` scatters
    those into a dense ``(n_pts, n_dofs)`` array that is almost entirely zero;
    this returns them as they are computed, alongside the columns they belong
    in, which is all :func:`sparse_equilibrated_lstsq` needs.

    :param eles:
        1-D integer array of element indices, shape ``(n_pts,)``.
    :param xis:
        Parametric coordinates, shape ``(n_pts, ndim)``.

    :returns:
        tuple
            ``(weights, columns)``, both ``(n_pts, n_slots * K)``.
            ``weights[i, k]`` belongs at column ``columns[i, k]`` of row *i*:
            a basis weight times the map weight carrying it to that DOF.
            Repeats within a row are summed, as the dense form's scatter-add
            does.
    """
    eles = jnp.atleast_1d(jnp.asarray(eles)).astype(int)
    basis = self.generate_weight_matrix(xis).T.reshape(eles.shape[0], -1)
    weights = basis[..., None] * jnp.asarray(self.ele_weights)[eles]
    columns = jnp.asarray(self.ele_map)[eles]
    return weights.reshape(eles.shape[0], -1), columns.reshape(eles.shape[0], -1)


def _splu_augmented_solve(weights, columns, n_cols, top, bottom):
    """Solve the augmented least-squares system for the matrix *W* given by its blocks.

    ``[[I, W], [W.T, 0]] @ [r, y] = [top, bottom]``, whose second block row
    makes ``y`` the least-squares solution and ``r`` the residual::

        r = top - W y          W.T W y = W.T top - bottom

    Solved through the normal matrix, which is where the sparsity pays: it is
    symmetric positive definite, so a sparse LU under a symmetric ordering
    keeps the factor sparse.  Squaring the condition number to get there is
    safe only in float64 and only after the columns are equilibrated -- a
    B-spline control net starts at 1.6e5, which float32 cannot carry -- so this
    runs in float64 whatever the field's precision.
    """
    weights = np.asarray(weights, dtype=np.float64)
    columns = np.asarray(columns)
    top = np.asarray(top, dtype=np.float64)
    bottom = np.asarray(bottom, dtype=np.float64)

    n_rows, K = columns.shape
    #coo sums duplicate (row, column) pairs, as the dense form's scatter-add
    #does for a node an element refers to more than once
    W = sp.coo_matrix((weights.reshape(-1),
                       (np.repeat(np.arange(n_rows), K), columns.reshape(-1))),
                      shape=(n_rows, n_cols)).tocsc()

    y = splu((W.T @ W).tocsc(), permc_spec='MMD_AT_PLUS_A').solve(W.T @ top - bottom)
    return top - W @ y, y


def sparse_equilibrated_lstsq(weights, columns, n_cols, b):
    """Least squares on a mesh's block-sparse weight matrix.

    The sparse counterpart of :func:`column_equilibrated_lstsq`, for the
    systems built out of a mesh rather than handed in by a caller.  Those are
    block-sparse by construction -- a query point lies in one element, so its
    row touches only that element's nodes -- and the density falls as the mesh
    grows: a refined cube reaches 0.16% at four thousand elements, where the
    dense form is a 2.2 GB array that is 99.84% zeros.

    Columns are equilibrated exactly as in the dense solve.  Here it does more
    than recover digits: the solve forms ``W.T @ W``, squaring the condition
    number, which would be fatal at the 4.4e4 a refined tricubic Hermite mesh
    starts from and is merely uninteresting at the 4.1e2 equilibration leaves.

    Differentiable in both directions, so a refinement can sit inside a loss
    function.  The square system handed to :func:`jax.lax.custom_linear_solve`
    is the augmented one -- ``[[I, W], [W.T, 0]]`` against ``[b, 0]`` -- rather
    than the normal equations, and that choice is about precision, not
    elegance.  The normal form would put ``W.T @ b`` on the JAX side of the
    boundary, where it is summed in float32 before the solve ever sees it, and
    that rounding alone costs a B-spline fit a factor of 46.  The augmented
    form passes *b* across untouched and forms the product in float64 inside.
    JAX derives the JVP and the transpose from the operator either way, so
    forward and reverse mode both work and agree by construction, and gradients
    reach the weights as well as the targets.

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
            Fitted parameters, ``(n_cols,)`` or ``(n_cols, fdim)`` to match *b*.
    """
    weights, columns, b = jnp.asarray(weights), jnp.asarray(columns), jnp.asarray(b)
    one_dimensional = b.ndim == 1
    b = b[:, None] if one_dimensional else b

    squares = jnp.zeros(n_cols).at[columns].add(weights ** 2)
    #a column with no entries constrains nothing; leave it unscaled rather than
    #divide by ~0, which would amplify its round-off instead
    scale = jnp.where(squares > 0, jnp.sqrt(jnp.where(squares > 0, squares, 1.0)), 1.0)
    equilibrated = weights / scale[columns]

    n_rows = b.shape[0]

    def matvec(v):
        """``[[I, W], [W.T, 0]] @ v``, as gathers and scatter-adds."""
        r, y = v[:n_rows], v[n_rows:]
        return jnp.concatenate([
            r + (equilibrated[..., None] * y[columns]).sum(1),
            jnp.zeros_like(y).at[columns].add(equilibrated[..., None] * r[:, None, :])])

    def solve(_, rhs):
        out = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        return jax.pure_callback(
            lambda w, c, v: np.concatenate(
                _splu_augmented_solve(w, c, n_cols, v[:n_rows], v[n_rows:])).astype(out.dtype),
            out, equilibrated, columns, rhs)

    #one array rather than a pair: a tuple leaves the zero block with a
    #symbolic zero tangent, which custom_linear_solve will not take
    rhs = jnp.concatenate([b, jnp.zeros((n_cols, b.shape[1]), dtype=b.dtype)])
    y = jax.lax.custom_linear_solve(matvec, rhs, solve, symmetric=True)[n_rows:]
    y = y / scale[:, None]
    return y[:, 0] if one_dimensional else y


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
        field, leaving the mesh untouched.  Full length, with any held
        parameter carrying the value it already had, in the precision the
        solve ran at -- writing into the field restores those exactly, this
        does not, so that the return stays traceable.
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

    Parameters fixed with :meth:`MeshNode.fix_parameter` are held at their
    current values: they move to the right-hand side and the solve runs over
    the free columns only, which is the constrained minimiser rather than an
    unconstrained fit that is afterwards overwritten.  The point count the
    system needs falls with them -- ``n_pts`` must reach the number of *free*
    columns, not every column.

    Fixing is per component (``fix_parameter('loc', inds=[2])`` pins only
    *z*), so the free set can differ between the components of the field.
    *weight_mat* is shared across them, so components that share a free set
    share a solve: one for the usual meshes, at most *fdim* when the
    constraints cut across components.
    """
    n_cols = self.true_param_array.shape[0] // self.fdim
    free = self.optimisable_param_bool.reshape(n_cols, self.fdim)
    targets = targets[:, None] if targets.ndim == 1 else targets

    if sparse_columns is None and not skip_bool: #skipped to make jax easier
        kept = np.any(np.asarray(targets) != target_empty, axis=-1)
        weight_mat, targets = weight_mat[kept], targets[kept]
    if sparse_columns is not None:
        sparse_columns = np.asarray(sparse_columns)

    #the held parameters are a known contribution to the targets, so they move
    #to the right-hand side and leave a smaller system in the free ones
    current = np.asarray(self.true_param_array).reshape(n_cols, self.fdim)
    held = np.where(free, 0.0, current)
    new_params = jnp.asarray(held)

    #fix_parameter takes component indices, so each component of the field has
    #its own free set; weight_mat is shared across them, so components sharing
    #a free set share a solve.  Nothing fixed, or whole parameters fixed, is
    #one group.
    patterns, group = np.unique(free.T, axis=0, return_inverse=True)
    for pattern, cols in enumerate(patterns):
        comps = np.flatnonzero(group.ravel() == pattern)
        if not cols.any():
            continue #every parameter of these components is held
        b = targets[:, comps]
        if sparse_columns is not None:
            kept_cols = cols[sparse_columns]
            if not cols.all():
                b = b - (weight_mat[..., None] * held[sparse_columns][:, :, comps]).sum(1)
            solved = sparse_equilibrated_lstsq(jnp.where(kept_cols, weight_mat, 0.0),
                                               jnp.where(kept_cols, (np.cumsum(cols) - 1)[sparse_columns], 0),
                                               int(cols.sum()), b)
        else:
            A = weight_mat if cols.all() else weight_mat[:, cols]
            if not cols.all():
                b = b - weight_mat @ held[:, comps]
            if not skip_bool:
                assert A.shape[0] >= A.shape[1], "Attempted to solve an undertederimined system, more datapoints are needed"
            solved = column_equilibrated_lstsq(A, b)[0]
        new_params = new_params.at[np.ix_(cols, comps)].set(solved)

    if return_params:
        return new_params.flatten()
    #back in numpy at the field's own precision: the nodes are written through
    #in place, and a held parameter is restored from the source rather than
    #read back out of a solve that ran in float32, so it does not drift
    full = np.asarray(new_params).astype(current.dtype)
    full[~free] = current[~free]
    self.update_from_params(full.flatten(), generate=True)


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
