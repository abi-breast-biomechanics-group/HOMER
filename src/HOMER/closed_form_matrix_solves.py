"""
closed_form_matrix_solves.py - explicit 2x2 and 3x3 linear solves.

The Newton-Raphson refinement in :mod:`HOMER.embedding` solves one tiny
system per query point per iteration, vmapped over the whole point cloud.
Cramer's rule written out beats a general solver at that size and keeps the
whole step inside one XLA kernel, so these are used in place of
``jnp.linalg.solve``.
"""
import jax.numpy as jnp

def explicit_solve_2x2(A, b):
    """Solve ``A x = b`` for a 2x2 system by Cramer's rule.

    :param A: The 2x2 matrix.
    :param b: The 2-vector right-hand side.

    :returns:
        The 2-vector solution.  ``1e-12`` is added to the determinant, so a
        singular system returns a large finite value rather than a NaN --
        which keeps a vmapped Newton step from poisoning its whole batch.
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] + 1e-12
    ans_0 = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    ans_1 = (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det
    return jnp.array([ans_0, ans_1])

def explicit_solve_3x3(A, b):
    """Solve ``A x = b`` for a 3x3 system by cofactor expansion.

    :param A: The 3x3 matrix.
    :param b: The 3-vector right-hand side.

    :returns:
        The 3-vector solution, with the same ``1e-12`` determinant guard as
        :func:`explicit_solve_2x2`.
    """
    C00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    C01 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    C02 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    
    det = A[0, 0] * C00 + A[0, 1] * C01 + A[0, 2] * C02 + 1e-12
    
    C10 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    C11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    C12 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    
    C20 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    C21 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    C22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    ans_0 = (C00 * b[0] + C10 * b[1] + C20 * b[2]) / det
    ans_1 = (C01 * b[0] + C11 * b[1] + C21 * b[2]) / det
    ans_2 = (C02 * b[0] + C12 * b[1] + C22 * b[2]) / det
    return jnp.array([ans_0, ans_1, ans_2])
