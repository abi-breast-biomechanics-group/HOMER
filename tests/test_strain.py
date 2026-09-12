"""Green-Lagrange strain between a reference mesh and a deformed one.

``strain_test_george.py`` was a ten-case study driven by ``--test N``: apply a
known deformation, fit a mesh to it, evaluate the strain, and compare it
against a hand-derived deformation gradient -- on a scatter plot, by eye.
The ground truth was already written down in that script, so the comparison
is done here instead, for all ten cases at once.
"""

from copy import deepcopy

import numpy as np
import pytest

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L1Basis

from _helpers import arr

MP = 0.1        #deformation magnitude, as in the original script
TOL = 2e-4      #float32 through a linear fit and two Jacobian evaluations


def unit_cube_mesh():
    """Trilinear unit cube on [0, 1]^3, rebased to cubic Hermite.

    Hermite is needed because the deformations below are quadratic: the mesh
    must be able to represent them exactly for the strain to be checkable
    against a closed form.
    """
    locs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    element = MeshElement(node_indexes=list(range(8)),
                          basis_functions=(L1Basis, L1Basis, L1Basis))
    mesh = Mesh(nodes=[MeshNode(loc=np.array(l, dtype=float)) for l in locs],
                elements=element)
    return mesh.rebase([H3Basis] * 3)


def axial(axis):
    """X_axis = x_axis + mp * x_axis^2, everything else unchanged."""
    def deform(x):
        out = x.copy()
        out[:, axis] = x[:, axis] + MP * x[:, axis] ** 2
        return out

    def gradient(x):
        F = np.zeros((len(x), 3, 3)) + np.eye(3)
        F[:, axis, axis] = 1 + 2 * MP * x[:, axis]
        return F

    return deform, gradient


def combined(*axes):
    parts = [axial(a) for a in axes]

    def deform(x):
        out = x.copy()
        for axis in axes:
            out[:, axis] = x[:, axis] + MP * x[:, axis] ** 2
        return out

    def gradient(x):
        F = np.zeros((len(x), 3, 3)) + np.eye(3)
        for axis in axes:
            F[:, axis, axis] = 1 + 2 * MP * x[:, axis]
        return F

    return deform, gradient


def shear(target_axis, source_axis):
    """X_target = x_target + mp * x_source^2."""
    def deform(x):
        out = x.copy()
        out[:, target_axis] = x[:, target_axis] + MP * x[:, source_axis] ** 2
        return out

    def gradient(x):
        F = np.zeros((len(x), 3, 3)) + np.eye(3)
        F[:, target_axis, source_axis] = 2 * MP * x[:, source_axis]
        return F

    return deform, gradient


def complex_deformation():
    """Case 9: quadratic in x with cross-coupling into y and z."""
    def deform(x):
        return np.stack((x[:, 0] + MP * x[:, 0] ** 2 + MP * x[:, 0] + MP,
                         x[:, 1] + MP * (x[:, 0] - 1) + MP * x[:, 2],
                         x[:, 2] + MP * x[:, 2]), axis=1)

    def gradient(x):
        F = np.zeros((len(x), 3, 3)) + np.eye(3)
        F[:, 0, 0] = 1 + MP + 2 * MP * x[:, 0]
        F[:, 1, 0] = MP
        F[:, 1, 2] = MP
        F[:, 2, 2] = 1 + MP
        return F

    return deform, gradient


CASES = {
    "axial-x": axial(0),
    "axial-y": axial(1),
    "axial-z": axial(2),
    "axial-xy": combined(0, 1),
    "axial-xz": combined(0, 2),
    "axial-yz": combined(1, 2),
    "axial-xyz": combined(0, 1, 2),
    "shear-x-on-y": shear(0, 1),
    "shear-z-on-y": shear(2, 1),
    "complex": complex_deformation(),
}


def deformed_mesh(deform, fit_res=10):
    """Reference mesh plus a copy fitted to the deformation of its own points."""
    reference = unit_cube_mesh()
    fitted = deepcopy(reference)

    grid = reference.xi_grid(res=fit_res)
    eles = np.zeros(grid.shape[0], dtype=int)
    positions = arr(reference.evaluate_embeddings_in_every_element(grid))

    fitted.linear_fit(deform(positions), weight_mat=fitted.get_xi_weight_mat(eles, grid))
    return reference, fitted


def green_lagrange(F):
    return 0.5 * (np.transpose(F, (0, 2, 1)) @ F - np.eye(3))


@pytest.mark.parametrize("name", list(CASES))
def test_strain_matches_the_analytic_deformation_gradient(name):
    deform, gradient = CASES[name]
    reference, fitted = deformed_mesh(deform)

    grid = reference.xi_grid(res=5)
    strain = arr(reference.evaluate_strain_in_every_element(grid, fitted))
    positions = arr(reference.evaluate_embeddings_in_every_element(grid))

    np.testing.assert_allclose(strain, green_lagrange(gradient(positions)), atol=TOL)


@pytest.mark.parametrize("name", ["axial-x", "shear-z-on-y", "complex"])
def test_return_F_gives_the_deformation_gradient_itself(name):
    """E = (F^T F - I)/2, so the two outputs must be consistent."""
    deform, gradient = CASES[name]
    reference, fitted = deformed_mesh(deform)
    grid = reference.xi_grid(res=5)

    F = arr(reference.evaluate_strain_in_every_element(grid, fitted, return_F=True))
    strain = arr(reference.evaluate_strain_in_every_element(grid, fitted))

    np.testing.assert_allclose(green_lagrange(F), strain, atol=1e-6)
    np.testing.assert_allclose(F, gradient(arr(reference.evaluate_embeddings_in_every_element(grid))),
                               atol=TOL)


def test_strain_of_an_undeformed_mesh_is_zero():
    reference = unit_cube_mesh()
    grid = reference.xi_grid(res=4)

    strain = arr(reference.evaluate_strain_in_every_element(grid, deepcopy(reference)))

    np.testing.assert_allclose(strain, 0.0, atol=1e-6)


def test_rigid_motion_produces_no_strain():
    """Strain is frame-indifferent: a rotation and translation must not register."""
    import jax.numpy as jnp
    from HOMER.utils import rodrigues_exp

    rotation = np.asarray(rodrigues_exp(jnp.array([0.4, -0.7, 0.2])), dtype=float)

    def rigid(x):
        #jax leaves numpy's error state tripped, so this would otherwise warn
        with np.errstate(all='ignore'):
            return np.asarray(x, dtype=float) @ rotation.T + np.array([3.0, -1.0, 2.0])

    reference, moved = deformed_mesh(rigid)
    grid = reference.xi_grid(res=4)

    strain = arr(reference.evaluate_strain_in_every_element(grid, moved))

    np.testing.assert_allclose(strain, 0.0, atol=1e-4)


def test_uniform_stretch_gives_the_textbook_strain():
    """A stretch of lambda along x gives E_xx = (lambda^2 - 1)/2."""
    stretch = 1.2
    reference, fitted = deformed_mesh(lambda x: x * [stretch, 1.0, 1.0])
    grid = reference.xi_grid(res=4)

    strain = arr(reference.evaluate_strain_in_every_element(grid, fitted))

    expected = np.zeros((len(grid), 3, 3))
    expected[:, 0, 0] = (stretch ** 2 - 1) / 2
    np.testing.assert_allclose(strain, expected, atol=TOL)


def test_strain_is_available_per_element_pair_and_wide():
    reference, fitted = deformed_mesh(CASES["axial-x"][0])
    grid = reference.xi_grid(res=4)
    eles = np.zeros(len(grid), dtype=int)

    everywhere = arr(reference.evaluate_strain_in_every_element(grid, fitted))
    paired = arr(reference.evaluate_strain_ele_xi_pair(eles, grid, fitted))

    np.testing.assert_allclose(paired, everywhere, atol=1e-6)


def test_plot_strains_draws_without_a_window(plotter):
    """``test_strain_eval.py`` ended in ``plotter.show()``; this only checks
    that the helper builds drawable geometry."""
    reference, fitted = deformed_mesh(CASES["complex"][0])
    grid = reference.xi_grid(res=3)
    strain = arr(reference.evaluate_strain_in_every_element(grid, fitted))

    reference.plot_strains(eles=np.zeros(grid.shape[0], dtype=int), xis=grid,
                           strains=strain, scene=plotter)

    assert len(plotter.actors) > 0
