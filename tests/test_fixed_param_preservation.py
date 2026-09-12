"""Fixed nodal parameters must survive refinement and rebasing.

A node of the new mesh inherits the constraints of the old node it sits exactly
on top of.  The correspondence is resolved from basis-node indices and exact
rational node locations, so none of this depends on the nodal coordinates or
their dtype.
"""

import logging
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from HOMER.mesh import Mesh, MeshField, MeshNode, MeshElement
from HOMER.basis_definitions import H3, L1, L2, L3, B3
from HOMER.geometry import basic_surface, cube

from _helpers import arr, bulged_patch


def node_at(mesh, loc, tol=1e-4):
    """Index of the node sitting at *loc* (test-side lookup only).

    The tolerance is loose because rebasing fits the new parameters by least
    squares, so a corner is only reproduced to the fit residual.  Nothing in
    the library itself matches nodes this way.
    """
    dists = np.linalg.norm(np.array([n.loc for n in mesh.nodes]) - np.asarray(loc, dtype=float), axis=1)
    idx = int(np.argmin(dists))
    assert dists[idx] < tol, f"no node at {loc}, closest was {dists[idx]}"
    return idx


def n_constrained(mesh):
    return sum(1 for n in mesh.nodes if n.fixed_params)


@contextmanager
def captured_warnings():
    """Collect root-logger warnings without depending on pytest's caplog."""
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


############################################### refinement

def test_refine_preserves_corner_constraints():
    mesh = basic_surface(basis=[H3] * 2)
    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter('loc', inds=[2])
    mesh.nodes[node_at(mesh, [0, 1, 1])].fix_parameter(['loc', 'du', 'dv', 'dudv'])
    mesh.generate_mesh()

    mesh.refine(2)

    partial = mesh.nodes[node_at(mesh, [0, 0, 0])]
    assert set(partial.fixed_params) == {'loc'}
    assert list(partial.fixed_params['loc']) == [2]

    full = mesh.nodes[node_at(mesh, [0, 1, 1])]
    assert set(full.fixed_params) == {'loc', 'du', 'dv', 'dudv'}
    for param in full.fixed_params.values():
        assert list(param) == [0, 1, 2]

    assert n_constrained(mesh) == 2 #the new midside/centre nodes stay free
    assert (~mesh.optimisable_param_bool).sum() == 1 + 4 * 3


def test_refine_restores_pinned_location_exactly():
    mesh = basic_surface(basis=[H3] * 2)
    mesh.nodes[node_at(mesh, [0, 1, 0])].fix_parameter('loc', values=np.array([0.0, 0.75, 0.25]))
    mesh.generate_mesh()

    mesh.refine(3)

    pinned = mesh.nodes[node_at(mesh, [0, 0.75, 0.25])]
    assert np.array_equal(pinned.loc, np.array([0.0, 0.75, 0.25])) #bit-exact, not just close


def test_refine_preserves_interior_lagrange_node():
    mesh = basic_surface(basis=[L2] * 2)
    mesh.nodes[node_at(mesh, [0, 0.5, 0.5])].fix_parameter('loc')
    mesh.generate_mesh()

    mesh.refine(2) #the element centre becomes a shared corner of the four children

    assert list(mesh.nodes[node_at(mesh, [0, 0.5, 0.5])].fixed_params['loc']) == [0, 1, 2]
    assert n_constrained(mesh) == 1


def test_refine_preserves_every_corner():
    corners = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    for basis in (L1, L2, L3, H3):
        mesh = basic_surface(basis=[basis] * 2)
        for corner in corners:
            mesh.nodes[node_at(mesh, corner)].fix_parameter('loc')
        mesh.generate_mesh()

        mesh.refine(2)

        assert n_constrained(mesh) == 4, basis
        for corner in corners:
            assert list(mesh.nodes[node_at(mesh, corner)].fixed_params['loc']) == [0, 1, 2], basis


def test_refine_by_non_uniform_xi_preserves_corners():
    mesh = basic_surface(basis=[L1] * 2)
    mesh.nodes[node_at(mesh, [0, 1, 1])].fix_parameter('loc', inds=[1])
    mesh.generate_mesh()

    mesh.refine(by_xi_refinement=[np.array([0, 0.3, 1.0]), np.array([0, 0.65, 1.0])])

    assert list(mesh.nodes[node_at(mesh, [0, 1, 1])].fixed_params['loc']) == [1]
    assert n_constrained(mesh) == 1


def test_refine_by_uniform_xi_preserves_interior_node():
    mesh = basic_surface(basis=[L2] * 2)
    mesh.nodes[node_at(mesh, [0, 0.5, 0.5])].fix_parameter('loc')
    mesh.generate_mesh()

    mesh.refine(by_xi_refinement=[np.array([0, 0.5, 1.0]), np.array([0, 0.5, 1.0])])

    assert n_constrained(mesh) == 1


def test_refine_can_be_opted_out_of():
    mesh = basic_surface(basis=[H3] * 2)
    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter('loc')
    mesh.generate_mesh()

    mesh.refine(2, preserve_fixed_params=False)

    assert n_constrained(mesh) == 0
    assert mesh.optimisable_param_bool.all()


def test_refine_3d_preserves_corner_constraints():
    mesh = cube(basis=[H3] * 3)
    mesh.nodes[node_at(mesh, [-0.5, -0.5, -0.5])].fix_parameter(['loc', 'dudvdw'])
    mesh.nodes[node_at(mesh, [0.5, 0.5, 0.5])].fix_parameter('loc', inds=[0])
    mesh.generate_mesh()

    mesh.refine([2, 3, 2]) #anisotropic, so a per-direction index mix-up would show

    assert set(mesh.nodes[node_at(mesh, [-0.5, -0.5, -0.5])].fixed_params) == {'loc', 'dudvdw'}
    assert list(mesh.nodes[node_at(mesh, [0.5, 0.5, 0.5])].fixed_params['loc']) == [0]
    assert n_constrained(mesh) == 2
    assert (~mesh.optimisable_param_bool).sum() == 3 + 3 + 1


def test_refine_3d_preserves_all_eight_corners():
    mesh = cube(basis=[L1] * 3)
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float) - 0.5
    for corner in corners:
        mesh.nodes[node_at(mesh, corner)].fix_parameter('loc')
    mesh.generate_mesh()

    mesh.refine(2)

    assert n_constrained(mesh) == 8
    assert len(mesh.nodes) == 27


############################################### rebasing

def test_rebase_to_higher_order_keeps_loc_only():
    mesh = basic_surface(basis=[L1] * 2)
    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter('loc')
    mesh.generate_mesh()

    out = mesh.rebase([H3] * 2)

    corner = out.nodes[node_at(out, [0, 0, 0])]
    assert set(corner.fixed_params) == {'loc'} #the new derivatives have nothing to inherit from
    assert n_constrained(out) == 1


def test_rebase_to_lower_order_drops_derivative_constraints():
    mesh = basic_surface(basis=[H3] * 2)
    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter(['loc', 'du', 'dv', 'dudv'])
    mesh.generate_mesh()

    with captured_warnings() as warnings:
        out = mesh.rebase([L1] * 2)

    assert set(out.nodes[node_at(out, [0, 0, 0])].fixed_params) == {'loc'}
    assert any('dropped' in w for w in warnings)


def test_rebase_adds_free_nodes_only():
    mesh = basic_surface(basis=[L1] * 2)
    for corner in ([0, 0, 0], [0, 1, 1]):
        mesh.nodes[node_at(mesh, corner)].fix_parameter('loc')
    mesh.generate_mesh()

    out = mesh.rebase([L2] * 2)

    assert len(out.nodes) == 9
    assert n_constrained(out) == 2
    assert not out.nodes[node_at(out, [0, 0.5, 0.5])].fixed_params


def test_rebase_restores_pinned_location_exactly():
    """The counterpart of ``test_refine_restores_pinned_location_exactly``.

    Rebasing fits the new parameters by least squares, so a pinned landmark
    only survives because the transfer re-asserts its value afterwards.  The
    bases here are interpolatory, which is the case where that is meaningful.
    """
    for basis in (L2, L3, H3):
        mesh = basic_surface(basis=[L1] * 2)
        mesh.nodes[node_at(mesh, [0, 1, 0])].fix_parameter('loc', values=np.array([0.0, 0.75, 0.25]))
        mesh.generate_mesh()

        out = mesh.rebase([basis] * 2)

        pinned = out.nodes[node_at(out, [0, 0.75, 0.25])]
        assert np.array_equal(np.asarray(pinned.loc), np.array([0.0, 0.75, 0.25])), basis


def test_a_pinned_value_is_not_truncated_into_an_integer_node():
    """``basic_surface`` states its corners as whole numbers, so an unguarded
    ``loc[inds] = values`` lands in an integer array and silently drops the
    fractional part -- the same trap ``update_from_params`` promotes around."""
    mesh = basic_surface(basis=[L1] * 2)
    assert mesh.nodes[0].loc.dtype.kind == 'i' #the precondition, not the point

    mesh.nodes[node_at(mesh, [0, 1, 0])].fix_parameter('loc', values=np.array([0.0, 0.75, 0.25]))

    assert np.array_equal(np.asarray(mesh.nodes[node_at(mesh, [0, 0.75, 0.25])].loc),
                          np.array([0.0, 0.75, 0.25]))


def test_a_rebased_mesh_fits_around_its_transferred_constraints():
    """The two halves together: rebasing carries the constraint over, and the
    next ``linear_fit`` on the result holds it instead of fitting through it.

    The fit targets a different surface from the one rebased, so the pinned
    corner is somewhere the unconstrained solve would not have left it.
    """
    pinned = np.array([0.0, 0.75, 0.25])
    mesh = basic_surface(basis=[L1] * 2)
    mesh.nodes[node_at(mesh, [0, 1, 0])].fix_parameter('loc', values=pinned)
    mesh.generate_mesh()
    out = mesh.rebase([L2] * 2)
    index = node_at(out, pinned)

    grid = out.xi_grid(6)
    eles = np.zeros(len(grid), dtype=int)
    out.linear_fit(arr(bulged_patch().evaluate_embeddings_ele_xi_pair(eles, grid)),
                   weight_mat=out.get_xi_weight_mat(eles, grid))

    assert np.array_equal(arr(out.nodes[index].loc), pinned)


def fit_residual(mesh, source, res=10):
    """``||W p - targets||`` on the system ``rebase`` builds -- the quantity the
    fit minimises, which is not the same as distance to the source surface."""
    egrid = source.xi_grid(res=res, boundary_points=False)
    el = np.repeat(np.arange(len(source.elements)), res ** source.ndim)
    xi = np.tile(egrid.reshape(-1, source.ndim), (len(source.elements), 1))
    targets = arr(source.evaluate_embeddings_ele_xi_pair(el, xi))
    weights = arr(mesh.get_xi_weight_mat(el, xi))
    return np.linalg.norm(weights @ arr(mesh.true_param_array).reshape(-1, source.fdim) - targets)


def test_a_constraint_is_honoured_by_the_fit_not_stamped_on_after_it():
    """Rebasing to a basis that cannot represent the source is where a pinned
    location actually binds, and where the ordering shows.

    Holding it through the fit lets the other parameters take up the slack; the
    alternative -- fit everything, then write the pinned value back over the
    answer -- leaves those parameters where the unconstrained solve put them.
    The first is a better fit by the measure the fit minimises, and cannot be
    worse: the stamped mesh satisfies the constraint too, so it is one of the
    candidates the constrained solve chooses among.

    (Refinement keeps the basis and reproduces its parent exactly, so the
    constraint is slack there and the ordering makes no difference.)
    """
    source = bulged_patch()                  #L2 and curved; L1 cannot hold it
    source.nodes[node_at(source, [0, 0, 1])].fix_parameter('loc')
    source.generate_mesh()
    pinned = arr(source.nodes[node_at(source, [0, 0, 1])].loc)

    held = source.rebase([L1] * 2)
    loose = source.rebase([L1] * 2, preserve_fixed_params=False)

    #the constraint has to bind, or the comparison below is vacuous
    assert np.abs(arr(loose.nodes[node_at(loose, [0, 0, 1], tol=0.6)].loc) - pinned).max() > 0.1

    stamped = loose
    stamped.nodes[node_at(stamped, [0, 0, 1], tol=0.6)].fix_parameter('loc', values=pinned)
    stamped.generate_mesh()

    np.testing.assert_array_equal(arr(held.nodes[node_at(held, [0, 0, 1], tol=0.6)].loc), pinned)
    assert fit_residual(held, source) < fit_residual(stamped, source)


def test_rebase_to_same_basis_is_unchanged():
    mesh = basic_surface(basis=[H3] * 2)
    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter(['loc', 'du'])
    mesh.generate_mesh()

    out = mesh.rebase([H3] * 2)

    assert set(out.nodes[node_at(out, [0, 0, 0])].fixed_params) == {'loc', 'du'}
    assert n_constrained(out) == 1


############################################### control-net (non-interpolatory) bases

def b3_surface():
    mesh = basic_surface(basis=[L1] * 2)
    mesh.refine(2) #B3 control points are shared, so more than one element is needed
    return mesh.rebase([B3] * 2)


def test_refine_b3_transfers_flag_without_restoring_the_coarse_value():
    mesh = b3_surface()
    for node in mesh.nodes:
        node.fix_parameter('loc')
    mesh.generate_mesh()

    unconstrained = deepcopy(mesh)
    unconstrained.unfix_mesh()

    mesh.refine(2)
    unconstrained.refine(2, preserve_fixed_params=False)

    assert n_constrained(mesh) > 0 #lattice-coincident control points inherit the flag
    assert (~mesh.optimisable_param_bool).sum() == 3 * n_constrained(mesh)
    #B3 is a control net: no coarse control value may be written onto the refined
    #net, so the refined parameters must be exactly what the fit produced.
    np.testing.assert_array_equal(mesh.true_param_array, unconstrained.true_param_array)


def test_refine_b3_drops_the_outer_control_points():
    """The phantom control points outside the domain have no refined counterpart."""
    mesh = b3_surface()
    for node in mesh.nodes:
        node.fix_parameter('loc')
    mesh.generate_mesh()

    with captured_warnings() as warnings:
        mesh.refine(2)

    assert n_constrained(mesh) < len(mesh.nodes)
    assert any('had no counterpart' in w for w in warnings)


def test_refine_b3_warns_that_the_constraint_is_approximated():
    mesh = b3_surface()
    for node in mesh.nodes:
        node.fix_parameter('loc')
    mesh.generate_mesh()

    with captured_warnings() as warnings:
        mesh.refine(2)

    assert any('not interpolatory' in w for w in warnings)


############################################### secondary fields

def test_field_constraints_survive_mesh_refine():
    mesh = basic_surface(basis=[L1] * 2)
    field = MeshField(nodes=[MeshNode(loc=np.zeros(1)) for _ in mesh.nodes],
                      elements=deepcopy(mesh.elements))
    field.nodes[node_at(mesh, [0, 1, 1])].fix_parameter('loc')
    field.generate_mesh()
    mesh['scalar'] = field

    mesh.nodes[node_at(mesh, [0, 0, 0])].fix_parameter('loc')
    mesh.generate_mesh()
    mesh.refine(2)

    assert n_constrained(mesh) == 1
    assert n_constrained(mesh['scalar']) == 1
    assert (~mesh['scalar'].optimisable_param_bool).sum() == 1


if __name__ == '__main__':
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
