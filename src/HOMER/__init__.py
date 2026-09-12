from HOMER.mesh import Mesh, MeshElement, MeshNode, MeshField, reorder_nodes
from HOMER.basis_definitions import (H3, L1, L3, L2, L4, B3,
                                     Basis, BasisGroup, Lagrange, basis_by_name)
#the pre-1.0 spellings; importable, but kept out of __all__ below
from HOMER.basis_definitions import (H3Basis, L1Basis, L3Basis, L2Basis,
                                     L4Basis, B3Basis)
from HOMER.io import load_mesh, save_mesh
from HOMER.jacobian_evaluator import (jacobian, make_jac_for_mesh_func,
                                     make_static_jac_for_mesh_func,
                                     matrix_free_jacobian)
from HOMER.geometry import cube
from HOMER.mac_plotting_patch import apply_macos_fullscreen_close_patch

# macOS tears down the Cocoa render window while it may still be in a
# full-screen space, which shows up as a bus error on close.  No-op elsewhere.
apply_macos_fullscreen_close_patch()

import os

import jax

# JAX compiles on the first call and that compile dominates it, so the
# persistent cache is worth having on.  As a *default* only: the setting is
# process-global and applies to every JAX computation in the program, not just
# HOMER's, so an explicit choice always wins -- either the
# JAX_COMPILATION_CACHE_DIR environment variable, or a jax.config.update made
# before HOMER is imported.  Set that variable to an empty string to opt out.
#
# The location is the platform's per-user cache directory rather than a shared
# /tmp: on a multi-user machine everyone would otherwise write to one place,
# and Windows has no /tmp at all.
#
# The size threshold is left at JAX's own default, which is what keeps the
# cache from growing without bound.
if ("JAX_COMPILATION_CACHE_DIR" not in os.environ
        and jax.config.jax_compilation_cache_dir is None):
    from platformdirs import user_cache_dir

    jax.config.update("jax_compilation_cache_dir", user_cache_dir("HOMER"))

# JAX stores a compilation only when it took longer than this, so that work too
# cheap to be worth a disk round trip is not written out.  Its default of one
# second is tuned for programs that compile a handful of large kernels; HOMER
# compiles many small ones, and a mesh that takes seconds to build does it in
# hundreds of compiles that each fall under the floor, so nothing is ever
# stored and the cache stays empty.  Lowering the floor is what makes it work
# at all.  As a default only, like the directory above.
if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.01)


#: The public surface.  The deprecated ``*Basis`` spellings are imported above
#: and still resolve, but are deliberately absent here, so ``from HOMER import
#: *`` and tab-completion offer one name per basis rather than two.
__all__ = [
    'Mesh', 'MeshElement', 'MeshNode', 'MeshField', 'reorder_nodes',
    'H3', 'L1', 'L2', 'L3', 'L4', 'B3',
    'Basis', 'BasisGroup', 'Lagrange', 'basis_by_name',
    'load_mesh', 'save_mesh',
    'jacobian', 'matrix_free_jacobian',
    'make_jac_for_mesh_func', 'make_static_jac_for_mesh_func',
    'cube',
]
