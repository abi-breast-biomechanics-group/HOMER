from HOMER.mesh import Mesh, MeshElement, MeshNode, MeshField, reorder_nodes
from HOMER.basis_definitions import (H3Basis, L1Basis, L3Basis, L2Basis, L4Basis, B3Basis,
                                     Basis, BasisGroup, Lagrange, basis_by_name)
from HOMER.io import load_mesh, save_mesh
from HOMER.jacobian_evaluator import jacobian, matrix_free_jacobian
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
# The size and compile-time thresholds are left at JAX's own defaults.  They
# exist so that compilations too cheap to be worth storing are not stored,
# which is what keeps the cache from growing without bound.
if ("JAX_COMPILATION_CACHE_DIR" not in os.environ
        and jax.config.jax_compilation_cache_dir is None):
    from platformdirs import user_cache_dir

    jax.config.update("jax_compilation_cache_dir", user_cache_dir("HOMER"))
