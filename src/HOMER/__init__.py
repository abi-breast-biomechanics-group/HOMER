from HOMER.mesher import Mesh, MeshElement, MeshNode, MeshField
from HOMER.basis_definitions import (H3Basis, L1Basis, L3Basis, L2Basis, L4Basis, B3Basis,
                                     Basis, BasisGroup, Lagrange, basis_by_name)
from HOMER.io import load_mesh, save_mesh
from HOMER.jacobian_evaluator import jacobian
from HOMER.geometry import cube
from HOMER.mac_plotting_patch import apply_macos_fullscreen_close_patch

# macOS tears down the Cocoa render window while it may still be in a
# full-screen space, which shows up as a bus error on close.  No-op elsewhere.
apply_macos_fullscreen_close_patch()

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
