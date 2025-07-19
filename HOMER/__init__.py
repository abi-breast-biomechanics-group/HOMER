from HOMER.mesher import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L1Basis, L3Basis, L2Basis
from HOMER.io import load_mesh, save_mesh
from HOMER.jacobian_evaluator import jacobian

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
