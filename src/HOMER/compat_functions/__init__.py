"""
compat_functions - readers for mesh formats HOMER does not write itself.

* :mod:`HOMER.compat_functions.load_ipmesh` - OpenCMISS ``ipnode``/``ipelem``
  pairs, including node version numbering.
* :mod:`HOMER.compat_functions.load_VTU` - VTK unstructured grids.

Nothing here is imported by the rest of the library; each reader is brought in
explicitly by the caller that needs it.
"""
