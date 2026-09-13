"""Render the HOMER wordmark, the mesh behind assets/HOMER.png.

The mesh itself is :func:`HOMER.examples.wordmark`, so this script is only the
camera: face-on down z reads it as text, and turning it shows it is a volume
mesh all along.
"""

import pyvista as pv

from HOMER.examples import wordmark

s = pv.Plotter(window_size=(2100, 900))
wordmark().plot(s, node_size=1, tiling=(10, 6))
s.view_xy()
s.camera.Dolly(2.0)
s.reset_camera_clipping_range()
s.show()
