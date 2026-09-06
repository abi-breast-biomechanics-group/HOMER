"""Shared configuration for the HOMER test suite.

Every test here runs headless and unattended.  The scripts this suite grew
out of ended in ``plotter.show()`` and left the question "is that right?" to
whoever was sitting in front of the screen; each of those checks has been
replaced by an assertion against something knowable in advance -- an analytic
value, a conservation law, a round trip, or agreement between two independent
code paths.  The timing scripts that had no assertion to make at all now live
in ``benchmarks/`` and are not collected.

The backend switches below must run before anything imports ``pyplot`` or
opens a render window, which is why they sit at module scope in conftest.
"""

import os

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

matplotlib.use("Agg")

import pyvista as pv

pv.OFF_SCREEN = True

import pytest


@pytest.fixture
def plotter():
    """An off-screen plotter that is always torn down.

    Rendering is real -- ``screenshot`` returns pixels -- but nothing blocks
    waiting for a window to be closed.
    """
    scene = pv.Plotter(off_screen=True)
    try:
        yield scene
    finally:
        scene.close()
