"""Open the fitted H3 donut mesh from ``fit_donut.py`` in an interactive scene."""
from pathlib import Path

from HOMER import load_mesh

HERE = Path(__file__).resolve().parent

donut = load_mesh(HERE / "h3_donut.json")
donut.plot()
