"""Render ``docs/assets/favicon.png`` from the fitted H3 donut.

The favicon is the donut of ``examples/H3_donut`` at tab size, which is a
harsh constraint: a 32px tab gives the mesh about a thousand pixels to say
"Hermite surface" with.  Three things follow from that, and together they are
why this is a script rather than a saved render.

PyVista measures line and point sizes in pixels of the *render*, not of the
finished icon, so every width below is given as the width it lands at in a
32px tab and scaled up on the way in.  A width chosen by eye on the full-size
render comes out a tenth of a pixel once reduced, which is what made the first
favicon a grey smudge.

The hexagonal overlay is drawn far denser than it can resolve, deliberately.
Below a pixel the cells stop reading as lines and average into a tonal wash
that gives the tube a body, so the icon reads as a solid object with a hole
through it rather than a tangle of wires.

The icon is reduced from the render in a single LANCZOS step.  Reducing in
stages rounds the curve off the torus and leaves it visibly polygonal, which
is easy to mistake for the mesh itself being coarse.
"""

from pathlib import Path

import pyvista as pv
from PIL import Image, ImageDraw

from HOMER import load_mesh

ROOT = Path(__file__).resolve().parent.parent
DONUT = ROOT / "examples/H3_donut/h3_donut.json"
FAVICON = ROOT / "docs/assets/favicon.png"

RENDER = 2048
ICON = 256
CAMERA = [(2.3, -2.3, 2.7), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

#Widths as they land in a 32px tab; see the module docstring.
PX32 = RENDER / 32 * 0.9
ELEMENT_LINE = 1.25
NODE = 2.5
HEX_LINE = 0.30
HEX_OPACITY = 0.25
TILING = (20, 12)

#A full-bleed square reads as a hard-edged tile against browser chrome.
CORNER = 0.18


def render(path):
    """The donut over a transparent background, at full render resolution."""
    plotter = pv.Plotter(off_screen=True, window_size=(RENDER, RENDER),
                         border=False)
    load_mesh(DONUT).plot(plotter,
                          node_size=NODE * PX32, node_colour="red",
                          line_colour="black", line_width=ELEMENT_LINE * PX32,
                          mesh_colour="gray", mesh_opacity=HEX_OPACITY,
                          mesh_width=HEX_LINE * PX32, tiling=TILING)
    plotter.camera_position = CAMERA
    plotter.reset_camera()
    plotter.camera.zoom(1.3)
    plotter.screenshot(str(path), transparent_background=True)
    plotter.close()


def icon(render_path):
    """Trim to the donut, plate it, round the corners, reduce once."""
    img = Image.open(render_path).convert("RGBA")
    img = img.crop(img.getchannel("A").getbbox())
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))

    square = Image.alpha_composite(
        Image.new("RGBA", square.size, (255, 255, 255, 255)), square)

    mask = Image.new("L", square.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, side - 1, side - 1], radius=int(side * CORNER), fill=255)
    square.putalpha(mask)

    return square.resize((ICON, ICON), Image.LANCZOS)


if __name__ == "__main__":
    scratch = FAVICON.with_suffix(".render.png")
    render(scratch)
    icon(scratch).save(FAVICON)
    scratch.unlink()
    print(f"Saved {FAVICON.relative_to(ROOT)}")
