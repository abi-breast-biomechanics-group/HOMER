# Plotting

Every field draws itself with `plot()`.  `MeshField.plot` draws geometry;
`Mesh.plot` adds a secondary field on top of it.  Both return the
`pyvista.Plotter` when you pass one in, and show a new window when you do not.

```python exec="true" source="above" session="plotting"
from HOMER.geometry import cube

mesh = cube()
mesh.plot()          # opens a window
```

---

## Drawing into a scene you own

Pass a `pyvista.Plotter` and nothing is shown until you say so, which is how
you compose subplots, overlays and screenshots.

```python exec="true" source="above" session="plotting"
import pyvista as pv

from HOMER.examples import hermite_cube

before = hermite_cube()
after = hermite_cube()
after.refine(2)

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); before.plot(s)
s.subplot(0, 1); after.plot(s, node_colour='g')
s.link_views()
s.show()
```

Overlaying two configurations in one scene is the same idea, with the
reference drawn faintly:

```python exec="true" source="above" session="plotting"
mesh_ref = cube()
mesh_def = cube(scale=1.3)

s = pv.Plotter()
mesh_ref.plot(s, mesh_opacity=0.1)
mesh_def.plot(s, node_colour='g', mesh_opacity=0.1)
s.show()
```

---

## What gets drawn

A mesh is drawn as three layers, each with its own colour, width and opacity:

| layer | parameters |
| --- | --- |
| node markers | `node_colour`, `node_size`, `labels` |
| element edges | `line_colour`, `line_width`, `line_opacity` |
| element surfaces | `mesh_colour`, `mesh_width`, `mesh_opacity`, `tiling` |

`tiling` is the `(xn, yn)` repetition of the hexagonal unit surface used to
sample each element; a 5/3 ratio looks good, and the default is `(10, 6)`.

`labels=True` numbers the nodes.  It forces `node_size` to 0, because labels
and spheres at the same location are unreadable, so setting both warns.
`elem_labels=True` numbers the element centres.

```python exec="true" source="above" session="plotting"
mesh.plot(labels=True, elem_labels=True, mesh_opacity=0.25)
```

!!! note "Labelled scenes are pictures, not scenes"
    The label above is a still image rather than one of this site's turnable
    scenes.  Labels are drawn by a 2-D text actor that has to be re-placed
    every time the camera moves, and the web rasteriser these pages use for
    interactive scenes has no label engine at all — a labelled scene exported
    to it simply arrives with the numbers missing.  So anything labelled is
    rendered by VTK and published as a PNG.  Interactively, on your own
    machine, labels work normally.

---

## Colouring by a scalar

Pass an array instead of a colour name, and name the scalar so PyVista can
build the colour bar:

```python exec="true" source="above" session="plotting"
# one value per point of the hexagonal surface the draw samples
hex_pts, _ = mesh.get_hex_surface(list(range(len(mesh.elements))))
per_point_values = hex_pts[:, 2]

mesh.plot(mesh_colour=per_point_values, mesh_col_scalar_name='strain')
```

The same pairing works for `node_colour`/`node_col_scalar_name` and
`line_colour`/`line_col_scalar_name`.

---

## Drawing a secondary field

`Mesh.plot` takes `field_to_draw`, the key of a field created with
`new_field`:

```python exec="true" source="above" session="plotting"
import numpy as np

from HOMER import L1

rng = np.random.default_rng(0)
pts = rng.random((500, 3))

# outward directions from the cube centre, as tests/test_fields.py samples them
directions = pts - 0.5
directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

mesh.new_field('fibre', field_dimension=3, new_basis=L1**3,
               field_locs=pts, field_values=directions)

mesh.plot(field_to_draw='fibre', default_xi_res=6)
```

The default artist draws 3-D fields as line segments and 1-D scalar fields as
coloured spheres.  `field_artist` replaces it with a callable
`(plotter, locs, values, field_xi) -> None`:

```python exec="true" source="above" session="plotting"
def arrows(scene, locs, values, field_xi):
    scene.add_arrows(locs, values, mag=0.1)

mesh.plot(field_to_draw='fibre', field_artist=arrows)
```

`draw_xyz_field=False` suppresses the geometry so only the field is drawn,
and `field_xi` replaces the uniform `default_xi_res` grid with xi locations
of your own.

!!! warning "A field drawn over a volume is hard to read"
    A 3-D field sampled through a solid draws markers behind the front face
    and in front of the back one, so depth, overlap and the arbitrary glyph
    scale all fight the eye at once.  Keep `default_xi_res` low, or hand
    `field_xi` a single parametric plane and read one slice at a time.  Treat
    the picture as a check on direction and magnitude rather than as a
    measurement, and use `evaluate_embeddings` when you need a number.

A secondary field is itself a `MeshField`, so it can also draw alone:

```python exec="true" source="above" session="plotting"
mesh['fibre'].plot()
```

---

## Drawing a mesh under trial parameters

Every draw accepts `fit_params`, so an optimiser's current iterate — `result.x`
from a least-squares solve, say — can be shown without writing it back into
the mesh:

```python exec="true" source="above" session="plotting"
trial = np.asarray(mesh.optimisable_param_array) * 1.1

s = pv.Plotter()
mesh.plot(s)                                    # the mesh as it stands
mesh.plot(s, fit_params=trial, node_colour='g') # the same mesh under the iterate
s.show()
```

Drawn into one scene, the two are directly comparable: the mesh's own
parameters in red and the trial ones in green, with no change in mesh state.

---

## Replacing actors in a live scene

`render_name` prefixes the actors a draw creates, so a later draw with the
same name replaces them rather than stacking a second copy — which is what
you want when animating a fit.

```python
s = pv.Plotter()
s.show(interactive_update=True)
for step in iterates:
    mesh.plot(s, fit_params=step, render_name='fit')
    s.update()
```

---

## Rendering without a display

Off-screen rendering needs to be set before PyVista opens anything.  The test
suite does this in `tests/conftest.py`:

```python
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib
matplotlib.use("Agg")

import pyvista as pv
pv.OFF_SCREEN = True
```

With that in place, `s.screenshot('out.png')` works over SSH and in CI.

---

## Strain

`plot_strains` draws strain tensors evaluated at `(eles, xis)`; see the
[strain guide](strain.md).
