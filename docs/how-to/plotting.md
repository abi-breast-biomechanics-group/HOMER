# Plotting

Every field draws itself with `plot()`.  `MeshField.plot` draws geometry;
`Mesh.plot` adds a secondary field on top of it.  Both return the
`pyvista.Plotter` when you pass one in, and show a new window when you do not.

```python
from HOMER.geometry import cube

mesh = cube()
mesh.plot()          # opens a window
```

---

## Drawing into a scene you own

Pass a `pyvista.Plotter` and nothing is shown until you say so, which is how
you compose subplots, overlays and screenshots.

```python
import pyvista as pv

s = pv.Plotter(shape=(1, 2))
s.subplot(0, 0); before.plot(s)
s.subplot(0, 1); after.plot(s)
s.show()
```

Overlaying two configurations in one scene is the same idea, with the
reference drawn faintly:

```python
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

```python
mesh.plot(labels=True, elem_labels=True, mesh_opacity=0.25)
```

---

## Colouring by a scalar

Pass an array instead of a colour name, and name the scalar so PyVista can
build the colour bar:

```python
mesh.plot(mesh_colour=per_point_values, mesh_col_scalar_name='strain')
```

The same pairing works for `node_colour`/`node_col_scalar_name` and
`line_colour`/`line_col_scalar_name`.

---

## Drawing a secondary field

`Mesh.plot` takes `field_to_draw`, the key of a field created with
`new_field`:

```python
mesh.plot(field_to_draw='fibre', default_xi_res=6)
```

The default artist draws 3-D fields as line segments and 1-D scalar fields as
coloured spheres.  `field_artist` replaces it with a callable
`(plotter, locs, values) -> None`:

```python
def arrows(scene, locs, values):
    scene.add_arrows(locs, values, mag=0.1)

mesh.plot(field_to_draw='fibre', field_artist=arrows)
```

`draw_xyz_field=False` suppresses the geometry so only the field is drawn,
and `field_xi` replaces the uniform `default_xi_res` grid with xi locations
of your own.

A secondary field is itself a `MeshField`, so it can also draw alone:

```python
mesh['fibre'].plot()
```

---

## Drawing a mesh under trial parameters

Every draw accepts `fit_params`, so an optimiser's current iterate can be
shown without writing it back into the mesh:

```python
mesh.plot(fit_params=result.x)
```

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
