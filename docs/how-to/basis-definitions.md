# Basis Function Definitions

Every field HOMER holds is a weighted sum of basis functions over an element's parametric coordinates, each of which runs over
$\xi \in [0, 1]$.  `HOMER.basis_definitions` is where those functions are
written down, and this guide draws them, so that what an element is doing
between its nodes is visible rather than inferred.

The development follows Sir Peter Hunter and Andrew Pullan's
[*FEM/BEM Notes*](https://auckland.figshare.com/articles/journal_contribution/FEM_BEM_Notes/5440000),
the Auckland course notes these elements come from; section numbers below
point into Chapter 1, *Finite Element Basis Functions*.

---

## A field over one element

A one-dimensional field over an element is the sum of its nodal parameters
$u_n$, each weighted by a basis function (§1.1–1.2):

$$u(\xi) = \sum_n \phi_n(\xi)\, u_n$$

A `Basis` is that list of $\phi_n$, plus what a mesh needs to know about them.

```python exec="true" source="above" session="basis-definitions"
import numpy as np
from HOMER import L1

print(L1.weights)          # what each basis function weights
print(L1.node_locs)        # where in [0, 1] its node sits
print(L1.order)            # polynomial order

xi = np.linspace(0, 1, 5)
print(np.asarray(L1.fn(xi)))   # (n_pts, n_basis)
```

`fn` takes the $\xi$ values and returns one column per weight, so the field is
`phi @ params` and nothing more.  It is the same call for every basis on this
page; only the number of columns changes.

---

## The Lagrange family

`L1` to `L4` are the Lagrange bases: order 1 to 4, with nodes spread evenly
across the element and one parameter — the field value — at each (§1.2, §1.4).

```python exec="true" source="above" session="basis-definitions"
import matplotlib.pyplot as plt
from HOMER import L2, L3, L4

xi = np.linspace(0, 1, 201)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
for ax, basis in zip(axes.flat, (L1, L2, L3, L4)):
    for weight, phi in zip(basis.weights, np.asarray(basis.fn(xi)).T):
        ax.plot(xi, phi, label=weight)
    ax.vlines(basis.node_locs, -0.4, 1.1, colors='k', lw=0.5, ls=':')
    ax.axhline(0, lw=0.5, c='k')
    ax.set_title(f'{basis.name} — order {basis.order}')
    ax.legend(fontsize=8, ncols=2)

fig.supxlabel('xi'); fig.supylabel('phi')
plt.show()
```

Two properties are visible in every panel.
Firstly, $\phi_n$ is 1 at its own node and 0 at every other one. 
$\phi_a(\xi_b) = \delta_{ab}$, so the parameters *are* the field values at the nodes.
This is what the `interpolatory` property denotes.

```python exec="true" source="above" session="basis-definitions"
print(np.round(np.asarray(L3.fn(np.array(L3.node_locs))), 6))
print(L3.interpolatory)
```

Secondly, the functions sum to 1 everywhere.  A constant field is reproduced
exactly, and so is a rigid translation of the geometry — an element cannot
distort simply by being moved:

```python exec="true" source="above" session="basis-definitions"
print(np.asarray(L4.fn(xi)).sum(axis=1)[:4])
```

---


## Cubic Hermite: value and slope

`H3` carries two parameters at each of its two nodes: the field value, and its
derivative with respect to $\xi$.  Four parameters means the basis functions
are cubic, and they are chosen so that each one contributes to exactly one of
those four quantities (§1.6, eq. 1.14):

```python exec="true" source="above" session="basis-definitions"
from HOMER import H3

print(H3.weights)      # value and slope, at node 0 then node 1
print(H3.node_locs)

fig, (ax, dax) = plt.subplots(1, 2, figsize=(10, 4))
for weight, phi, dphi in zip(H3.weights,
                             np.asarray(H3.fn(xi)).T,
                             np.asarray(H3.deriv[1](xi)).T):
    ax.plot(xi, phi, label=weight)
    dax.plot(xi, dphi, label=weight)

for a, title in ((ax, 'H3'), (dax, 'dH3/dxi')):
    a.vlines(H3.node_locs, *a.get_ylim(), colors='k', lw=0.5, ls=':')
    a.axhline(0, lw=0.5, c='k')
    a.set_title(title); a.set_xlabel('xi'); a.legend(fontsize=8)

plt.show()
```

The left panel is the `x0`/`x1` pair behaving like `L1`'s, but flattened to
zero slope at both ends, and the `dx0`/`dx1` pair vanishing at both nodes —
they contribute no value there.  The right panel is where they do their work: each derivative
weight has unit slope at its own node and zero slope at the other, so the
parameter multiplying it *is* $\partial u / \partial \xi$ there.

```python exec="true" source="above" session="basis-definitions"
ends = np.array(H3.node_locs)
print(np.round(np.asarray(H3.fn(ends)), 6))        # values at the nodes
print(np.round(np.asarray(H3.deriv[1](ends)), 6))  # slopes at the nodes
```

Because that derivative is shared with the neighbouring element, the field is
C¹ across the boundary rather than merely continuous — the reason `H3` is the
default for geometry that has to stay smooth.

The two value weights still sum to 1, and a translation leaves the derivative
parameters untouched, so a moved element is undistorted here too.  What each
node has to store
follows from how many directions are Hermite, which is what `node_fields`
describes and `used_node_fields` resolves for a given element:

```python exec="true" source="above" session="basis-definitions"
from HOMER import MeshElement

print(MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=H3**2).used_node_fields)
print(MeshElement(node_indexes=list(range(8)), basis_functions=H3**3).used_node_fields)
```

!!! note "Derivatives are with respect to xi, not arclength"
    The notes take one further step (§1.6, eq. 1.15): each nodal derivative is
    scaled by an element scale factor, so that it is the *arclength*
    derivative that is shared between elements.  HOMER keeps the
    $\xi$-derivative itself and leaves `MeshElement.scale_factors` unset —
    the same interpolation with every scale factor taken as 1.  A model
    exported with scale factors therefore needs them folded into its nodal
    derivatives before it is read in.

---

## Cubic B-spline: parameters that are not nodal values

`B3` is the odd one out.  Its four parameters per direction are control
points, shared with the neighbouring elements rather than owned by this one,
and the curve does not pass through them:

```python exec="true" source="above" session="basis-definitions"
from HOMER import B3

print(B3.weights, B3.node_locs, B3.interpolatory)

fig, ax = plt.subplots(figsize=(6, 4))
for weight, phi in zip(B3.weights, np.asarray(B3.fn(xi)).T):
    ax.plot(xi, phi, label=weight)
ax.plot(xi, np.asarray(B3.fn(xi)).sum(axis=1), 'k--', lw=1, label='sum')
ax.set_title('B3'); ax.set_xlabel('xi'); ax.legend(fontsize=8)
plt.show()
```

No weight reaches 1, and at either end three of the four are still in play —
the element starts at $(P_0 + 4P_1 + P_2)/6$, a blend of three control points
rather than a point on the net:

```python exec="true" source="above" session="basis-definitions"
print(np.round(np.asarray(B3.fn(np.array([0., 1.]))), 4))
```

That overlap is what buys C² continuity, and the `node_locs` of `-1, 0, 1, 2`
say the same thing from the element's side: its support reaches one control
point back and two forward, and those are shared with the elements on either
side.  The sum is still 1, so the geometry still follows the control net
rigidly.

`interpolatory=False` is the flag the rest of HOMER reads: refinement and
`rebase` use it to decide whether a fixed nodal value may be carried across
verbatim.  See [Basis conversion](rebase.md).

---

## Basis functions as weighting functions

Read alternatively, $\phi_n$ is a *weighting function* on node $n$'s parameter, zero by the time neighbouring nodes are
reached (§1.3).

The Galerkin method is the step of using those same functions as the weights
in a weighted-residual statement of the governing equation (§2.1, and §2.3 for why that choice of weight).

```python exec="true" source="above" session="basis-definitions"
from HOMER.geometry import cube

mesh = cube()
gauss_xi, gauss_w = mesh.gauss_grid([3, 3, 3])
print(gauss_xi.shape, gauss_w.sum())
```

27 points across the element, with weights summing to the volume of the unit
$\xi$-cube they cover.

The Sobolev smoothing term used when fitting is one such integral already
assembled — `evaluate_sobolev` weights every mixed derivative of the element
at those Gauss points — and is described under
[Mesh fitting](fitting.md).

---

## The outer product

A 2-D element is the outer product of two 1-D bases, one per parametric
direction (§1.5):

$$\Phi_{ab}(\xi_0, \xi_1) = \phi_a(\xi_0)\, \phi_b(\xi_1)$$

For `L1` in both directions this gives the four bilinear weights of Figure
1.9 of the notes — each 1 at its own corner and 0 at the other three, the
Kronecker property carried over from 1-D by the product:

```python exec="true" source="above" session="basis-definitions"
grid = np.linspace(0, 1, 41)
xi0, xi1 = np.meshgrid(grid, grid, indexing='ij')
phi = np.asarray(L1.fn(grid))

fig, axes = plt.subplots(2, 2, figsize=(9, 7),
                         subplot_kw={'projection': '3d'})
for ax, (a, b) in zip(axes.flat, [(0, 0), (1, 0), (0, 1), (1, 1)]):
    ax.plot_surface(xi0, xi1, np.outer(phi[:, a], phi[:, b]), cmap='viridis')
    ax.set_title(f'{L1.weights[a]}(xi_0) * {L1.weights[b]}(xi_1)', fontsize=9)
    ax.set_xlabel('xi_0', labelpad=-6); ax.set_ylabel('xi_1', labelpad=-6)
    ax.set_zticks([0, 1]); ax.tick_params(labelsize=6, pad=-2)

plt.show()
```

A 3-D element is the same construction with a third factor, and the number of
element parameters is the product of the 1-D counts: 4 for `L1**2`, 8 for
`L1**3`, 16 for `H3**2`.

### How the product is indexed

An element does not store that surface; it stores which pair of 1-D weights
each of its parameters multiplies.  That is `BasisProductInds`, and the
evaluation is `N2_weights` — one multiply per pair:

```python exec="true" source="above" session="basis-definitions"
from HOMER.basis_definitions import N2_weights

element = MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=L1**2)
print(element.BasisProductInds)

point = np.array([[0.3, 0.7]])
w0, w1 = L1.fn(point[:, 0]), L1.fn(point[:, 1])
print(np.ravel(N2_weights(w0, w1, element.BasisProductInds)))
print([float(w0[0, a] * w1[0, b]) for a, b in element.BasisProductInds])
```

The pairs are ordered to match the parameter vector, which runs node by node
with each node's derivative fields in a fixed order.  For `H3**2` that
ordering is legible in the indices themselves — the first four pairs are node
0's value, `du`, `dv` and `dudv`, built from `x0`/`dx0` in each direction:

```python exec="true" source="above" session="basis-definitions"
print(MeshElement(node_indexes=[0, 1, 2, 3], basis_functions=H3**2).BasisProductInds[:4])
```

[Node indexing](node-indexing.md) covers that ordering from the mesh side.

### The algebra is the product

Which is why the directions of an element are combined with `*`:

```python exec="true" source="above" session="basis-definitions"
print(L1 * L1)      # bilinear, the element drawn above
print(H3 * L1)      # Hermite in xi_0, linear in xi_1
print(H3**3)        # tricubic Hermite; ** is the tensor power
```

`H3 * L1` is not a pointwise product of two functions — it is the statement
that the element takes the outer product of a Hermite direction with a linear
one, in that order.  The result is a `BasisGroup`, and
[Mixed basis functions](mixed-basis.md) is the guide to using them.

---

## Further reading

Hunter, P.J. and Pullan, A.J., *FEM/BEM Notes*, University of Auckland —
[figshare](https://auckland.figshare.com/articles/journal_contribution/FEM_BEM_Notes/5440000).

| Topic | Section |
| --- | --- |
| Interpolating a field with nodal parameters | §1.1–1.2 |
| Basis functions as weighting functions | §1.3 |
| Quadratic and higher-order Lagrange | §1.4 |
| Two- and three-dimensional tensor-product elements | §1.5 |
| Cubic Hermite, scale factors, the bicubic basis | §1.6 |
| The Galerkin weighted-residual method | §2.1, §2.3 |
| Gaussian quadrature | §2.8 |
