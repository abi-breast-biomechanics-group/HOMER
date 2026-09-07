# API – mesh operations

Every function in these modules is a method of
[`MeshField`](mesh.md#HOMER.mesh.field.MeshField) that lives in its own
module: the first argument is the field itself, and `HOMER.mesh.field` binds
them into the class.  Call them as methods — `mesh.refine(2)`, not
`refinement.refine(mesh, 2)`.

`HOMER.mesh.reordering` is the exception: it is a plain function over a
field, because renumbering is something done *to* a mesh after a manipulation
rather than something a mesh does.  Call it as
`reorder_nodes(mesh)`.

::: HOMER.mesh.evaluation

---

::: HOMER.mesh.parameters

---

::: HOMER.mesh.topology

---

::: HOMER.mesh.refinement

---

::: HOMER.mesh.reordering

---

::: HOMER.mesh.plotting

---

::: HOMER.mesh.element_eval
