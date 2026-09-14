---
title: 'HOMER: High Order MEsh Representations'
tags:
  - Python
  - Biomedical Engineering
  - modelling
  - optimisation
  - JAX
  - automatic differentiation
  - finite element
  - high order meshes
  - mesh fitting
authors:
  - name: Robin Laven
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: George Harpur
    affiliation: 1
  - name: Matthew French
    affiliation: 1
  - name: Gonzalo Maso Talou
    affiliation: 1
  - name: Thiranja Prasad Babarenda Gamage
    affiliation: 1
  - name: Martyn P. Nash
    affiliation: "1 2"
  - name: Poul M.F. Nielsen
    affiliation: "1 2"
affiliations:
 - name: Auckland BioEngineering Institute, New Zealand
   index: 1
 - name: The Deparment of Engineering Science and Biomedical Engineering, New Zealand
   index: 2
date: 14 September 2026
bibliography: paper.bib

---

# Summary

Mathematical models of objects need to match both the shape and motion of their physical counterparts.
Ensuring that this is correct typically involves large optimisations that can have potentially millions of datapoints and hundreds of thousands of parameters.
Without properly setting up these optimisation problems, you can't do them.
HOMER provides a JAX-based representation of these mathetmatical models designed to handle the careful book-keeping necessary for solving these problems.
It contains representations of basis functions and jacobian utilities designed to make it easy to write arbitrary optimisation functions that solve quickly and accurately.


# Statement of need

High order meshes are useful tools for modelling and understanding the geometry of the world around us.
They are particularly useful for describing and understanding biological tissues.
However, writing and manipulating these meshes can be time consuming.
Many old tools are currently out of date.


# State of the field

Converting measuremets to objects is a classic problem in computer-science and medical imaging.
As such, it has classical solutions: marching cubes, GMESH, which are often fed into solvers such as Fenicsx and Fembio.
Autodifferention is a very powerful tool for enabling analysis of modelled systems.
This has outputs that are very useful for a range of problems especially adjoint problems which require optimisation problem gradients.

There has been a reneissance of autodifferentiated, JAX compatible and otherwise, tools for the solution of FEM problems.
However, HOMER focusses on the topology of the mesh and how it changes, rather than solving equations over the mesh.
(Technically, you can express and solve these equations in HOMER, but there are no simple primitives.)
This gives HOMER it's niche, a shape representation toolbox that makes it easier to write an as yet unknown loss function.


# Software Design

HOMER's fundamental goal is to make it easy to express arbitrary optimisation problems while using high-order basis elements, and this informs its software design.
Implementing a correct Jacobian for an arbitrary problem is very complicated, placing a large burden on the implementer to ensure mathematical and logical consistency.
Many bugs may quietly degrade optimisation performance without causing explicit warnings, making this hard to debug, and large jacobians are hard to verify numerically.
Autodifferentiation via JAX solves this issue, but there is a conflict between the functional nature of JAX and the representation of physical objects.

HOMER aims to solve this by
(1) turning arbitrary basis function and mesh topologies into JAX functions,
(2) providing a way to turn a mesh into a function with a clear input (fit_params), and
(3) implementing good defaults for useful but complicated tools, such as point projection and sparse-Jacobian handling.
Together, these tools allow HOMER to present an interface that allows extremely expressive parameterisation of loss functions.
The interface does require that the user follows JAX's functional conventions for array manipulation, but in exchange allows for fast optimisations over a wide range of problems.


# Research Impact

Although it was originally written to support the skin tissue deformation analysis in the maintainer's thesis, the flexible nature of HOMER's represntations has allowed it to be used on a variety of projects.
It provides the fitting and optimisation routines used by Mobstr3D to analyse DENSE CMR imaging.



# AI usage disclosure

AI tools have been used to document, test, and refactor HOMER.
Generative models were also used to optimise some functions, e.g. mesh.embed_points() and
suggested some algorithmic tools, such as the approximate mesh-free jacobian for use with
lsmr non-linear least-squares solvers.
The majority of this used Claude Opus 5.1 via the scientific plan.
All documentation generated was manually reviewed and updated. 
While much of the testing infrastructure has been generated with AI tools, a subset of tests are human-curated integration tests that verify that HOMER is achieving its global goals and aims.

No AI was used to edit, draft, or review this paper.md.

# Acknowledgements

# References
