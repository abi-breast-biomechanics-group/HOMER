from HOMER import mesh, mesh_node, mesh_element, H3Basis, L1Basis, L3Basis, L2Basis
import numpy as np
import pyvista as pv
import jax


point0 = mesh_node(loc=([0,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([2,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0], id=1)
point1 = mesh_node(loc=([0,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point2 = mesh_node(loc=([0,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([0,0,0]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point3 = mesh_node(loc=([0,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([2,0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point4 = mesh_node(loc=([1,0,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point5 = mesh_node(loc=([1,0,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,-0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point6 = mesh_node(loc=([1,1,1]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5, 0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])
point7 = mesh_node(loc=([1,1,0]), du=[0,0,0], dv=[0,0,0], dw = ([1,0.5,-0.5]), dudv=[0,0,0], dudw=[0,0,0], dvdw=[0,0,0], dudvdw=[0,0,0])


element1 = mesh_element(node_indexes=[0,1,2,3,4,5,6,7], basis_functions=(H3Basis, H3Basis, H3Basis))
objMesh = mesh(nodes = [point0, point1, point2, point3, point4, point5, point6, point7], elements = element1)
objMesh.refine(2)
objMesh.plot(node_colour='r')
