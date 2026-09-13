from HOMER import cube


mesh = cube()
mesh.refine([3,1,2])
mesh.plot(labels=True)
