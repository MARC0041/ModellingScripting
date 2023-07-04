# %% 
import math
# from ModellingInPython import matplotlib.pyplot as plt
# from ModellingInPython import numpy as np
import numpy as np
import ipyvolume as ipv
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# /System/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python: No module named install
# export PATH="/Users/marcusc/opt/anaconda3/envs/python311/bin:$PATH"

def create_circle_vectors(radius = 1, num_points = 50, z_height = 0):
    vectors = []
    for i in range(num_points):
        theta = 2 * math.pi * float(i) / num_points
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        z = float(z_height)
        vectors.append((x,y,z))
    return vectors

# Create some 3D vectors
def visualise_vectors(vectors = [[0.0,0.0,0.0]], point_size = 1):
    fig = ipv.figure(width=400, height = 400)
    x = [v[0] for v in vectors]
    y = [v[1] for v in vectors]
    z = [v[2] for v in vectors]
    scatter = ipv.scatter(x,y,z, marker = 'sphere',size=point_size)
    ipv.show()

def test():
    print("hello world")
    return 2.5

# %% 
# v = create_circle_vectors(50, 1000)
# visualise_vectors(v)
# %%
