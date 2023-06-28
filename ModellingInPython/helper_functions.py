import math
# from ModellingInPython import matplotlib.pyplot as plt
import numpy as np
import ipyvolume as ipv

# /System/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python: No module named install
# export PATH="/Users/marcusc/opt/anaconda3/envs/python311/bin:$PATH"

def create_circle_vectors(radius = 5, num_points = 10):
    vectors = []
    for i in range(num_points):
        theta = 2 * math.pi * float(i) / num_points
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        vectors.append(x,y,0)
        


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Create some 3D vectors
def visualise_vectors(vectors = [[0,0,0]]):
    N = 1000
    x,y,z = np.random.normal(0,1,(3,N))
    fig = ipv.figure()
    scatter = ipv.scatter(x,y,z, marker = 'sphere')
    ipv.show()
def test_fn():
    print("hello world")
    return 2.5