# %%
import stl
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from myplot import plot_verticles

sphere = mesh.Mesh.from_file("/Users/marcusc/Downloads/_3DBenchy_-_The_jolly_3D_printing_torture-test_by_CreativeTools_se_763622/files/3DBenchy.stl")
print(f"vectors: {sphere.vectors}")
# create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(sphere.vectors))
                      
scale = sphere.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()
# %%
