# %% 
import math
import numpy as np
import ipyvolume as ipv
from geomdl import BSpline, utilities
from matplotlib import cm

# /System/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python: No module named install
# export PATH="/Users/marcusc/opt/anaconda3/envs/python311/bin:$PATH"

# Links: used to get spline: https://github.com/orbingol/geomdl-examples/tree/master/surface

def get_circle_vectors(radius = 1, num_points = 50, z_height = 0):
    """
    Returns a list of vectors that form a circle

    Inputs:
        radius: float
        num_points: int
    """
    vectors = []
    for i in range(num_points+1):
        theta = 2 * math.pi * float(i) / num_points
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        z = float(z_height)
        vectors.append((x,y,z))
    return np.array(vectors)
def visualise_vectors(vectors = [[0.0,0.0,0.0]], point_size = 1):
    """Visualises the vectors inputted in a notebook
    Takes in a list of vectors and a point_size for each vector

    Inputs: 
        vectors: List<List<float, float, float>> 
        point_size: float
    """
    fig = ipv.figure(width=400, height = 400)
    x = [v[0] for v in vectors]
    y = [v[1] for v in vectors]
    z = [v[2] for v in vectors]
    scatter = ipv.scatter(x,y,z, marker = 'sphere',size=point_size)
    ipv.xlim(min(x), max(x))
    ipv.ylim(min(z), max(z))
    ipv.zlim(min(y), max(y))
    ipv.squarelim()
    ipv.show()
def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """Finds angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)
def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)
def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)
def rotate_vector(vector, axis, angle, deg = True):
    """Rotate a vector around an axis by some angle

    Inputs:
        vector: List<float, float, float>
        axis: <float, float, float>
        angle: float (in degrees)
        deg: bool (if the angle is in degree; if the angle is in radians, use False)"""
    # Convert angle to radians
    angle_rad = np.radians(angle) if deg else angle
    # rot_vector = angle_rad * np.array(axis)
    # rotation = R.from_rotvec(rot_vector)
    # rotated_vector = rotation.apply(vector)

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the rotation matrix
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation_matrix = np.array([[c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                                [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                                [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]])
    
    # Apply the rotation to the vector
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector
def rotate_vectors(vectors, axis, angle, deg = True):
    """Rotate a list of vectors around an axis by some angle. 
    All vectors in the list will be rotated by the same rotation matrix.

    Inputs:
        vector: List<float, float, float>
        axis: <float, float, float>
        angle: float (in degrees)"""
    # Convert angle to radians
    angle_rad = np.radians(angle) if deg else angle
    # rot_vector = angle_rad * np.array(axis)
    # rotation = R.from_rotvec(rot_vector)
    # rotated_vector = rotation.apply(vector)

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the rotation matrix
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation_matrix = np.array([[c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                                [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                                [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]])
    
    # Apply the rotation to the vector
    rotated_vector = np.dot(rotation_matrix, vectors.T).T
    return rotated_vector
def get_spline_vectors_from_controlpts(controlpts, sample_count=10, degree = 3):
    """
    Returns a list of points that follows the derived curve using bspline

    Inputs: 
        controlpts: list of control points in a list. e.g. [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        sample_count: length of the returned vector list
    Outputs:
        list of vectors that follows the derived bspline curve
    """
    crv = BSpline.Curve()
    # Set curve degree
    crv.degree = max(1, min(degree, len(controlpts)-1)) # if len(cp) <=3, degree must be less than that. but always >=1
    # Set control points
    crv.ctrlpts = controlpts
    # Auto−generate the knot vector
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    # specify evalutation delta
    crv.sample_size = sample_count
    # Evaluate the curve - evalpts calls crv.evaluate() by default 
    points = crv.evalpts
    return points
def sweep_surface(cross_section_vectors, spline_vectors):
    """
    Sweeps the cross_section_vectors along the direction of the spline_vectors
    Returns a list of cross sectional vectors rotated to follow the curvature of the spline

    Inputs:
        cross_section_vectors: List<List<float, float, float>>
        spline_vectors: List<List<float, float, float>>
    Outputs:
        List<List<List<float, float, float>>>
    """
    layers = []
    z_vector = np.array([0,0,1])
    for i in range(len(spline_vectors)-1):
        dir_vec = np.array([spline_vectors[i+1][0] - spline_vectors[i][0],
                   spline_vectors[i+1][1] - spline_vectors[i][1], 
                   spline_vectors[i+1][2] - spline_vectors[i][2] ])
        angle_rad = angle_between(dir_vec, z_vector)
        # print(f"shape x: {z_vector.reshape(1, -1)}")
        # print(f"shape v: {dir_vec.reshape(-1, 1)}")
        rotated_vectors = rotate_vectors(cross_section_vectors, np.cross([0,0,1], dir_vec), angle_rad, False)
        translated_vectors = rotated_vectors + spline_vectors[i]
        layers.append(translated_vectors)

    return np.array(layers)
def visualise_surface(layers, color = False):
    """
    Takes in a list of layers, 
    a layer is defined as a list of vectors in that particular layer
    visualises the surface based on the layers given
    """
    X = np.array([[layers[layer][theta][0] for theta in range(len(layers[0]))] for layer in range(len(layers))])
    Y = np.array([[layers[layer][theta][1] for theta in range(len(layers[0]))] for layer in range(len(layers))])
    Z = np.array([[layers[layer][theta][2] for theta in range(len(layers[0]))] for layer in range(len(layers))])
    ipv.figure()
    if color:
        colormap = cm.coolwarm
        znorm = np.array(Z - Z.min())
        color = colormap(np.array(znorm)/znorm.ptp())
        ipv.plot_surface(X, Z, Y, color=color[..., :3])
    else:
        ipv.plot_surface(X, Z, Y)   


    ipv.xlim(X.min(), X.max())
    ipv.ylim(Z.min(), Z.max())
    ipv.zlim(Y.min(), Y.max())
    ipv.squarelim()
    ipv.show()

# to be implemented
"""
1. Interpolation
2. Transition from circle to square
3. rotate a square in the z axis
4. import an stl and perform transformations on the object
"""
def connected_points():
    node1 = np.array([0.0,0.0,0.0])
    node2 = np.array([1.0,1.0,1.0])
    node3 = np.array([0.0,1.0,1.0])
    node4 = np.array([1.0,0.0,1.0])
    fig = ipv.figure()
    scatter = ipv.scatter([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], marker='sphere', size=5)
    line = ipv.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='blue')
    line = ipv.plot([node1[0], node3[0]], [node1[1], node3[1]], [node1[2], node3[2]], color='blue')
    line = ipv.plot([node1[0], node4[0]], [node1[1], node4[1]], [node1[2], node4[2]], color='blue')
    ipv.show()
    return


# %% Tests
def test():
    print("hello world")
    return 2.5
def example1_circle():
    v = get_circle_vectors(50, 50, 0)
    v1 = v.copy()
    for i in range(len(v)):
        v1.append(rotate_vector(v[i], [0,1,0], 180))
        v1.append(rotate_vector(v[i], [0,1,0], 165))
        v1.append(rotate_vector(v[i], [0,1,0], 150))
        v1.append(rotate_vector(v[i], [0,1,0], 135))
        v1.append(rotate_vector(v[i], [0,1,0], 120))
        v1.append(rotate_vector(v[i], [0,1,0], 105))
        v1.append(rotate_vector(v[i], [0,1,0], 90))
        v1.append(rotate_vector(v[i], [0,1,0], 75))
        v1.append(rotate_vector(v[i], [0,1,0], 60))
        v1.append(rotate_vector(v[i], [0,1,0], 45))
        v1.append(rotate_vector(v[i], [0,1,0], 30))
        v1.append(rotate_vector(v[i], [0,1,0], 15))
        # v[i] = y_rotation(v[i], math.pi/4)
    visualise_vectors(v1)
def example2_surface():
    s = 1/2**0.5
    # 4 vertices for the tetrahedron
    x = np.array([1.,  -1, 0,  0])
    y = np.array([0,   0, 1., -1])
    z = np.array([-s, -s, s,  s])
    # and 4 surfaces (triangles), where the number refer to the vertex index
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1,3,2)]
    ipv.figure()
    # we draw the tetrahedron
    mesh = ipv.plot_trisurf(x, y, z, triangles=triangles, color='orange')
    # and also mark the vertices
    ipv.scatter(x, y, z, marker='sphere', color='blue')
    ipv.xyzlim(-2, 2)
    ipv.show()
def example3_surface():
    a = np.arange(-5, 5, 1)
    U, V = np.meshgrid(a, a)
    X = U
    Y = V
    Z = X*Y**2

    from matplotlib import cm
    colormap = cm.coolwarm
    znorm = Z - Z.min()
    color = colormap(np.array(znorm)/znorm.ptp())

    ipv.figure()
    ipv.plot_surface(X, Z, Y, color=color[..., :3])
    # ipv.plot_wireframe(X, Z, Y, color="red")
    ipv.show()
def example4_animation():
    # create 2d grids: x, y, and r
    u = np.linspace(-10, 10, 25)
    x, y = np.meshgrid(u, u)
    r = np.sqrt(x**2+y**2)
    print("x,y and z are of shape", x.shape)
    # and turn them into 1d
    x = x.flatten()
    y = y.flatten()
    r = r.flatten()
    print("and flattened of shape", x.shape)
    # create a sequence of 15 time elements
    time = np.linspace(0, np.pi*2, 15)
    z = np.array([(np.cos(r + t) * np.exp(-r/5)) for t in time])
    print("z is of shape", z.shape)
    ipv.figure()
    s = ipv.scatter(x, z, y, marker="sphere")
    ipv.animation_control(s, interval=200)
    ipv.ylim(-3,3)
    ipv.show()
    return
def example5_controlpoints():
    cps = [[10,5,-10], [10,20,-30], [40,10,25], [-10,5,0]]
    spline_vectors = get_spline_vectors_from_controlpts(cps, 40,2)
    visualise_vectors(spline_vectors)
def example6_rotatecirclevectors():
    vectors = get_circle_vectors(5)
    v_rotated = rotate_vectors(vectors, [1,0,0], 45)
    vectors += v_rotated
    visualise_vectors(vectors)
def example7_sweep():
    circle = get_circle_vectors(5,20, 0)
    spline = get_spline_vectors_from_controlpts([[0,0,0], [1,1,1]], 10)
    vectors = sweep_surface(circle, spline)
    print(f"shape: {vectors.shape}")
    visualise_vectors(vectors.reshape(-1, vectors.shape[-1]))
    return
def example8_sweep():
    circle = get_circle_vectors(5,40, 0)
    cps = [[10,5,-10], [10,20,-30], [40,10,25], [-10,5,0]]
    spline = get_spline_vectors_from_controlpts(cps, 40,2)
    vectors = sweep_surface(circle, spline)
    visualise_vectors(vectors.reshape(-1, vectors.shape[-1]))
    return
def example9_sweepsurface():

    circle = get_circle_vectors(5,20, 0)
    spline = get_spline_vectors_from_controlpts([[0,0,0], [1,1,1]], 10)
    vectors = sweep_surface(circle, spline)
    # flat_vectors = vectors.reshape(-1, vectors.shape[-1])
    # X = np.array([v[0] for v in vectors])
    # Y = np.array([v[1] for v in vectors])
    # Z = np.array([v[2] for v in vectors])
    X = np.array([[vectors[layer][theta][0] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    Y = np.array([[vectors[layer][theta][1] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    Z = np.array([[vectors[layer][theta][2] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    # print(f"shape: {vectors.shape}")
    from matplotlib import cm
    colormap = cm.coolwarm
    znorm = np.array(Z - Z.min())
    color = colormap(np.array(znorm)/znorm.ptp())

    ipv.figure()
    ipv.plot_surface(X, Z, Y, color=color[..., :3])
    ipv.show()
    return
def example10_pipe():
    # Define the theta and z ranges
    theta = np.linspace(0, 2*np.pi, 10)
    z = np.linspace(-1, 1, 50)

    # Create the meshgrid
    Theta, Z = np.meshgrid(theta, z)

    # Define the surface coordinates
    X = np.cos(Theta)
    Y = np.sin(Theta)

    # Create the 3D surface plot
    ipv.figure()
    ipv.plot_surface(X, Y, Z, color="blue")
    ipv.xlabel('X')
    ipv.ylabel('Y')
    ipv.zlabel('Z')
    ipv.show()
    return
def example11_sweepsurface():
    circle = get_circle_vectors(5,40, 0)
    cps = [[10,-5,-10], [10,20,-15], [20,10,25], [-10,5,0]]
    spline = get_spline_vectors_from_controlpts(cps, 40,4)
    vectors = sweep_surface(circle, spline)
    visualise_surface(vectors, True)
    # X = np.array([[vectors[layer][theta][0] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    # Y = np.array([[vectors[layer][theta][1] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    # Z = np.array([[vectors[layer][theta][2] for theta in range(len(vectors[0]))] for layer in range(len(vectors))])
    # ipv.figure()
    # ipv.plot_surface(X, Z, Y)
    # ipv.show()
    return
def example12_matplotlib_visualise():
    # Define the theta and z ranges
    theta = np.linspace(0, 2*np.pi, 10)
    z = np.linspace(-1, 1, 50)

    # Create the meshgrid
    Theta, Z = np.meshgrid(theta, z)

    # Define the surface coordinates
    X = np.cos(Theta)
    Y = np.sin(Theta)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
    return

# %%
# example3_surface()
# example6_rotatecirclevectors()
# example7_sweep()
# example8_sweep()
# example9_sweepsurface()
# example10_pipe()
# example11_sweepsurface()
# example1_circle()
example12_matplotlib_visualise()

# %%

# Functions needed
"""
1. Create surface from vertices
2. iteration loop optimisation - lattices? 
3. image operations
"""

"""
Pytorch implementations
1. possible use of KNNs to detect features in the mesh

pygem free form deformation from stl
    https://notebook.community/fsalmoir/PyGeM/tutorials/tutorial-1-stl
Visualisation in notebooks: 3D Visualization of STL Files with Python & VTK
    https://levelup.gitconnected.com/3d-visualization-of-stl-files-with-python-vtk-8c1f284c7f6e
"""
# %%
