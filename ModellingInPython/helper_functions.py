# %% 
import math
import numpy as np
import ipyvolume as ipv
from geomdl import BSpline, utilities

# /System/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python: No module named install
# export PATH="/Users/marcusc/opt/anaconda3/envs/python311/bin:$PATH"

def create_circle_vectors(radius = 1, num_points = 50, z_height = 0):
    """
    Returns a list of vectors that form a circle
    Inputs:
        radius: float
        num_points: int
    """
    vectors = []
    for i in range(num_points):
        theta = 2 * math.pi * float(i) / num_points
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        z = float(z_height)
        vectors.append((x,y,z))
    return vectors
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
def rotate_vector(vector, axis, angle):
    """Rotate a vector around an axis by some angle
    Inputs:
        vector: List<float, float, float>
        axis: <float, float, float>
        angle: float (in degrees)"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
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
def get_spline_vectors_from_controlpts(controlpts, sample_count=10):
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
    crv.degree = max(1, min(3, len(controlpts)-1)) # if len(cp) <=3, degree must be less than that. but always >=1
    # Set control points
    crv.ctrlpts = controlpts
    # Auto−generate the knot vector
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    # specify evalutation delta
    crv.sample_size = sample_count
    # Evaluate the curve - evalpts calls crv.evaluate() by default 
    points = crv.evalpts
    return points

    
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


    pass
def sweep_surface():
    return



# %% Tests
def test():
    print("hello world")
    return 2.5
def example1_circle():
    v = create_circle_vectors(50, 50, 0)
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
    a = np.arange(-5, 5, 0.2)
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
    cps = [[10,5,10], [10,20,-30], [40,10,25], [-10,5,0]]
    spline_vectors = get_spline_vectors_from_controlpts(cps, 40)
    visualise_vectors(spline_vectors)
# %%
example5_controlpoints()
