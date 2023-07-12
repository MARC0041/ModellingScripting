# %%
"""
Installation instructions:

# create a new conda env
    conda create -n pytorch3d python=3.9
    conda activate pytorch3d
    conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath

# make sure torch is installed. Use https://pytorch.org/ to figure out what command to use for your system
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    # DID NOT WORK
        even though I can import, import more from p3d did not work and the following issue was encountered
        https://github.com/facebookresearch/pytorch3d/issues/377

# install the pytorch3d module then close and reopen vs code-- VERSION 0.7.4
    pip install pytorch3d    
        DID NOT WORK:(probably because I am using a mac)
        conda install pytorch3d -c pytorch3d
        conda install pytorch3d -c pytorch3d-nightly

Useful links for installation:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
https://pytorch.org/

Useful examples:
https://pytorch3d.org/tutorials/
"""

import torch
import pytorch3d as p3d
# from pytorch3d.io import IO
# from pytorch3d.structures.meshes import Meshes
print(f"torch ver: {torch.__version__}")
print(f"p3d ver: {p3d.__version__}")




# %%
import torch
# from pytorch3d.io import IO
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import Translate
# from scipy import spatial
import numpy as np
import sys
import datetime

batch_size = 10

time_start = datetime.datetime.now()

def log(message):
   print(datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'), message)

if torch.cuda.is_available():
   device = torch.device("cuda:0")
   torch.cuda.set_device(device)
else:
   device = torch.device("cpu")

depth = 3
side = 1/(3**depth)
with_grad = False

# vertices
vertices_base = [
       [0,     0,      0],
       [side,  0,      0],
       [side,  side,   0],
       [0,     side,   0],
       [0,     0,      side],
       [side,  0,      side],
       [side,  side,   side],
       [0,     side,   side]
   ]
vertices = torch.tensor(
   vertices_base,
   device=device,
   dtype=torch.float32,
   requires_grad=with_grad,
)

# faces
#hull = spatial.ConvexHull(vertices_base)
faces_data = [
    [3, 2, 1],
    [3, 1, 0],
    [0, 1, 5],#
    [5, 4, 0],
    [7, 3, 0],
    [0, 4, 7],#
    [1, 2, 6],#
    [6, 5, 1],
    [2, 3, 6],#
    [3, 7, 6],#
    [4, 5, 6],#
    [6, 7, 4]
 ]
faces = torch.tensor(
   faces_data,
   device=device,
   dtype=torch.int64,
)

# holes
holes = []
side = 1
round_factor = 14
for d in range(1, depth+1):
   side /= 3
   for x in range(1,3**d,3):
       for y in range(1,3**d,3):
           holes.append({
               'x':[round(side*x,round_factor), round(side*x+side,round_factor)],
               'y':[round(side*y,round_factor), round(side*y+side,round_factor)]
               })

def voxel_in_hole(holes, x, y):
   comp_error = 0
   for hole in holes:
       if  x + comp_error >= hole['x'][0] and x - comp_error < hole['x'][1] and \
           y + comp_error >= hole['y'][0] and y - comp_error < hole['y'][1]:
           return True
   return False

# voxels
voxels = []
batch = 0
mesh = None
for x in range(3**depth):
   log('x '+ str(x)+' in '+str(3**depth))
   for y in range(3**depth):
       for z in range(3**depth):
           if  voxel_in_hole(holes, round(y/(3**depth),round_factor), round(z/(3**depth),round_factor)) or \
               voxel_in_hole(holes, round(x/(3**depth),round_factor), round(y/(3**depth),round_factor)) or \
               voxel_in_hole(holes, round(x/(3**depth),round_factor), round(z/(3**depth),round_factor)):
               continue
           voxel_translate = Translate(x/(3**depth), y/(3**depth), z/(3**depth), device=device)
           voxel_vertices_translated = voxel_translate.transform_points(vertices)
           voxels.append(Meshes(verts=[voxel_vertices_translated], faces=[faces]))
   batch += 1
   if batch > batch_size:
       if not mesh is None:
           log('append mesh')
           voxels.append(mesh)
       log('join')
       mesh = join_meshes_as_scene(voxels)
       voxels = []
       batch = 0


log('join')
if len(voxels):
   if not mesh is None:
       voxels.append(mesh)
   mesh = join_meshes_as_scene(voxels)
log('save')
# IO().save_mesh(mesh, 'pytorch3d_example_03.obj')
log('end')

time_end = datetime.datetime.now()
print('spent', (time_end - time_start).seconds, 'seconds')
# %%
# %%
