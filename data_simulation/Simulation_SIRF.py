#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt

#%%
from numpy.linalg import norm
import matplotlib.animation as animation
import os
import sys
import shutil
#import scipy
#from scipy import optimize
import sirf.STIR as pet
### 4D Emission image
#%% 
emi_file_path =  '/slms/inm/research/moco/William/Thesis/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/data_simulation/img_dat_4D.mat'

with h5py.File(emi_file_path, 'r') as file:
    emi_data = file['emi'][:]

# %%
emi_data.shape


# %%
emi_sliced_data = emi_data[150:290] 
emi_transposed_data = np.transpose(emi_sliced_data, (1, 0, 2, 3))
#%%
emi_img = emi_transposed_data
# %%
plt.imshow(emi_img[0,:,:,128])
# %%
plt.imshow(emi_img[10,:,128,:])

### 4D attenuation image
# %%
atn_file_path =  '/slms/inm/research/moco/William/Thesis/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/data_simulation/img_atn_4D.mat'

with h5py.File(atn_file_path, 'r') as file:
    atn_data = file['atn'][:]
    atn_sliced_data = atn_data[150:290]  
atn_transposed_data = np.transpose(atn_sliced_data, (1, 0, 2, 3))
#%%
atn_img = atn_transposed_data
# %%
plt.imshow(atn_img[0,:,:,128])
# %%
plt.imshow(atn_img[10,:,128,:])


### Rigid motion and respiratory motion
# %%
from scipy.ndimage import affine_transform
#%%
def rigid_motion(img, translation,rotation):
    r = np.deg2rad(rotation)
    affine_matrix = np.array([[np.cos(r), -np.sin(r), translation[0]],
        [np.sin(r), np.cos(r), translation[1]],
        [0, 0, 1]])

    rigid_motion_image = affine_transform(img, affine_matrix[:2, :2], offset=-affine_matrix[:2, 2])
    return rigid_motion_image

def rigid_motion_3d(img, translation, rotation):
    rotation = np.deg2rad(rotation)
    cx, cy, cz = np.cos(rotation)
    sx, sy, sz = np.sin(rotation)
    
    Rx = np.array([[1, 0, 0, 0],
                   [0, cx, -sx, 0],
                   [0, sx, cx, 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[cy, 0, sy, 0],
                   [0, 1, 0, 0],
                   [-sy, 0, cy, 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[cz, -sz, 0, 0],
                   [sz, cz, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    R = Rx @ Ry @ Rz

    T = np.eye(4)
    T[:3, 3] = translation

    transformation_matrix = R @ T

    transformed_img = affine_transform(img, transformation_matrix[:3, :3], offset=-transformation_matrix[:3, 3],
                                       order=1, mode='constant', cval=np.min(img))
    return transformed_img
def apply_random_transformations_3d(image_data):
    num_frames, depth, height, width = image_data.shape
    
    transformed_images = np.zeros_like(image_data)
    
    for frame in range(num_frames):
        translation = np.random.uniform(-5, 5, size=3)  
        rotation = np.random.uniform(-1, 1, size=3) 
        
        transformed_images[frame] = rigid_motion_3d(image_data[frame], translation, rotation)
    
    return transformed_images

#%%
motion_image_emi = apply_random_transformations_3d(emi_img)
motion_image_atn = apply_random_transformations_3d(atn_img)
#%%
ring_spacing = 0.40625
z_spacing = ring_spacing/2
num_rings = 64
num_planes = num_rings*2 - 1
motion_image_emi = motion_image_emi[:,motion_image_emi.shape[1]-num_planes:,:,:]
motion_image_atn = motion_image_atn[:,motion_image_atn.shape[1]-num_planes:,:,:]
#%%
emi_img = emi_img[:,emi_img.shape[1]-num_planes:,:,:]
atn_img = atn_img[:,atn_img.shape[1]-num_planes:,:,:]
#%%
for t in range(0, 16, 1):  # Show every 4th frame to see the motion
    plt.figure(figsize=(4, 4))
    plt.imshow(motion_image_emi[t, 8, :, :])
    plt.title(f"Time Frame {t}")
    plt.show()
#%%
for t in range(0, 16, 1):  # Show every 4th frame to see the motion
    plt.figure(figsize=(4, 4))
    plt.imshow(emi_img[t, :, 128, :])
    plt.title(f"Time Frame {t}")
    plt.show()
### Save ground truth in interfile format
# %%
atn_cm = atn_img * 10

for i in range(emi_img.shape[0]):  
    sirf_img = pet.ImageData()
    sirf_img.initialise((num_planes, 256, 256), vsize=(2.031250, 2.03642, 2.03642))
    
    sirf_img.fill(emi_img[i, :, :, :].astype(np.float32))
    
    sirf_img.write(f'dat_frame_{i+1}.hv')

sirf_img.fill(atn_cm[0, :, :].astype(np.float32))
sirf_img.write('atn.hv')

### Save motion images in interfile format
motion_atn_cm = motion_image_atn * 10

for i in range(motion_image_emi.shape[0]):  
    sirf_img = pet.ImageData()
    sirf_img.initialise((num_planes, 256, 256), vsize=(2.031250, 2.03642, 2.03642))
    
    sirf_img.fill(motion_image_emi[i, :, :, :].astype(np.float32))
    
    sirf_img.write(f'dat_frame_motion{i+1}.hv')

sirf_img.fill(motion_atn_cm[0, :, :].astype(np.float32))
sirf_img.write('atn_motion.hv')


