#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py

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
os.chdir('/slms/inm/research/moco/William/Thesis/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/data_simulation/')

#%% 
emi_file_path =  'emi_data.mat'

with h5py.File(emi_file_path, 'r') as file:
    emi_data = file['emi'][:]

# %%
print(emi_data.shape)


### 4D attenuation image
# %%
atn_file_path =  'atn_data.mat'

with h5py.File(atn_file_path, 'r') as file:
    atn_data = file['atn'][:]
print(atn_data.shape)

#%%
emi_img = emi_data
atn_img = atn_data
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
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Centering the image for rotation
    center = np.array(img.shape) / 2
    offset = center - center @ R + translation
    
    # Apply affine transformation
    transformed_img = affine_transform(img, R, offset=offset, order=1, mode='constant', cval=np.min(img))
    
    return transformed_img
# def rigid_motion_3d(img, translation, rotation):
#     rotation = np.deg2rad(rotation)
#     cx, cy, cz = np.cos(rotation)
#     sx, sy, sz = np.sin(rotation)
    
#     Rx = np.array([[1, 0, 0, 0],
#                    [0, cx, -sx, 0],
#                    [0, sx, cx, 0],
#                    [0, 0, 0, 1]])
#     Ry = np.array([[cy, 0, sy, 0],
#                    [0, 1, 0, 0],
#                    [-sy, 0, cy, 0],
#                    [0, 0, 0, 1]])
#     Rz = np.array([[cz, -sz, 0, 0],
#                    [sz, cz, 0, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 1]])

#     R = Rx @ Ry @ Rz

#     T = np.eye(4)
#     T[:3, 3] = translation

#     transformation_matrix = R @ T

#     transformed_img = affine_transform(img, transformation_matrix[:3, :3], offset=-transformation_matrix[:3, 3],
#                                        order=1, mode='constant', cval=np.min(img))
#     return transformed_img
# def apply_random_transformations_3d(image_data):
#     num_frames, depth, height, width = image_data.shape
    
#     transformed_images = np.copy(image_data)
    
#     translation = np.random.uniform(-5, 5, size=3)  
#     rotation = np.random.uniform(-1, 1, size=3) 
#     random_frame = np.random.randint(num_frames)

#     transformed_images[random_frame] = rigid_motion_3d(image_data[random_frame], translation, rotation)
    
#     return transformed_images
def apply_random_transformations_3d(image_data):
    num_frames, depth, height, width = image_data.shape
    
    transformed_images = np.copy(image_data)
    
    for frame_idx in range(0, num_frames, 4):
        translation = np.random.uniform(-5, 5, size=3)  
        rotation = np.random.uniform(-1, 1, size=3)
        transformed_images[frame_idx] = rigid_motion_3d(image_data[frame_idx], translation, rotation)
    
    return transformed_images
def apply_specified_transformations_3d(image_data):
    num_frames, depth, height, width = image_data.shape
    
    translations = [-4, -2, 2, 4]
    rotations = [-1, 1]
    total_transforms = len(translations) * len(rotations)
    transformed_images = np.zeros((num_frames * total_transforms, depth, height, width))
    
    index = 0
    for frame_idx in range(0,num_frames,4):
        for translation in translations:
            for rotation in rotations:
                transformed_images[index] = rigid_motion_3d(image_data[frame_idx], [translation,translation,translation], [rotation,rotation,rotation])
                index += 1
                
    return transformed_images
#%%
motion_image_emi = apply_specified_transformations_3d(emi_img)
motion_image_atn = apply_specified_transformations_3d(atn_img)
#%%
ring_spacing = 0.40625
z_spacing = ring_spacing/2
num_rings = 64
num_planes = num_rings*2 - 1

#%%
for t in range(0, 16, 1):  # Show every 4th frame to see the motion
    plt.figure(figsize=(4, 4))
    plt.imshow(motion_image_emi[t, :, 128, :])
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

sirf_img.fill(atn_cm[0,:, :, :].astype(np.float32))
sirf_img.write('atn.hv')
#%%
### Save motion images in interfile format
motion_atn_cm = motion_image_atn * 10

for i in range(motion_image_emi.shape[0]):  
    sirf_img = pet.ImageData()
    sirf_img.initialise((num_planes, 256, 256), vsize=(2.031250, 2.03642, 2.03642))
    
    sirf_img.fill(motion_image_emi[i, :, :, :].astype(np.float32))
    
    sirf_img.write(f'dat_frame_motion{i+1}.hv')

    sirf_img.fill(motion_atn_cm[i,:, :, :].astype(np.float32))
    sirf_img.write(f'atn_motion{i+1}.hv')
#%%
import subprocess
pardir_result = subprocess.run('stir_config --examples-dir', shell=True, capture_output=True, text=True)

# Check if the command ran successfully
if pardir_result.returncode == 0:
    pardir = pardir_result.stdout.strip()
else:
    raise RuntimeError("Failed to get examples directory from stir_config")
#%%
for i in range(16):
    subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par dat_frame_{i+1}.nii dat_frame_{i+1}.hv', shell=True, capture_output=True, text=True)
    subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par dat_frame_motion{i+1}.nii dat_frame_motion{i+1}.hv', shell=True, capture_output=True, text=True)
# %%
subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par atn.nii atn.hv', shell=True, capture_output=True, text=True)
subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par atn_motion.nii atn_motion.hv', shell=True, capture_output=True, text=True)

# %%
import sirf.STIR
import scipy.io
import os
images = []
os.chdir('/slms/inm/research/moco/William/Thesis/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/data_simulation')
for i in range(64):
    img = sirf.STIR(f'frame_motion{i+1}.hv').as_array()
    images.append(img)
images_array = np.array(images)
scipy.io.savemat('frame_motion_images.mat', {'images': images_array})

# %%
