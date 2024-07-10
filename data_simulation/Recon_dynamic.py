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
import subprocess
#%%
os.chdir('/SAN/inm/moco/William/Thesis/Using-Deep-Learning-to-correct-parametric-images-for-motion-in-dynamic-PET/data_simulation/')
template=pet.AcquisitionData('Siemens mMR',span=11, max_ring_diff=60)
template.write('template_sinogram_mMR.hs')
#%%
# attn_image = pet.ImageData('atn.hv')
# #%% save max for future displays
# for i in range(16):
#     image = pet.ImageData(f'dat_frame_{i+1}.hv')
#     cmax = image.max()*.6

#     acq_model_for_attn = pet.AcquisitionModelUsingRayTracingMatrix()
#     asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model_for_attn)
#     asm_attn.set_up(template)
#     attn_factors = asm_attn.forward(template.get_uniform_copy(1))
#     asm_attn = pet.AcquisitionSensitivityModel(attn_factors)
#     # create acquisition model
#     # if True:
#     acq_model = pet.AcquisitionModelUsingParallelproj()
#     # else:
#     # acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
#     #     # we will increase the number of rays used for every Line-of-Response (LOR) as an example
#     #     # (it is not required for the exercise of course)
#     # acq_model.set_num_tangential_LORs(10)
#     acq_model.set_acquisition_sensitivity(asm_attn)
#     # set-up
#     acq_model.set_up(template,image)
#     test_fwd = acq_model.forward(image)
#     background_term = test_fwd.get_uniform_copy(test_fwd.max()*.05)
#     acq_model.set_background_term(background_term)
#     # create reconstructor
#     obj_fun = pet.make_Poisson_loglikelihood(test_fwd)
#     obj_fun.set_acquisition_model(acq_model)
#     recon = pet.OSMAPOSLReconstructor()
#     recon.set_objective_function(obj_fun)
#     recon.set_num_subsets(21)
#     recon.set_num_subiterations(63)
#     initial_image=image.get_uniform_copy(cmax / 4)
#     # initialisation and reconstruction
#     recon.set_current_estimate(initial_image)
#     recon.set_up(initial_image)
#     recon.process()
#     reconstructed_image=recon.get_output()
#     reconstructed_image.write(f'frame{i+1}.hv')
#%% 
attn_image = pet.ImageData('atn_motion.hv')
for i in range(16):
    image = pet.ImageData(f'dat_frame_motion{i+1}.hv')
    cmax = image.max()*.6

    acq_model_for_attn = pet.AcquisitionModelUsingRayTracingMatrix()
    asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model_for_attn)
    asm_attn.set_up(template)
    attn_factors = asm_attn.forward(template.get_uniform_copy(1))
    asm_attn = pet.AcquisitionSensitivityModel(attn_factors)
    # create acquisition model
    # if True:
    acq_model = pet.AcquisitionModelUsingParallelproj()
    # else:
    # acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    #     # we will increase the number of rays used for every Line-of-Response (LOR) as an example
    #     # (it is not required for the exercise of course)
    # acq_model.set_num_tangential_LORs(10)
    acq_model.set_acquisition_sensitivity(asm_attn)
    # set-up
    acq_model.set_up(template,image)
    test_fwd = acq_model.forward(image)
    background_term = test_fwd.get_uniform_copy(test_fwd.max()*.05)
    acq_model.set_background_term(background_term)
    # create reconstructor
    obj_fun = pet.make_Poisson_loglikelihood(test_fwd)
    obj_fun.set_acquisition_model(acq_model)
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(21)
    recon.set_num_subiterations(63)
    initial_image=image.get_uniform_copy(cmax / 4)
    # initialisation and reconstruction
    recon.set_current_estimate(initial_image)
    recon.set_up(initial_image)
    recon.process()
    reconstructed_image=recon.get_output()
    reconstructed_image.write(f'frame_motion{i+1}.hv')
    

#%%
# img = pet.ImageData('frame5.nii')
# mo_img = pet.ImageData('frame_motion1.hv')
# plt.imshow(img.as_array()[64,:,:])

# # # %%
# # plt.imshow(mo_img.as_array()[:,120,:])
# # # %%
# # plt.imshow(img.as_array()[:,120,:])

# # %%
# pardir_result = subprocess.run('stir_config --examples-dir', shell=True, capture_output=True, text=True)

# # Check if the command ran successfully
# if pardir_result.returncode == 0:
#     pardir = pardir_result.stdout.strip()
# else:
#     raise RuntimeError("Failed to get examples directory from stir_config")
# #%%
# for i in range(16):
#     subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par frame{i+1}.nii frame{i+1}.hv', shell=True, capture_output=True, text=True)
#     subprocess.run(f'stir_math --output-format {pardir}/samples/stir_math_ITK_output_file_format.par frame_motion{i+1}.nii frame_motion{i+1}.hv', shell=True, capture_output=True, text=True)

# # %%
# img = pet.ImageData('frame1.nii')

# # %%
# print(img.shape)
# #%%
# plt.imshow(img.as_array()[:,128,:])
# # %%
# images = []
# for i in range(16):
#     img = pet.ImageData(f'frame{i+1}.nii').as_array()
#     images.append(img)
# images = np.stack(images)
# # %%
# from scipy.io import savemat
# images_dict = {'img':images}
# savemat('images.mat',images_dict)
# # %%
# images = []
# for i in range(16):
#     img = pet.ImageData(f'frame_motion{i+1}.nii').as_array()
#     images.append(img)
# images = np.stack(images)
# # %%
# from scipy.io import savemat
# images_dict = {'img':images}
# savemat('images_motion.mat',images_dict)
# %%
