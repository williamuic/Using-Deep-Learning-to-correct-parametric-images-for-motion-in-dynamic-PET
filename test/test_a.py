import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from hdf5_processing_separate import create_dataloader

def denormalize(tensor, min_value, max_value):
    return ((tensor + 1) / 2 * (max_value - min_value)) + min_value

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataloader(opt.dataroot, opt.data_type, 'Ki', opt.batch_size, False, transform=None, global_norm=opt.global_norm, no_norm=opt.no_norm)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print(dataset.num_workers)

    # Create directory if it doesn't exist
    if not os.path.exists('test_results_a'):
        os.makedirs('test_results_a')

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader

        # Forward pass
        output = model.netG(model.real_A)  # real_A is the input image

        # Denormalize and save images locally
        real_A_denorm = denormalize(model.real_A, dataset.dataset.min_motion, dataset.dataset.max_motion)
        output_denorm = denormalize(output, dataset.dataset.min_motion_free, dataset.dataset.max_motion_free)
        real_B_denorm = denormalize(model.real_B, dataset.dataset.min_motion_free, dataset.dataset.max_motion_free)

        np.save(f'test_results_a/real_A_{i}.npy', real_A_denorm.cpu().numpy())
        np.save(f'test_results_a/fake_B_{i}.npy', output_denorm.detach().cpu().numpy())
        np.save(f'test_results_a/real_B_{i}.npy', real_B_denorm.cpu().numpy())
