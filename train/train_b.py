import time
import torch
import numpy as np
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from data_preprocessing.hdf5_processing_separate import create_dataloader
from torchvision import transforms
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    dataset = create_dataloader(opt.dataroot, opt.data_type, 'Vd', opt.batch_size, True, transform=None, global_norm=opt.global_norm, no_norm=opt.no_norm)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Create the generator model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations  
    # Define a loss function
    criterion = torch.nn.L1Loss()

    l1_losses = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        temp_losses = []

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing

            # Forward pass
            output = model.netG(model.real_A)  # real_A is the input image
            loss_G_L1 = criterion(output, model.real_B)  # real_B is the target image

            # Backward pass
            model.optimizer_G.zero_grad()  # set G's gradients to zero
            loss_G_L1.backward()  # calculate gradients for G
            model.optimizer_G.step()  # update G's weights

            if total_iters % opt.print_freq == 0:
                losses = {'G_L1': loss_G_L1.item()}
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                temp_losses.append(losses['G_L1'])
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        avg_l1_loss = sum(temp_losses) / len(temp_losses) if temp_losses else 0
        l1_losses.append(avg_l1_loss)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # Plot L1 loss values
    plt.plot(range(1, len(l1_losses) + 1), l1_losses, label='L1')
    plt.title('Average L1 Loss over epochs') 
    plt.xlabel('Epoch')
    plt.ylabel('Average L1 Loss')
    plt.legend()
    plt.savefig(f'{opt.checkpoints_dir}/{opt.name}/avg_l1_loss_plot.png')
    plt.show()
    plt.pause(5)
