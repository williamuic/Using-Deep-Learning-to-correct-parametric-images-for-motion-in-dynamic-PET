# evaluation_metrics.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
import os

class EvaluationMetrics:
    def __init__(self, folder_path, num_images, pairs, channels, y_start=64, y_end=192, x_start=0, x_end=256):
        self.folder_path = folder_path
        self.num_images = num_images
        self.pairs = pairs
        self.channels = channels
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end

    def plot_image(self, img, title):
        fig, axs = plt.subplots(1, img.shape[0], figsize=(12,12))
        for i in range(img.shape[0]):
            im = axs[i].imshow(img[i])
            axs[i].set_title(f'{title} - Channel {i+1}')
            axs[i].axis('off')

            # create an axes on the right side of axs[i]. The width of cax will be 5% of axs[i] and the padding between cax and axs[i] will be fixed at 0.05 inch.
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar(im, cax=cax)
        plt.show()

    def view_images(self, index):
        for name in self.pairs:
            file_path = os.path.join(self.folder_path, f"{name}{index}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                self.plot_image(data.squeeze(), f"{name}{index}")
            else:
                print(f"The file {file_path} does not exist.")

    def calculate_metrics(self, real_img, fake_img):
        real_img = real_img.squeeze()
        fake_img = fake_img.squeeze()

        metrics = {
            "Ki": {},
            "Vd": {}
        }

        if real_img.ndim == 2:  # If the image only has one channel
            real_channel = real_img[self.y_start:self.y_end, self.x_start:self.x_end]
            fake_channel = fake_img[self.y_start:self.y_end, self.x_start:self.x_end]

            rmse_value = np.sqrt(mse(real_channel, fake_channel))
            nmse_value = mse(real_channel, fake_channel) / np.var(real_channel)
            range_value = real_channel.max() - real_channel.min()
            nrmse_value = rmse_value / range_value
            ssim_value = ssim(real_channel, fake_channel, data_range=fake_channel.max() - fake_channel.min())
            psnr_value = psnr(real_channel, fake_channel, data_range=fake_channel.max() - fake_channel.min())

            channel_name = self.channels[0]
            metrics[channel_name] = {
                "RMSE": rmse_value,
                "NMSE": nmse_value,
                "NRMSE": nrmse_value,
                "SSIM": ssim_value,
                "PSNR": psnr_value,
            }
        else:  # If the image has multiple channels
            for i in range(real_img.shape[0]):
                real_channel = real_img[i, self.y_start:self.y_end, self.x_start:self.x_end]
                fake_channel = fake_img[i, self.y_start:self.y_end, self.x_start:self.x_end]

                rmse_value = np.sqrt(mse(real_channel, fake_channel))
                nmse_value = mse(real_channel, fake_channel) / np.var(real_channel)
                range_value = real_channel.max() - real_channel.min()
                nrmse_value = rmse_value / range_value
                ssim_value = ssim(real_channel, fake_channel, data_range=fake_channel.max() - fake_channel.min())
                psnr_value = psnr(real_channel, fake_channel, data_range=fake_channel.max() - fake_channel.min())

                channel_name = "Ki" if i == 0 else "Vd"
                metrics[channel_name] = {
                    "RMSE": rmse_value,
                    "NMSE": nmse_value,
                    "NRMSE": nrmse_value,
                    "SSIM": ssim_value,
                    "PSNR": psnr_value,
                }

        return metrics


    def calculate_average_metrics(self):
        metrics_totals = {
            pair: {
                "Ki": {"RMSE": 0, "NMSE": 0, "NRMSE": 0, "SSIM": 0, "PSNR": 0},
                "Vd": {"RMSE": 0, "NMSE": 0, "NRMSE": 0, "SSIM": 0, "PSNR": 0}
            } for pair in self.pairs
        }

        for i in range(self.num_images):
            for pair in self.pairs:
                real_img_path = os.path.join(self.folder_path, f"{pair[0]}{i}.npy")
                fake_img_path = os.path.join(self.folder_path, f"{pair[1]}{i}.npy")
                if os.path.exists(real_img_path) and os.path.exists(fake_img_path):
                    real_img = np.load(real_img_path)
                    fake_img = np.load(fake_img_path)

                    # Add an extra dimension to single-channel images
                    if real_img.ndim == 2:
                        real_img = real_img[np.newaxis, :]
                    if fake_img.ndim == 2:
                        fake_img = fake_img[np.newaxis, :]

                    metrics = self.calculate_metrics(real_img, fake_img)
                    for channel_name in self.channels:
                        for metric_name, metric_value in metrics[channel_name].items():
                            metrics_totals[pair][channel_name][metric_name] += metric_value
                else:
                    print(f"Missing one of the files {real_img_path}, {fake_img_path}")
                    return

        # Calculate and print average metrics
        for pair in self.pairs:
            for channel_name in self.channels:
                print(f"\nAverage metrics for {pair[0]} and {pair[1]} - Channel {channel_name}:")
                for metric_name in metrics_totals[pair][channel_name]:
                    avg_metric_value = metrics_totals[pair][channel_name][metric_name] / self.num_images
                    print(f"{metric_name}: {avg_metric_value:.5f}")

    def evaluate_one_pair(self, pair, index):
        real_img_path = os.path.join(self.folder_path, f"{pair[0]}{index}.npy")
        fake_img_path = os.path.join(self.folder_path, f"{pair[1]}{index}.npy")
        if os.path.exists(real_img_path) and os.path.exists(fake_img_path):
            real_img = np.load(real_img_path)
            fake_img = np.load(fake_img_path)

            # Add an extra dimension to single-channel images
            if real_img.ndim == 2:
                real_img = real_img[np.newaxis, :]
            if fake_img.ndim == 2:
                fake_img = fake_img[np.newaxis, :]

            metrics = self.calculate_metrics(real_img, fake_img)
            for channel_name in self.channels:
                print(f"\nMetrics for {pair[0]} and {pair[1]} - Channel {channel_name}:")
                for metric_name, metric_value in metrics[channel_name].items():
                    print(f"{metric_name}: {metric_value:.5f}")
            
            for channel_index, channel_name in enumerate(self.channels):
                fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                im_real = axs[0].imshow(real_img[0, channel_index, self.y_start:self.y_end, self.x_start:self.x_end])
                axs[0].set_title(f'{pair[0]} Image - Channel {channel_name}')
                fig.colorbar(im_real, ax=axs[0], fraction=0.046, pad=0.04)

                im_fake = axs[1].imshow(fake_img[0, channel_index, self.y_start:self.y_end, self.x_start:self.x_end])
                axs[1].set_title(f'{pair[1]} Image - Channel {channel_name}')
                fig.colorbar(im_fake, ax=axs[1], fraction=0.046, pad=0.04)

                plt.show()
        else:
            print(f"One or both of the files {real_img_path}, {fake_img_path} do not exist.")