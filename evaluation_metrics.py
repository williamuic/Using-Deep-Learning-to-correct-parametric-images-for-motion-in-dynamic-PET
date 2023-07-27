import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
import os


class EvaluationMetrics:
    """
    A class used to calculate evaluation metrics for comparing images.

    Attributes
    ----------
    folder_path : str
        Path to the folder containing the images.
    num_images : int
        Number of images to be evaluated.
    pairs : list
        List of pairs of image names to compare.
    channels : list
        List of image channels.
    y_start, y_end, x_start, x_end : int
        Indices to crop the images.

    Methods
    -------
    plot_image(img, title):
        Plots a given image.
    view_images(index):
        Loads and displays images of a certain index.
    calculate_metrics(real_img, fake_img):
        Calculates metrics for a pair of real and fake images.
    calculate_average_metrics():
        Calculates and prints average metrics over a number of images.
    evaluate_one_pair(pair, index):
        Evaluates a pair of images of a certain index.
    """

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
        """
        Plots the given image.

        Parameters:
        img: numpy array
            The image data to plot.
        title: str
            The title for the plot.
        """
        fig, axs = plt.subplots(1, img.shape[0], figsize=(12, 12))
        for i in range(img.shape[0]):
            im = axs[i].imshow(img[i])
            axs[i].set_title(f'{title} - Channel {i+1}')
            axs[i].axis('off')

            # create an axes on the right side of axs[i]
            # The width of cax will be 5% of axs[i] and the padding between cax and axs[i] will be fixed at 0.05 inch
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar(im, cax=cax)
        plt.show()

    def view_images(self, index):
        """
        Load and display images of a certain index.

        Parameters:
        index: int
            The index of the images to display.
        """
        for name in self.pairs:
            file_path = os.path.join(self.folder_path, f"{name}{index}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                self.plot_image(data.squeeze(), f"{name}{index}")
            else:
                print(f"The file {file_path} does not exist.")

    def calculate_metrics(self, real_img, fake_img):
        """
        Calculates various metrics for a pair of real and fake images.

        Parameters:
        real_img: numpy array
            The real image data.
        fake_img: numpy array
            The fake image data.

        Returns:
        A dictionary containing RMSE, NMSE, NRMSE, SSIM, and PSNR for each channel.
        """
        real_img = real_img.squeeze()
        fake_img = fake_img.squeeze()

        metrics = {
            "Ki": {},
            "Vd": {}
        }

        # If the image only has one channel
        if real_img.ndim == 2:
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
        """
        Calculates and prints average metrics over a number of images.
        """
        metrics_values = {key: {metric: [] for metric in ["RMSE", "NMSE", "NRMSE", "SSIM", "PSNR"]}
                          for key in self.channels}

        for i in range(self.num_images):
            for pair in self.pairs:
                real_file_path = os.path.join(self.folder_path, f"{pair[0]}{i}.npy")
                fake_file_path = os.path.join(self.folder_path, f"{pair[1]}{i}.npy")

                if os.path.exists(real_file_path) and os.path.exists(fake_file_path):
                    real_img = np.load(real_file_path)
                    fake_img = np.load(fake_file_path)

                    metrics = self.calculate_metrics(real_img, fake_img)

                    for channel in self.channels:
                        for metric in ["RMSE", "NMSE", "NRMSE", "SSIM", "PSNR"]:
                            metrics_values[channel][metric].append(metrics[channel][metric])

        for channel in self.channels:
            print(f"\n{channel} Channel Metrics:\n")
            for metric in ["RMSE", "NMSE", "NRMSE", "SSIM", "PSNR"]:
                print(f"Avg {metric}: {np.mean(metrics_values[channel][metric])}")

    def evaluate_one_pair(self, pair, index):
        """
        Evaluates a pair of images of a certain index.

        Parameters:
        pair: tuple
            The pair of images to evaluate.
        index: int
            The index of the images to evaluate.
        """
        real_file_path = os.path.join(self.folder_path, f"{pair[0]}{index}.npy")
        fake_file_path = os.path.join(self.folder_path, f"{pair[1]}{index}.npy")

        if os.path.exists(real_file_path) and os.path.exists(fake_file_path):
            real_img = np.load(real_file_path)
            fake_img = np.load(fake_file_path)

            self.view_images(index)
            metrics = self.calculate_metrics(real_img, fake_img)

            for channel in self.channels:
                print(f"\n{channel} Channel Metrics:\n")
                for metric, value in metrics[channel].items():
                    print(f"{metric}: {value}")
        else:
            print(f"The files {real_file_path} and/or {fake_file_path} do not exist.")
