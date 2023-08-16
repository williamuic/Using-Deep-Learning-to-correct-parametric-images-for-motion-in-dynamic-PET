import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric

class EvaluationMetrics:
    def __init__(self, folder_path, num_images, pairs, channels, x_center, y_center, window_size=15):
        self.folder_path = folder_path
        self.num_images = num_images
        self.pairs = pairs
        self.channels = channels
        self.x_center = x_center
        self.y_center = y_center
        self.window_size = window_size

    def crop_image_multichannel(self, image):
        x_start = max(0, self.x_center - self.window_size)
        x_end = min(image.shape[-2], self.x_center + self.window_size)
        y_start = max(0, self.y_center - self.window_size)
        y_end = min(image.shape[-1], self.y_center + self.window_size)
        
        return [image[0, i, y_start:y_end, x_start:x_end] for i in range(self.channels)]

    def _compute_metrics_single_channel(self, image1, image2):
        data_range = image1.max() - image1.min()
        rmse = np.sqrt(np.mean((image1 - image2) ** 2))
        nmse = np.sum((image1 - image2) ** 2) / np.sum(image1 ** 2)
        nrmse_range = rmse / (np.max(image1) - np.min(image1))
        nrmse_mean = rmse / np.mean(image1)
        ssim_value, _ = ssim_metric(image1, image2, full=True, data_range=data_range)
        psnr = psnr_metric(image1, image2, data_range=data_range)
        return rmse, nmse, nrmse_range, nrmse_mean, ssim_value, psnr

    def compute_metrics_separately(self, image1, image2):
        return [self._compute_metrics_single_channel(image1[i], image2[i]) for i in range(self.channels)]

    def show_ROI(self, real_img, fake_img):
        for i in range(self.channels):
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(real_img[i])
            plt.title(f'Real Image ROI - Channel {i+1}')
            
            plt.subplot(1, 2, 2)
            plt.imshow(fake_img[i])
            plt.title(f'Fake Image ROI - Channel {i+1}')
            
            plt.tight_layout()
            plt.show()

    def calculate_average_metrics(self, ranges=None, show_roi=False):
        indices_to_process = []
        for r in ranges:
            indices_to_process.extend(list(range(r[0], r[1])))
        
        metrics_accumulators = {
            f"{pair[0]}_{pair[1]}_ch{i+1}": {
                'rmse': 0,
                'nmse': 0,
                'nrmse_range': 0,
                'nrmse_mean': 0,
                'ssim': 0,
                'psnr': 0,
                'counter': 0
            } for pair in self.pairs for i in range(self.channels)
        }
        
        for i in indices_to_process:
            for pair in self.pairs:
                real_img_path = os.path.join(self.folder_path, f"{pair[0]}{i}.npy")
                fake_img_path = os.path.join(self.folder_path, f"{pair[1]}{i}.npy")
                if os.path.exists(real_img_path) and os.path.exists(fake_img_path):
                    real_img = np.load(real_img_path)
                    fake_img = np.load(fake_img_path)
                    
                    real_cropped = self.crop_image_multichannel(real_img)
                    fake_cropped = self.crop_image_multichannel(fake_img)
                    
                    if show_roi:
                        self.show_ROI(real_cropped, fake_cropped)
                    
                    metrics_list = self.compute_metrics_separately(real_cropped, fake_cropped)
                    for ch_idx, metrics in enumerate(metrics_list):
                        key = f"{pair[0]}_{pair[1]}_ch{ch_idx+1}"
                        metrics_accumulators[key]['rmse'] += metrics[0]
                        metrics_accumulators[key]['nmse'] += metrics[1]
                        metrics_accumulators[key]['nrmse_range'] += metrics[2]
                        metrics_accumulators[key]['nrmse_mean'] += metrics[3]
                        metrics_accumulators[key]['ssim'] += metrics[4]
                        metrics_accumulators[key]['psnr'] += metrics[5]
                        metrics_accumulators[key]['counter'] += 1

                        print(f"Metrics for image index {i}, pair {pair}, channel {ch_idx+1}")
                        print(metrics)

        for key, metrics in metrics_accumulators.items():
            num_images = metrics['counter']
            if num_images == 0:
                continue
            print(f"\nAverage Metrics for key {key} over specified range:")
            for metric_key in ['rmse', 'nmse', 'nrmse_range', 'nrmse_mean', 'ssim', 'psnr']:
                metrics[metric_key] /= num_images
                print(f"{metric_key.upper()}: {metrics[metric_key]}")
