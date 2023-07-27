import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class HDF5Dataset(Dataset):
    """
    HDF5Dataset is a PyTorch Dataset for loading HDF5 files.

    Attributes:
        file_path (str): Path to the hdf5 file.
        data_type (str): Type of the data to be loaded.
        transform (callable, optional): Optional transform to be applied on a sample.
        global_norm (bool, optional): If True, normalization is applied globally. Defaults to False.
        no_norm (bool, optional): If True, normalization is not applied. Defaults to False.
    """

    def __init__(self, file_path, data_type, transform=None, global_norm=False, no_norm=False):
        """
        Initialize the HDF5Dataset.

        Args:
            file_path (str): Path to the hdf5 file.
            data_type (str): Type of the data to be loaded.
            transform (callable, optional): Optional transform to be applied on a sample.
            global_norm (bool, optional): If True, normalization is applied globally. Defaults to False.
            no_norm (bool, optional): If True, normalization is not applied. Defaults to False.
        """
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        self.data_type = data_type
        self.transform = transform
        self.global_norm = global_norm
        self.no_norm = no_norm

        # Open the file to get the motion groups and motion free groups
        with h5py.File(file_path, 'r') as file:
            self.motion_groups = sorted([name for name in file if 'MotionData' in name],
                                        key=lambda x: int(x.split('_')[1]))
            self.motion_free_groups = sorted([name for name in file if 'MotionFreeData' in name],
                                             key=lambda x: int(x.split('_')[1]))

        # If global_norm is true, initialize min and max values
        if self.global_norm:
            self.min_motion = float('inf')
            self.max_motion = float('-inf')
            self.min_motion_free = float('inf')
            self.max_motion_free = float('-inf')

            # Iterate over motion groups to get global min and max values
            for motion_group, motion_free_group in zip(self.motion_groups, self.motion_free_groups):
                with h5py.File(self.file_path, 'r') as file:
                    # Load data
                    ki_motion = file[f'{motion_group}/Ki{self.data_type}'][:]
                    vd_motion = file[f'{motion_group}/Vd{self.data_type}'][:]
                    ki_motion_free = file[f'{motion_free_group}/Ki{self.data_type}'][:]
                    vd_motion_free = file[f'{motion_free_group}/Vd{self.data_type}'][:]

                    # Update global min and max values
                    self.min_motion = min(self.min_motion, ki_motion.min(), vd_motion.min())
                    self.max_motion = max(self.max_motion, ki_motion.max(), vd_motion.max())
                    self.min_motion_free = min(self.min_motion_free, ki_motion_free.min(), vd_motion_free.min())
                    self.max_motion_free = max(self.max_motion_free, ki_motion_free.max(), vd_motion_free.max())

    def __getitem__(self, idx):
        """
        Get an item from the Dataset at a specific index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: A dictionary with input images, target images, and their paths.
        """
        # Get corresponding motion group and motion free group for the current index
        motion_group = self.motion_groups[idx]
        motion_free_group = self.motion_free_groups[idx]

        # Load data
        with h5py.File(self.file_path, 'r') as file:
            ki_motion = file[f'{motion_group}/Ki{self.data_type}'][:]
            vd_motion = file[f'{motion_group}/Vd{self.data_type}'][:]
            ki_motion_free = file[f'{motion_free_group}/Ki{self.data_type}'][:]
            vd_motion_free = file[f'{motion_free_group}/Vd{self.data_type}'][:]

        # Apply normalization if necessary
        if self.no_norm:
            pass
        elif self.global_norm:
            # Global normalization
            ki_motion = 2 * (ki_motion - self.min_motion) / (self.max_motion - self.min_motion) - 1
            vd_motion = 2 * (vd_motion - self.min_motion) / (self.max_motion - self.min_motion) - 1
            ki_motion_free = 2 * (ki_motion_free - self.min_motion_free) / (self.max_motion_free - self.min_motion_free) - 1
            vd_motion_free = 2 * (vd_motion_free - self.min_motion_free) / (self.max_motion_free - self.min_motion_free) - 1
        else:
            # Local normalization
            min_pixel_value = min(ki_motion.min(), vd_motion.min(), ki_motion_free.min(), vd_motion_free.min())
            max_pixel_value = max(ki_motion.max(), vd_motion.max(), ki_motion_free.max(), vd_motion_free.max())
            ki_motion = 2 * (ki_motion - min_pixel_value) / (max_pixel_value - min_pixel_value) - 1
            vd_motion = 2 * (vd_motion - min_pixel_value) / (max_pixel_value - min_pixel_value) - 1
            ki_motion_free = 2 * (ki_motion_free - min_pixel_value) / (max_pixel_value - min_pixel_value) - 1
            vd_motion_free = 2 * (vd_motion_free - min_pixel_value) / (max_pixel_value - min_pixel_value) - 1

        # Convert arrays to tensors
        ki_motion = torch.from_numpy(ki_motion).float().unsqueeze(0)
        vd_motion = torch.from_numpy(vd_motion).float().unsqueeze(0)
        ki_motion_free = torch.from_numpy(ki_motion_free).float().unsqueeze(0)
        vd_motion_free = torch.from_numpy(vd_motion_free).float().unsqueeze(0)

        # Concatenate tensors to create input and target images
        input_image = torch.cat([ki_motion, vd_motion], dim=0)
        target_image = torch.cat([ki_motion_free, vd_motion_free], dim=0)

        # Apply transform if any
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {'A': input_image, 'B': target_image, 'A_paths': motion_group, 'B_paths': motion_free_group}

    def __len__(self):
        """
        Get the number of items in the Dataset.

        Returns:
            int: Number of items in the Dataset.
        """
        return len(self.motion_groups)


def create_dataloader(file_path, data_type, batch_size, shuffle=True, transform=None, global_norm=False, no_norm=False):
    """
    Create a DataLoader with a HDF5Dataset.

    Args:
        file_path (str): Path to the hdf5 file.
        data_type (str): Type of the data to be loaded.
        batch_size (int): Size of the batch.
        shuffle (bool, optional): If True, data is shuffled. Defaults to True.
        transform (callable, optional): Optional transform to be applied on a sample.
        global_norm (bool, optional): If True, normalization is applied globally. Defaults to False.
        no_norm (bool, optional): If True, normalization is not applied. Defaults to False.

    Returns:
        DataLoader: A DataLoader with the specified settings.
    """
    # Create the Dataset
    dataset = HDF5Dataset(file_path, data_type, transform=transform, global_norm=global_norm, no_norm=no_norm)
    
    # Create and return the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
