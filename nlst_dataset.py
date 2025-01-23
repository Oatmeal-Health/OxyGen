"""A PyTorch dataset for NLST"""

import torch
from torch.utils import data
import numpy as np
from typing import List

from tensor_store import TensorStore

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLST_Dataset(data.Dataset):
    """A PyTorch dataset for NLST"""

    def __init__(
            self,
            filenames: List[str],
            subvolume_size: int,
            mask_size: int,
            mask_ratio: float,
            dataset_size: int):
        """Create a Dataset."""
        super().__init__()
        self.tstore = TensorStore(filenames)
        if dataset_size > 0:
            self.tstore.limit_by(dataset_size)
        self.subvolume_size = subvolume_size
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.weights = [1, 1]

    def __len__(self):
        return len(self.tstore)

    def __getitem__(self, index):
        scan = self.tstore.get_at(index)
        x = self.extract_subvolume(scan)
        item = {'x': x}
        item.update(self.mask_subvolume(x.clone()))
        return item

    def extract_subvolume(self, scan: torch.Tensor):
        # Remove the first 1 or 2 extra dimensions.
        scan = scan.squeeze(0).squeeze(0)

        # Calculate the actual dimensions of the subvolume
        x_size = y_size = z_size = self.subvolume_size

        ########### This is not needed for large scans ###########
        # Get the current dimensions of the scan
        z_dim, y_dim, x_dim = scan.shape
        if z_dim < z_size or y_dim < y_size or x_dim < x_size:
            # Calculate padding needed to match the sub-volume size
            z_pad = max(0, z_size - z_dim)
            y_pad = max(0, y_size - y_dim)
            x_pad = max(0, x_size - x_dim)

            # Add uniform padding if necessary
            padding = (
                x_pad // 2, x_pad - x_pad // 2,  # Padding for x-dimension
                y_pad // 2, y_pad - y_pad // 2,  # Padding for y-dimension
                z_pad // 2, z_pad - z_pad // 2   # Padding for z-dimension
            )
            scan = torch.nn.functional.pad(scan, padding, mode='constant', value=0)
        ########### This is not needed for large scans ###########

        # Calculate random start indices for sub-volume extraction
        z_start = np.random.randint(0, scan.shape[0] - z_size + 1)
        y_start = np.random.randint(0, scan.shape[1] - y_size + 1)
        x_start = np.random.randint(0, scan.shape[2] - x_size + 1)

        # Extract sub-volume of the scan
        x = scan[
            z_start:z_start + z_size,
            y_start:y_start + y_size,
            x_start:x_start + x_size
        ]

        # Add the batch dimension back
        return x.unsqueeze(0)

    def mask_subvolume(self, x) -> dict:
        """
        Mask a 3D CT subvolume based on the given `mask_size` and `mask_ratio`.
        
        Parameters:
        x (torch.Tensor): The 3D subvolume.
        
        Returns:
        dict:
            - x_masked torch.Tensor: The masked subvolume.
            - mask torch.Tensor: The mask array (same shape as input), where masked regions are 1 and others are 0.
        """
        # Squeeze batch dimension to work with [D, H, W] shape
        x = x.squeeze(0)

        # Initialize the mask array
        x_mask = torch.zeros_like(x, dtype=torch.uint8)

        # Remove chunks randomly from the x, totaling to `mask_ratio` of x
        mask_ratio = self.mask_ratio

        # Calculate the number of voxels to mask
        total_voxels = x.numel()
        mask_voxels = int(total_voxels * mask_ratio)

        # Calculate the number of chunks to remove
        ms = self.mask_size
        chunk_size = ms**3
        num_chunks = mask_voxels // chunk_size

        # Randomly select coordinates for all chunks
        z_coords = np.random.randint(0, max(1, x.shape[0] - ms), size=num_chunks)
        y_coords = np.random.randint(0, max(1, x.shape[1] - ms), size=num_chunks)
        x_coords = np.random.randint(0, max(1, x.shape[2] - ms), size=num_chunks)

        # Apply the mask to all selected chunks
        for z_itr, y_itr, x_itr in zip(z_coords, y_coords, x_coords):
            x[z_itr:(z_itr + ms), y_itr:(y_itr + ms), x_itr:(x_itr + ms)] = 0
            x_mask[z_itr:(z_itr + ms), y_itr:(y_itr + ms), x_itr:(x_itr + ms)] = 1

        x = x.unsqueeze(0)
        x_mask = x_mask.unsqueeze(0)
        return {'x_masked': x, 'mask': x_mask}
