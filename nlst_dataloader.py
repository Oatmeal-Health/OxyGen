'''
This is an NLST data loader based on NLST Dataset.

TODO: add volume augmentations, such as rotation; make sure that they are consistent across slices in the volume.
'''

from typing import List
import torch
import torch.distributed as dist
from torch.utils import data
import numpy as np
import os

from torch.utils.data.distributed import DistributedSampler

from nlst_dataset import NLST_Dataset


def nlst_dataloader(
        batch_size: int,
        subvolume_size: int,
        mask_ratio: float,
        mask_size: int,
        filenames: List[str],
        dataset_size: int = 0,
        shuffle: bool = False,  # Shuffle for non-distributed training
        num_workers: int = 4,   # Number of workers for DataLoader
        seed: int = 42          # Seed for reproducibility
    ) -> data.DataLoader:
    """Creates and returns an NLST dataloader with optional DistributedSampler."""

    # Validate the input arguments
    assert mask_size > 0, f"Invalid mask_size: {mask_size}. Must be greater than 0."
    assert 0 <= mask_ratio <= 1, f"Invalid mask_ratio: {mask_ratio}. Must be between 0.0 and 1.0."
    assert subvolume_size > 0, f"Invalid value for subvolume_size: {subvolume_size}, should be > 0."
    assert dataset_size >= 0, f"Invalid value for dataset_size: {dataset_size}, should be >= 0"

    # Dataset selection
    dataset = NLST_Dataset(
        filenames = filenames,
        subvolume_size=subvolume_size,
        mask_size=mask_size,
        mask_ratio=mask_ratio,
        dataset_size=dataset_size
    )

    # Use DistributedSampler for distributed training
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    # Custom worker initialization for consistent random seeds
    def worker_init_fn(worker_id):
        seed_all = seed + worker_id
        np.random.seed(seed_all)
        torch.manual_seed(seed_all)

    # DataLoader creation
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,  # Use sampler for distributed training
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return dataloader


"""
Test dataloader on NLST subset.

'Data/nlst-tensors/Tensors' dir should exist and contain tensors mentioned in test.csv.
Sym-link it from a different location or mount a bucket directly via gcsfuse.
"""

# Quick test
if __name__ == '__main__':
    import sys
    from tensor_store import TensorStore

    tensor_root = 'Data/nlst-tensors/Tensors'
    assert os.path.isdir(tensor_root), f'Ensure that directory {tensor_root} exists and contains tensors'
    scan_file = 'Data/test.csv'
    filenames = TensorStore.get_valid_filenames(scan_file, tensor_root)

    dataset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    loader = nlst_dataloader(
        batch_size=1,
        subvolume_size=96,
        mask_ratio=0.1,
        mask_size=5,
        filenames=filenames,
        dataset_size=dataset_size,
    )

    print(f"Loaded {len(loader.dataset)} data points referenced in {scan_file}")
