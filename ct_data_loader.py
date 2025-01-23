"""
CT Scan Data Module for PyTorh Lightning.

TODO: adapt for use with other PyTorch DataLoaders, not only NLST.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import lru_cache

from nlst_dataloader import nlst_dataloader
from config_mgmt import Config
from tensor_store import TensorStore


class CTScanDataModule(pl.LightningDataModule):
    """A Pytorch Lightning data module for loading CT scans."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tensor_root = self.cfg.data_store.tensor_root

    @lru_cache(maxsize=3)
    def _get_dataloader_type(self, split_group: str, scan_file: str) -> DataLoader:
        """Returns a (cached) data loader of the specific split type."""
        filenames = TensorStore.get_valid_filenames(scan_file, self.tensor_root)
        print(f"{len(list(set(filenames)))} unique filenames loaded from {split_group}:")

        dataloader = nlst_dataloader(
            batch_size=self.cfg.training.batch_size,
            subvolume_size=self.cfg.data.subvolume_size,
            mask_ratio=self.cfg.data.mask_ratio,
            mask_size=self.cfg.data.mask_size,
            dataset_size=int(self.cfg.data.dataset_size),
            filenames=filenames,
            shuffle=(split_group == 'train')
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader"""
        return self._get_dataloader_type('train', self.cfg.data_store.train_scans)

    def val_dataloader(self) -> DataLoader:
        """Returns val dataloader"""
        return self._get_dataloader_type('val', self.cfg.data_store.val_scans)

    def test_dataloader(self) -> DataLoader:
        """Returns test dataloader"""
        return self._get_dataloader_type('test', self.cfg.data_store.test_scans)
