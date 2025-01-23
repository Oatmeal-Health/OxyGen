"""
Takes care of tensor loading from disk and supplying it to the loader.
"""

import blosc
import torch
import io
from typing import List
import os


def load_blosc(file_path: str) -> torch.Tensor:
    """Load a PyTorch object from a Blosc-compressed file."""
    with open(file_path, "rb") as f:
        return torch.load(io.BytesIO(blosc.decompress(f.read())), weights_only=True)


def load_torch(file_path: str) -> torch.Tensor:
    """Load a PyTorch object from raw Torch file."""
    return torch.load(file_path, weights_only=True)


class TensorStore:
    """Take cares of reading tensors from disk in .pt and .pt.blosc format."""

    supported_types = {
        '.pt.blosc': load_blosc,
        '.pt': load_torch,
    }

    @staticmethod
    def get_valid_filenames(scan_file: str, tensor_root: str) -> List[str]:
        """Returns names of valid scan files. The caller must call to form full filenames."""
        assert tensor_root, "Tensor root should be set to the tensor directory"
        assert scan_file, f"Scan file should exist and contain names of scan files"
        filenames = [line.strip() for line in open(scan_file, 'r').readlines()]
        filenames = [f'{tensor_root}/{filename}' for filename in filenames]
        out = []
        for filename in filenames:
            if os.path.isfile(filename):
                for stype in TensorStore.supported_types:
                    if filename.endswith(stype):
                        out.append(filename)
        return out

    def __init__(self, filenames: List[str]):
        """Keeps track of good and bad files as it encounters them."""
        self.filenames = filenames
        self.bad = {}

    def get_at(self, idx: int) -> torch.Tensor:
        """Returns the tensor at the index or one close to it."""
        # TODO: random jumps for bad tensors, not just the next one.
        attempts = 0
        while attempts < 2*len(self.filenames):
            attempts += 1
            filename = self.filenames[idx]
            if filename in self.bad:
                idx = (idx + 1) % len(self.filenames)
                continue
            try:
                for ext, func in self.supported_types.items():
                    if filename.endswith(ext):
                        return func(filename)
            except:
                self.bad[filename] = 1

        raise Exception("NO NEXT TENSOR FOUND")

    def limit_by(self, n: int):
        self.filenames = self.filenames[:n]

    def __len__(self) -> int:
        return len(self.filenames) - len(self.bad)
