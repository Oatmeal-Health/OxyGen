"""
Convert Dicoms to compressed tensors.

Make sure to create empty "nlst-oatmeal" directory, then mount it. E.g.,
IN_LOC=./nlst-oatmeal
sudo mkdir $IN_LOC
gcsfuse --implicit-dirs --o rw nlst-oatmeal $IN_LOC

If outputting to a bucket, also mount nlst-tensors in the same fashion somewheres:
OUT_LOC=./nlst-tensors
sudo mkdir $OUT_LOC
sudo gcsfuse --implicit-dirs --o rw nlst-tensors $OUT_LOC
"""

import sys
import os
from glob import glob
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
import numcodecs
import io

sys.path.insert(0, f'../../Contrib/regina_barzilay_group_sybil')
from sybil.serie import Serie


# TODO: move to command line
MAX_WORKERS = 20
IN_ROOT = 'nlst-oatmeal'
OUT_ROOT = '/mnt/shared/backup/nlst-tensors/Tensors'


compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)

def save_with_blosc(obj: torch.Tensor, tensor_root: str) -> str:
    """Save a Pytorch object in compressed fashion. Compression ratio: 2.5:1. Returns name of file."""
    # Serialize object using torch
    buffer = io.BytesIO()
    torch.save(obj, buffer)

    # Compress the serialized data
    name = f'{tensor_root}.pt.blosc'
    with open(name, "wb") as f:
        f.write(compressor.encode(buffer.getvalue()))
    return name


def load_with_blosc(file_path: str) -> torch.Tensor:
    """Load a PyTorch object from a Blosc-compressed file."""
    with open(file_path, "rb") as f:
        return torch.load(io.BytesIO(compressor.decode(f.read())), weights_only=True)


def load_with_blosc_light(file_path: str) -> bool:
    """Check integrity of a Blosc-compressed file."""
    with open(file_path, "rb") as f:
        compressor.decode(f.read())
        return True


def dcm_to_tensor(dicom_path: str):
    """
    Save a dicom directory as a single compressed Torch tensor.
    """

    os.makedirs(OUT_ROOT, exist_ok=True)
    tensor_root = '/'.join(dicom_path.split('/')[1:]).replace('/', '__')
    tensor_root = f'{OUT_ROOT}/{tensor_root}'

    if os.path.isfile(f'{tensor_root}.pt') or os.path.isfile(f'{tensor_root}.pt.blosc'):
        pass
    else:
        expr = f'{IN_ROOT}/{dicom_path}/*.dcm'
        dcms = glob(expr)

        try:
            serie = Serie(dcms, num_images=0)
            serie._args.num_chan = 1
            volume = serie.get_volume()
            file_path = save_with_blosc(volume, tensor_root)
            assert load_with_blosc_light(file_path)
            print("Processed", tensor_root)
        except Exception as e:
            print(f"Exception processing {tensor_root}: {e}")

        # this does not work because they are not in sync...
        # assert os.path.isfile(cloud_tensor_path)


if __name__ == '__main__':

    assert len(sys.argv) == 3, f"Usage: {sys.argv[0]}: low high"
    low = int(sys.argv[1])
    high = int(sys.argv[2])

    taken_all = pd.read_csv('taken-dirs.csv')
    taken = taken_all.iloc[low:high]

    dirs = list(taken['root_dir'])
    random.shuffle(dirs)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(dcm_to_tensor, dicom_path) for dicom_path in dirs]
        for future in futures:
            future.result()

    # test results
    fnames = glob(f'{OUT_ROOT}/*')
    random.shuffle(fnames)
    for fname in fnames[:1000]:
        tensor = load_with_blosc(fname)
        print(tensor.shape)
