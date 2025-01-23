"""Test the integrity of a directory, to see if we want to re-generate the tensors."""

import sys
import pandas as pd
from typing import List
import tqdm

from tensor_store import TensorStore

assert len(sys.argv) >= 2, "Please provide the name of the file containing all file names to check."
root_dir = ''
if len(sys.argv) == 3:
    root_dir = sys.argv[2]

df = pd.read_csv(sys.argv[1], header=None)
filenames: List[str] = list(df[df.columns[0]])

for filename in tqdm.tqdm(filenames):
    for ext, loader in TensorStore.supported_types.items():
        if root_dir:
            filename = f'{root_dir}/{filename}'
        if filename.endswith(ext):
            try:
                loader(filename)
            except Exception as e:
                print(f'Exception loading {filename}: {e}')
