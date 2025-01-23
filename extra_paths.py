"""Paths to other packages not installed by sourced and changed locally."""

import sys

# Add the parent directory of the 'sybil' folder to the system path
# Source: git@github.com:Oatmeal-Health/regina_barzilay_group_sybil.git
try:
    sys.path.insert(0, f'Contrib/regina_barzilay_group_sybil')
except OSError:
    # assuming in container
    pass

import sybil


# Add the parent directory of the 'unetr_plus_plus' folder to the system path
# Source: git@github.com:Oatmeal-Health/amshaker_unetr_plus_plus.git
try:
    sys.path.insert(0, f'Contrib/amshaker_unetr_plus_plus')
except OSError:
    # in container
    pass

import unetr_pp
