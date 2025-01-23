#!/bin/bash

# NOTE: make sure that your settings in the YAML files are accurate,
# including data location, number of GPUs, etc.

############

# GCP bucket

TENSOR_DIR=Data/nlst-tensors

mkdir -p $TENSOR_DIR
gcsfuse --implicit-dirs --o rw nlst-tensors $TENSOR_DIR

############

# Shell script to be executed in the container once everything is set up.
bash test_run_upp_cube_64.sh

############
