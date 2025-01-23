#!/bin/bash

# NOTE: make sure that your settings in the YAML files are accurate,
# including data location, number of GPUs, etc.

############

# Local on CADE -- depends on /dev/sdc1 disk and specific location
MOUNT_POINT=/mnt/shared
DISK=/dev/sdc1
SUBDIR=nlst-tensors
mkdir -p $MOUNT_POINT
mount $DISK $MOUNT_POINT
# TODO change if the below location changes
ln -s $MOUNT_POINT/backup/$SUBDIR Data

############

# Shell script to be executed in the container once everything is set up.
bash test_run_upp_cube_64.sh

############
