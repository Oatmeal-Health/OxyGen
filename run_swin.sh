#!/usr/bin/bash

# Uses default settings mostly
python main_training.py --config-name unetr.yaml \
    model=swin-unetr \
    training.device_count=4
