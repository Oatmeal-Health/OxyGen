#!/usr/bin/bash

# Uses default settings mostly
python main_training.py --config-name unetr.yaml \
    model=unetr-pp-cube-64 \
    training.device_count=4
