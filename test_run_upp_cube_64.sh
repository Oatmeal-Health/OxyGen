#!/usr/bin/bash

# Uses mostly default settings except for batch size and device count and dataset size
python main_training.py --config-name unetr.yaml \
    training.batch_size=2 \
    training.device_count=1 \
    training.train_batch_limit=50 \
    training.max_epochs=10 \
    data_store=other_data_store \
    data.dataset_size=2000
