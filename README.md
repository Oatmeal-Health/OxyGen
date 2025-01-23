# Directions on how to train/visualize the model

1. Our SSL models are based on the UNETR architecture.

1. We currently use some GitHub repositories slightly modified for our use. They are referenced in `extra_paths.py`. For the code to make use of those repositories, you need to run (once) the `./add_contrib.sh` script.

## Conda-based training

1. Once you've created a new conda environment for the training, install the dependencies via Pip:
```
conda activate your-training-environment
pip install -r requirements.txt
```

1. For the default training to work, directory `Data/nlst-tensors` should exist and point to `nlst-tensors` bucket or directory. See `./train-in-container.sh` for the steps.

1. Smoke-test of training the default model with default parameters:
    ```
    ./test_run_upp_cube_64.sh
    ```

1. Parameters are maintained in a hierarchical `yaml` structure in `configs` directory. The above script will print out the parameters before starting training. You can modify those parameters in the script. For example, to set `mask_ratio` to 0.25, you would add `data.mask_ratio=0.25` to the command in the script. You can also see the current settings by executing `python config_mgmt.py --config-name unetr.yaml`.

1. An example of viewing the results in Tensorboard (make sure that it's installed).
    ```
    tensorboard --logdir=Output/mae_run_20241215_050154/logs/mae_training
    ```
    Note: replace the value of `--logdir` with the right one (there will be many subdirectories under `Output`).

## Docker-based training

1. To train the model inside a Docker container, build and execute the Docker image via `./build-run-image.sh` script.

1. Once the image has been built, the container executes `train-in-container.sh` script. So, the easiest way to modify the training behavior is to replace that script with another one.

1. You can modify the container environment by adjusting the last line of `build-run-image.sh` script.
