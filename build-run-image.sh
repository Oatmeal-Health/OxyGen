#!/usr/bin/bash

# To be executed from this directory.

# MAKE SURE THAT ALL YOUR DATA is in "Data" DIRECTOR!!!!!!!!!
# OTHERWISE THE BUILDER WOULD INCLUDE IT INTO DOCKER IMAGE!!!

# build the container
docker build -t om-fm:0.1 .

# Run the training script
docker run --memory=100g --gpus all -v /home/$USER/lung-nodule-detection/foundation-model/:/app --privileged --pid=host --shm-size 10g om-fm:0.1

#####################################################################
# may need to tweak the settings depending on regular and GPU memory.
# The other parameters are also trial and error.
# Do we need --runtime=nvidia ???
#####################################################################

# interactive way to run it
# docker run -it --memory=100g --gpus all -v /home/$USER/lung-nodule-detection/foundation-model/:/app --privileged --pid=host --shm-size 10g om-fm:0.1 /bin/bash
