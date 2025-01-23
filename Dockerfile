#################################################################
## Ensure that all bucket directories are unmounted!!!         ##
## If not, the container will take terabytes, not gigabytes!!! ##
#################################################################

# Use Google DL container -- preferred
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310

# build example: docker build -t om-fm:0.1 .
#
# interactive run example, with volume mapping and privileges, such as gcsfuse, and 100GB of normal memory:
#   docker run -it --memory=100g --gpus all -v /home/yakov/lung-nodule-detection/foundatio-model/:/app --privileged --pid=host --shm-size 10g om-fm:0.1
# no volume mapping...
#   docker run -it --memory=100g --gpus all --privileged --pid=host --shm-size 10g om-fm:0.1
#   nvidia-smi
#
# non-interactive run example
#   sudo docker run --rm --gpus all om-fm:0.1 nvidia-smi
#   docker run --memory=100g --gpus all -v /home/yakov/lung-nodule-detection/foundation-model/:/app --privileged --pid=host --shm-size 10g om-fm:0.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg libsm6 libxext6 \
    software-properties-common

# Install prerequisites
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m ensurepip && \
    python3 -m pip install --upgrade pip

# Install application dependencies
# https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt --ignore-installed blinker --root-user-action=ignore

COPY . /app

# Expose a port if necessary (optional)
# EXPOSE 8080

# Container vars, with root being /app
ENV PYTHONPATH=..:/app/Contrib/amshaker_unetr_plus_plus:/app/Contrib/regina_barzilay_group_sybil

# Set the default command to run your training job
CMD ["bash", "train-in-container.sh"]
