ARG TAG

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# System dependencies
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    # personal preference
    zsh \
    # needed for installing python dependencies
    git \
    # opencv
    libglib2.0-0 \
    libgl1-mesa-glx \
    # open3d
    libusb-1.0-0 \
    # python-prctl
    build-essential \
    libcap-dev \
    # GICP
    cmake \
    libgsl-dev \
    # KITTI Devkit
    ghostscript \
    gnuplot \
    texlive-extra-utils
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata

RUN apt-get install -q -y --no-install-recommends \
    make libssl-dev zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \ 
    xz-utils \ 
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN python -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# CUDA settings
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV TORCH_CUDA_ARCH_LIST="6.1 7.5+PTX"

# External dependencies
COPY extern /tmp/extern
RUN /tmp/extern/install.sh && rm -rf /tmp/extern

# Enable installing into conda for all users
RUN chmod go+w /opt/conda/lib/python3.8/site-packages

