ARG TAG

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

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
    texlive-extra-utils \
    # needed for installing python dependencies
    tmux \
    tmuxp \
    vim \
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
    x11-xserver-utils \
    tzdata && \ 
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get install -y tzdata 

# RUN apt-get install -q -y --no-install-recommends \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN python -m pip install spconv-cu111
RUN python -m pip install --upgrade git+https://github.com/klintan/pypcd.git
RUN python -m pip install pillow==8.3.2
RUN python -m pip install trimesh
RUN python -m pip install torchsummary
RUN python -m pip install scikit-learn
RUN python -m pip install plotly
RUN python -m pip install dash
RUN python -m pip install pyorbital
RUN apt install -y libgl1-mesa-glx

# RUN apt-get update && apt-get install -y tmux tmuxp vim 
# RUN apt-get install -y 

# Enable installing into conda for all users
RUN chmod go+w /opt/conda/lib/python3.8/site-packages    

RUN python -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
RUN python -m pip install scikit-image easy_dict llvmlite numba

COPY pcdet /tmp/pcdet/pcdet
COPY setup_pcdet.py /tmp/pcdet
# RUN cd /tmp/pcdet \
# 	&& python setup_pcdet.py install \
# 	&& rm -rf /tmp/pcdet

# CUDA settings
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
# ENV TORCH_CUDA_ARCH_LIST="6.1 7.5+PTX"
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

# External dependencies
COPY extern /tmp/extern
RUN /tmp/extern/install.sh && rm -rf /tmp/extern

# Enable installing into conda for all users
RUN chmod go+w /opt/conda/lib/python3.8/site-packages


# # FROM deepclr-deps:${TAG}
# FROM pcdet:${TAG}

ARG UID=1000
ENV USER=${HOST_USERNAME}
RUN useradd -u $UID -ms /bin/bash $USER | echo $USER

# Install DeepCLR
COPY LICENSE README.md requirements.txt setup.cfg setup.py /tmp/deepclr/
COPY deepclr /tmp/deepclr/deepclr


RUN chown -R ${USER}:${USER} /tmp/deepclr/deepclr
RUN chmod 755 /tmp/deepclr/deepclr

# RUN chown -R ${USER}:${USER} /home/leedk/deepclr
# RUN chmod 755 /home/leedk/deepclr

# RUN conda uninstall traitlets
# RUN conda uninstall Jinja2

# RUN conda install notebook ipykernel jupyterlab
# RUN conda install -c conda-forge traitlets
# RUN conda install Jinja2==3.0

# RUN cd /tmp/deepclr \
# 	&& python setup.py install \
# 	&& rm -rf /tmp/deepclr
