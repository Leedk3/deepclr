ARG TAG

FROM deepclr-deps:${TAG}

# System dependencies
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    # needed for installing python dependencies
    git \
    tmux tmuxp vim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata

# A weired problem that hasn't been solved yet
# RUN python -m pip uninstall -y SharedArray && \
#     python -m pip install SharedArray

RUN python -m pip install spconv-cu111

RUN python -m pip install --upgrade git+https://github.com/klintan/pypcd.git

RUN python -m pip install pillow==8.3.2
RUN python -m pip install trimesh
RUN python -m pip install torchsummary
RUN python -m pip install scikit-learn scikit-image
RUN python -m pip install plotly
RUN python -m pip install dash
RUN python -m pip install pyorbital
# RUN apt install -y libgl1-mesa-glx

RUN python -m pip install numba

# RUN apt-get update && apt-get install -y tmux tmuxp vim 
RUN apt-get install -y x11-xserver-utils

# Enable installing into conda for all users
RUN chmod go+w /opt/conda/lib/python3.8/site-packages

COPY pcdet /tmp/pcdet/pcdet
COPY setup_pcdet.py /tmp/pcdet
RUN cd /tmp/pcdet \
	&& python setup_pcdet.py install \
	&& rm -rf /tmp/pcdet
