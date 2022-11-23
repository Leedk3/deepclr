ARG TAG

FROM deepclr-deps:${TAG}

# Install DeepCLR
COPY LICENSE README.md requirements.txt setup.cfg setup.py /tmp/deepclr/
COPY deepclr /tmp/deepclr/deepclr

RUN conda uninstall traitlets
RUN conda uninstall Jinja2

RUN conda install notebook ipykernel jupyterlab
RUN pip install traitlets==5.5.0
RUN conda install Jinja2==3.0

RUN cd /tmp/deepclr \
		&& python setup.py install \
		&& rm -rf /tmp/deepclr
