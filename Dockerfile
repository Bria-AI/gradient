FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

# inatall library dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Set this enviroment variable for SageMaker to launch SMDDP correctly.
ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

# Add enviroment variable for processes to be able to call fork()
ENV RDMAV_FORK_SAFE=1

# Indicate the container type
ENV DLC_CONTAINER_TYPE=training
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y automake dh-make libcap2 libnuma-dev libtool make pkg-config udev curl librdmacm-dev rdma-core
RUN apt-get install -y libgfortran5 bison chrpath flex graphviz gfortran tk dpatch quilt swig tcl ibverbs-utils

RUN apt-get install -y git

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
ENV UV_SYSTEM_PYTHON=1
WORKDIR /gradient
ENV PYTHONPATH=/gradient
COPY . .
RUN uv sync
CMD python examples/bria4B_adapt/example_train.py
