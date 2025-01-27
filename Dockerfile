# FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel 
FROM sagemaker-base:latest

# inatall library dependencies
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Add enviroment variable for processes to be able to call fork()
ENV RDMAV_FORK_SAFE=1

# Indicate the container type
ENV DLC_CONTAINER_TYPE=training
ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive
# RUN apt-get install -y automake dh-make libcap2 libnuma-dev libtool make pkg-config udev curl librdmacm-dev rdma-core
# RUN apt-get install -y libgfortran5 bison chrpath flex graphviz gfortran tk dpatch quilt swig tcl ibverbs-utils

RUN apt-get install -y git

# Install the `uv` CLI
############################################
# Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"
# ENV UV_SYSTEM_PYTHON=1
############################################


WORKDIR /gradient
ENV PYTHONPATH=/gradient

# Install requirements
# COPY pyproject.toml .
# RUN uv sync 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY core core
COPY examples examples
COPY models models


ENV HF_HUB_CACHE=/gradient/.cache/huggingface/hub
WORKDIR /gradient/examples/bria4B_adapt

ENTRYPOINT [ "python", "example_train.py" ]
