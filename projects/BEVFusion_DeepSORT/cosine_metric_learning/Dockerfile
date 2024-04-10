FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

#Update packages
RUN apt update -y
RUN apt upgrade -y
RUN apt install software-properties-common -y
RUN apt-get update -y
RUN apt-get install software-properties-common -y

# Make python 3.7 the default
# RUN echo "alias python=python3.7" >> ~/.bashrc
# RUN export PATH=${PATH}:/usr/bin/python3.7
# RUN /bin/bash -c "source ~/.bashrc"

# Install system dependencies
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    #Instalando python 3.7
    #add-apt-repository ppa:deadsnakes/ppa && \
    #apt install python3.7 -y && \
    #Definindo python 3.7 como padrão
    #echo "alias python3=python3.7" >> ~/.bashrc && \
    #export PATH=/usr/bin/python3.7:${PATH} && \
    #/bin/bash -c "source ~/.bashrc" && \
    #update-alternatives --set python3 /usr/bin/python3.7 && \
    #################################
    apt-get install -y \
        apt-utils \
        vim \
        man \
        build-essential \
        wget \
        sudo \
        #python3.8 \
        python3-pip \
        htop \
        zlib1g-dev \
        swig unzip \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
        git-all&& \
        #python3.8-dev&& \
    rm -rf /var/lib/apt/lists/*

# Install python 3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update -y
RUN apt-get update -y
RUN apt install python3.7 -y
# RUN apt install python3.7-dev -y
RUN apt-get install python3-libnvinfer -y
RUN apt-get install tensorrt-dev -y

# Add 3.7 to the available alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# Set python3.7 as the default python
#RUN update-alternatives --set python /usr/bin/python3.7
RUN update-alternatives --set python /usr/bin/python3.7
RUN update-alternatives --set python3 /usr/bin/python3.7

# Install any python packages you need
COPY requirements.txt requirements.txt

# Upgrade pip. Install distutils to avoid error during pip install upgrade in python3.7
RUN apt install python3.7-distutils -y 
RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

# Install PyTorch and torchvision
#RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu112/torch_stable.html

ARG uid=2000
ARG user
RUN echo /$user $uid
RUN useradd -d /$user -u $uid $user --shell /bin/bash
USER $user

# Set the working directory
#WORKDIR /app

WORKDIR /$user/cosine_metric_learning
ENV PYTHONPATH /$user/cosine_metric_learning

ENV PATH /$user/.local/bin:$PATH
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV HOME /$user/cosine_metric_learning
ENV DATA_FOLDER data

# Set the entrypoint
#ENTRYPOINT [ "python3" ]