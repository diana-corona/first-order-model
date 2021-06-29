FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git \
    less nano libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir /src
WORKDIR /src

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
  git+https://github.com/1adrianb/face-alignment \
  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pillow==8.2.0

ARG USER_ID
RUN useradd -m diana -u $USER_ID
COPY . /src
RUN chown -R diana:diana /src
USER diana

ARG CUDA_VISIBLE_DEVICES=0
ARG NVIDIA_VISIBLE_DEVICES=1
ENV CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ENV NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES
ENV PERSISTENT=/mnt/persistent

ENTRYPOINT ["sh", "script"]
