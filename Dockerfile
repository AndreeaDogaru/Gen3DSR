#syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked set -eux; \
apt-get update -qq && \
apt-get install -qqy --no-install-recommends curl; \
rm -rf /var/lib/apt/lists/*; \
TINI_VERSION=v0.19.0; \
TINI_ARCH="$(dpkg --print-architecture)"; \
curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
chmod +x /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq && apt-get install -qqy --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest && \
	pyenv install-latest "3.10" && \
	pyenv global $(pyenv install-latest --print "3.10") && \
	pip install "wheel<1"
RUN apt update -y
RUN apt install -y software-properties-common python3-launchpadlib
RUN apt install -y gcc g++ aria2 git git-lfs wget libgl1 libglib2.0-0 ffmpeg cmake libgtk2.0-0 libopenmpi-dev
RUN apt install -y pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev

ENV CUDA_HOME=/usr/local/cuda-12.1/
ENV CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
ENV LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64"  
ENV LIBRARY_PATH=$CUDA_HOME/lib64  
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64  
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"  
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;9.0" 
ENV FORCE_CUDA=1 

RUN apt update -y
RUN apt install -y software-properties-common python3-launchpadlib
RUN apt install -y gcc g++ aria2 git git-lfs wget libgl1 libglib2.0-0 ffmpeg cmake libgtk2.0-0 libopenmpi-dev
RUN apt install -y pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
RUN pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY external /app/external
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/jinlinyi/PerspectiveFields.git
RUN pip install git+https://github.com/NVlabs/nvdiffrast
RUN pip install git+https://github.com/ashawkey/kiuikit 
RUN pip install https://github.com/AndreeaDogaru/mmcv/releases/download/v2.1.0/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl

RUN pip install external/dreamgaussian/diff-gaussian-rasterization 
RUN pip install external/dreamgaussian/simple-knn 
RUN python3 -m pip install -v -U xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install git+https://github.com/open-mmlab/mmdetection.git 
RUN pip install natten==0.15.1+torch220cu121 -f https://shi-labs.com/natten/wheels/ 
RUN python -m pip install -e external/detectron2 
RUN cd external/detectron2/projects/CropFormer/entity_api/PythonAPI && python3 setup.py build_ext --inplace
RUN cd external/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops &&  python3 setup.py build install

ARG HF_TOKEN='null'
RUN cd external/checkpoints && ./download.sh ${HF_TOKEN}

ENV OMP_NUM_THREADS=10
WORKDIR /app/src
