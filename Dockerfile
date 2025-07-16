# Use CUDA 11.8 runtime base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

# Install python3.10, pip, git, build tools and dependencies
RUN apt-get update && apt-get install -y \
  python3.10 python3.10-dev python3.10-venv python3-pip \
  git curl build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install xvfb for headless display
RUN apt-get update && apt-get install -y xvfb && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Torch dependencies installation
RUN pip install --upgrade pip && \
  pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
  pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
  pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
  pip install torch-geometric && \
  pip install tqdm "numpy<2" scipy pybullet matplotlib tensorboard scikit-image open3d>=0.16.0 opencv-python>=4.8.0 pyrealsense2==2.53.1.4623

RUN pip install tqdm numpy scipy fastapi uvicorn pybullet matplotlib tensorboard scikit_image open3d==0.16.0 opencv_python>=4.5.4 pyrealsense2==2.53.1.4623 requests einops kornia codecarbon python-multipart 

# Espone la porta 8000
EXPOSE 8000

WORKDIR /workspace
COPY . .

CMD ["python", "/workspace/src/api_server/cns_server.py"]

# docker run --gpus all -p 8000:8000 nome-immagine python main.py --device cuda:0
