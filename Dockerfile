# Start with an NVIDIA CUDA base image that includes CUDA and cuDNN
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# RUN apt-get update

# CMD ["nvidia-smi"]
# # Install Miniconda
# RUN apt-get update && apt-get install -y wget && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
#     bash /miniconda.sh -b -p /miniconda && \
#     rm /miniconda.sh
# RUN apt-get -y install git

# # Install NVIDIA driver dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     linux-headers-$(uname -r) \
#     && rm -rf /var/lib/apt/lists/*

# # Download and install the NVIDIA runfile installer
# WORKDIR /tmp
# RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run && \
#     chmod +x NVIDIA-Linux-x86_64-460.32.03.run && \
#     ./NVIDIA-Linux-x86_64-460.32.03.run -s --no-kernel-module

# # Other dependencies and setup
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     xorg xauth \
#     libosmesa6-dev \
#     libgl1-mesa-glx \
#     libglfw3 \
#     patchelf \
#     xfce4 \
#     x11vnc \
#     && rm -rf /var/lib/apt/lists/*

# # Clone your repository
# RUN git clone https://github.com/real-stanford/diffusion_policy.git

# # Add Miniconda to PATH
# ENV PATH="/miniconda/bin:${PATH}"

# # Create the Conda environment
# RUN conda env create -f diffusion_policy/conda_environment.yaml

# # Make RUN commands use the new environment
# SHELL ["conda", "run", "-n", "robodiff", "/bin/bash", "-c"]

# # Ensure the environment is activated
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "robodiff", "/bin/bash", "-c"]

# # Start XFCE and VNC server
# CMD ["bash", "-c", "startxfce4 & /usr/bin/x11vnc -forever -usepw -create"]