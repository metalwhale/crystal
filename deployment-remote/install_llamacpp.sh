#!/bin/bash

# Install some required packages
sudo apt-get install -y libgomp1

# Change to a temporary packaging directory
cd /tmp

# Install CUDA toolkit
# Doc: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y cuda-toolkit-12-8

# Install some prebuilt llama.cpp binaries and libraries with CUDA support,
# assume that all files inside `/app` directory of `ghcr.io/ggml-org/llama.cpp:server-cuda-b4927` image
# have already been copied to `/usr/local/src/llama.cpp/` directory on the host.
# Ref: https://github.com/ggml-org/llama.cpp/blob/b4927/.devops/cuda.Dockerfile
sudo mv /usr/local/src/llama.cpp/llama* /usr/local/bin/
sudo mv /usr/local/src/llama.cpp/lib* /usr/local/lib/
sudo ldconfig /usr/local/lib/

# Install some prebuilt llama.cpp binaries and libraries to work with .gguf files
curl https://github.com/ggml-org/llama.cpp/releases/download/b4927/llama-b4927-bin-ubuntu-x64.zip -L -o llama.zip
unzip ./llama.zip -d ./llama
sudo mv ./llama/build/bin/llama-gguf-split /usr/local/bin/
rm -rf ./llama.zip ./llama
