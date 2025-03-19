#!/bin/bash

# Prerequisites
sudo apt update -y
sudo apt install -y libgomp1

# Install llama.cpp
curl https://github.com/ggml-org/llama.cpp/releases/download/b4938/llama-b4938-bin-ubuntu-x64.zip -L -o /usr/src/llama.zip
unzip /usr/src/llama.zip -d /usr/src/llama
sudo mv /usr/src/llama/build/bin/lib* /usr/local/lib/
sudo mv /usr/src/llama/build/bin/llama-cli /usr/local/bin/
sudo mv /usr/src/llama/build/bin/llama-gguf-split /usr/local/bin/
rm -rf /usr/src/llama
sudo ldconfig /usr/local/lib
