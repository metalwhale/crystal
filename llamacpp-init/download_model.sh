#!/bin/bash

LLM_MODEL_PATH=$1
PACKAGING_DIR_PATH=${2:-/tmp}

if [ -f $LLM_MODEL_PATH ]; then
  exit 0
fi

cd ${PACKAGING_DIR_PATH}
mkdir -p models
curl https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/293ca9a10157b0e5fc5cb32af8b636a88bede891/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -L -o ./models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
curl https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/293ca9a10157b0e5fc5cb32af8b636a88bede891/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf -L -o ./models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
llama-gguf-split --merge ./models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ${LLM_MODEL_PATH}
rm -rf ./models
