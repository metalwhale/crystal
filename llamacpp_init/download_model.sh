#!/bin/sh

MODEL_URL=$1
OUTPUT_PATH=$2

if [ -f $OUTPUT_PATH ]; then
  exit 0
fi

curl ${MODEL_URL} -L -o ${OUTPUT_PATH}
