#!/bin/bash

PROJ_DIR=$(realpath $(dirname $0)/..)
WORK_DIR=/root/workdir
IMAGE=learn-cuda

build() {
    docker build -t $IMAGE $PROJ_DIR
}

run() {
    docker run \
        -it \
        --rm \
        --gpus all \
        --mount type=bind,src=$PROJ_DIR,dst=$WORK_DIR \
        -w $WORK_DIR \
        $IMAGE
}

if [[ $# -lt 1 ]]; then
    echo "Usage $0 [build | run]"
else
    $@
fi
