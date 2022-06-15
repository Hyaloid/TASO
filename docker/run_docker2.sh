#!/usr/bin/env bash

docker run --gpus all --pid=host --net=host \
--name taso_test \
-it \
taso_test bash