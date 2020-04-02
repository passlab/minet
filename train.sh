#!/usr/bin/env sh
set -e
cd /home/yshi10/unet/
pwd
export PYTHONPATH=/home/yshi10/unet:$PYTHONPATH
/home/yshi10/caffe/build/tools/caffe train --solver=solver.prototxt $@
