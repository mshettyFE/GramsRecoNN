#!/bin/bash
git submodule init
git submodule update
rm -rf GenData
mkdir GenData
cd GramsSimSensitivity
./config.sh
cd ../GenData
if [ $? -ne 0 ]
then
  exit
fi
cmake ../GramsSimSensitivity
make
cd ../SparseConvNet
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
python3 setup.py develop
