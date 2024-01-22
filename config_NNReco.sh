#!/bin/bash
git submodule init
git submodule update
rm -rf GramsSimWork
mkdir GramsSimWork
if [ $? -ne 0 ]
then
  exit
fi
cd GramsSimWork
cmake ../GramsSim
make
#cd ../SparseConvNet
#export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
#rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
#python3 setup.py develop
