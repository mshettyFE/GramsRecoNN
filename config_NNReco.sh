#!/bin/bash -l
conda init
conda activate "$CONDA_ENV_PATH"
git submodule init
git submodule update --remote --merge
rm -rf GramsSimWork
mkdir GramsSimWork
if [ $? -ne 0 ]
then
  exit
fi
cd GramsSimWork
cmake ../GramsSim
make
cp -r GDMLSchema/ ../gdml
cd ../SparseConvNet
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
python3 setup.py develop
