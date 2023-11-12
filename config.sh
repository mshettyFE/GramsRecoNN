#!/bin/bash
git submodule init
git submodule update
mkdir GenData/GramsWork
cd GenData/GramsWork
if [ $? -ne 0 ]
then
  exit
fi
cmake ../../GramsSim
make
