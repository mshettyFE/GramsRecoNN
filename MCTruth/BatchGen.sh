#!/bin/bash
max_batches=$1
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi
for ((i = 1 ; i <= $max_batches ; i++ ));
do
python3 CreateMCNNData.py MCTruth.toml "$i"
done
