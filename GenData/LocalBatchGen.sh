#!/bin/bash
# Generate N .safetensors file with labels ranging from 0 to N

# Make sure that max_batches was supplied
max_batches=$1
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

for ((i = 0 ; i <= $max_batches ; i++ ));
do
./DataGen.sh  -s ../TomlSanityCheck.py -f  Config.toml -r $i -m
done
