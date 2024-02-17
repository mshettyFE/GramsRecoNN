#!/bin/bash -l
process=$1
tar -xzf GramsSimWork.tar.gz
mkdir temp
mv Config.toml temp
mv CreateMCNNData.py temp
mv DataGen.sh temp
cd temp
./DataGen.sh ../TomlSanityCheck.py Config.toml $process
