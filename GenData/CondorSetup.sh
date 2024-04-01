#!/bin/bash -l
process=$1
tar -xzf GramsSimWork.tar.gz
mkdir temp
mv Config.toml temp
mv CreateData.py temp
mv DataGen.sh temp
cd temp
export CONDA_ENV_PATH=/nevis/riverside/share/ms6556/conda/envs/GramsDev # Fine with hardcoding this. Will only ever run condor at Nevis anyways
./DataGen.sh  -s ../TomlSanityCheck.py -f  Config.toml -r $process -m