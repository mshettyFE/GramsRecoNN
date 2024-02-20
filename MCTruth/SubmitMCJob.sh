#!/bin/bash -l

function IsSet(){
    if [ -z ${!1} ]
        then 
        echo "$1 is unset"
        exit
    fi
    echo "$1 = ${!1}"
}

TOML_SANITY_CHECK_LOC=$1 #TomlSanityCheck.py path location. Used to validate Config file
TOML_FILE_LOC=$2 # .toml file containing arguments

# Make sure that these variables are set to something
IsSet TOML_SANITY_CHECK_LOC
IsSet TOML_FILE_LOC

# set up enviornment. Don't know how to generalize this. Replace this with location of you conda enviornment
conda activate /nevis/riverside/share/ms6556/conda/envs/GramsDev

# Parse arguments from config file
RESULT=$(python $TOML_SANITY_CHECK_LOC $TOML_FILE_LOC -s "Condor")
if echo $RESULT | grep "Parsing Failed!" # Search for Parsing Failed! to indicate
then
    echo "$RESULT" # Print out error message
    exit
else
    echo "$RESULT" > config_vars.sh # Store bash variables in a temporary file
fi

source config_vars.sh # add bash variables to this script
IsSet Condor_OutputFolderPath
IsSet Condor_NBatches

tar -czf  GramsSimWork.tar.gz  ../GramsSimWork
sed -i '$ d' CondorInfo.cmd
echo "initialdir=${Condor_OutputFolderPath}" >> CondorInfo.cmd
condor_submit CondorInfo.cmd -queue $Condor_NBatches