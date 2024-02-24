#!/bin/bash -l
# Run GramsSim code to generate raw data for Python script to parse

# Check if variable is set to something/ if it exists at all
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
RUN_NUM=$3 # Seeds RNG. For generating actual data, just increment this for each run.
DRY_RUN=$4 # Don't actually generate data yet. Only show variables and generation commands

# Make sure that these variables are set to something
IsSet TOML_SANITY_CHECK_LOC
IsSet TOML_FILE_LOC

if ! [[ $RUN_NUM =~ ^[\-0-9]+$ ]] || !(( RUN_NUM >= 0)); # Regex. Checks for an integer in first half, then checks if RUN_ID is at least 0
then
    echo "Invalid Run ID: "$RUN_NUM
    exit
fi

# set up enviornment
conda activate $CONDA_ENV_PATH

# Parse arguments from config file
RESULT=$(python $TOML_SANITY_CHECK_LOC $TOML_FILE_LOC)
if echo $RESULT | grep "Parsing Failed!" # Search for Parsing Failed! to indicate
then
    echo "$RESULT" # Print out error message
    exit
else
    echo "$RESULT" > config_vars.sh # Store bash variables in a temporary file
fi

source config_vars.sh # add bash variables to this script
# Check to see if relavent variables are set (useful in case you change a variable name)
IsSet Condor_OutputFolderPath
IsSet GenData_GramsSimWorkPath
IsSet GenData_GDMLPath
IsSet GenData_nparticles
IsSet GenData_Geometry
IsSet GenData_OutputFileBaseName

OUTPUT_NAME_BASE="$GenData_OutputFolderPath/$GenData_OutputFileBaseName"

# Format GramsG4 Output
OUTPUT_TENSORS="$OUTPUT_NAME_BASE"
OUTPUT_TENSORS+="_"
OUTPUT_TENSORS+=$RUN_NUM
OUTPUT_TENSORS+=".safetensors"

# Hard code values for intermediate simulations
GRAMSSKY_OUTPUT="gramssky.hepmc3" # default value for gramssky output
RADIUS_SPHERE=300
RADIUS_DISC=100

# These variables are gdml dependent, so change depending on $GEO value
# ORIGIN_LOC is half way up the detector, while GDML_FILE is the particular gdml file of the geometry
ORIGIN_LOC="\"(0,0,-40.0)\""
GDML_FILE=""

if [ "$GenData_Geometry" == "cube" ]
then
    ORIGIN_LOC="\"(0,0,-40.0)\""
    GDML_FILE="$GenData_GDMLPath/ThinGrams.gdml"
elif [  "$GenData_Geometry" == "flat" ] 
then
    ORIGIN_LOC="\"(0,0,-10.0)\""
    GDML_FILE="$GenData_GDMLPath/ThinFlatGrams.gdml"
else
    echo "Invalid geometry"
    exit
fi

# Remember current directory
CUR_DIR=$(pwd)

# Generate associated macro file for gramsg4 to read gramssky

MAC_TEXT="/run/initialize
/run/beamOn $GenData_nparticles"
MAC_FILE_LOC="$GenData_GramsSimWorkPath/mac/temp.mac"

echo "$MAC_TEXT" > $MAC_FILE_LOC
# Run the things

cd $GenData_GramsSimWorkPath

cmd=(./gramssky --RadiusSphere "$RADIUS_SPHERE" --RadiusDisc "$RADIUS_DISC" --PositionGeneration Iso -n $GenData_nparticles --ThetaMinMax "\"(-1.571, 1.571)\"" --PhiMinMax "\"(0,6.283)\"" -s "$RUN_NUM" -r "$RUN_NUM" --OriginSphere "$ORIGIN_LOC" --EnergyGeneration Flat --EnergyMin 0.1 --EnergyMax 10)
if [ -z ${DRY_RUN} ];
then 
    echo "${cmd[@]}"
    ./gramssky --RadiusSphere "$RADIUS_SPHERE" --RadiusDisc "$RADIUS_DISC" --PositionGeneration Iso -n $GenData_nparticles --ThetaMinMax "\"(-1.571, 1.571)\"" --PhiMinMax "\"(0,6.283)\"" -s "$RUN_NUM" -r "$RUN_NUM" --OriginSphere "$ORIGIN_LOC" --EnergyGeneration Flat --EnergyMin 0.1 --EnergyMax 10;
else
    echo "${cmd[@]}"
fi

cmd=(./gramsg4 -g "$GDML_FILE" -i "$GRAMSSKY_OUTPUT" -s "$RUN_NUM" -r "$RUN_NUM" -m "$MAC_FILE_LOC")
if [ -z ${DRY_RUN} ];
then 
    echo "${cmd[@]}"
    ./gramsg4 -g "$GDML_FILE" -i "$GRAMSSKY_OUTPUT" -s "$RUN_NUM" -r "$RUN_NUM" -m "$MAC_FILE_LOC"
else
    echo "${cmd[@]}"
fi

cd $CUR_DIR

if [ -z ${DRY_RUN} ]; 
then python CreateMCNNData.py $TOML_FILE_LOC -r "$RUN_NUM";
fi 

cd $GenData_GramsSimWorkPath

OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/gramsg4.root"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC
OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/${GRAMSSKY_OUTPUT}"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC
conda deactivate
