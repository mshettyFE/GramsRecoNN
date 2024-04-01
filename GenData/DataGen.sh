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

TOML_SANITY_CHECK_LOC=""
TOML_FILE_LOC=""
RUN_NUM=""
DRY_RUN=""
MCTRUTH=""

while getopts ":s:f:r:dmh" opt; do
  case $opt in
    h)
        echo "DataGen.sh"
        echo "-s [SanityCheckLoc]: Location of TomlSanityCheck.py (required)"
        echo "-f [TomlFileLoc]: Location of Config.toml (required)"
        echo "-r [RNGSeed]: RNG Seed of run (required)"
        echo "-d [Dry Run Flag]: Don't do any computation. Just print out bash commands"
        echo "-m [MCTruth Flag]: Generate MC Truth data"
        echo "-h [Help]: Print this message"
        exit 1
        ;;
    s)
        TOML_SANITY_CHECK_LOC=$OPTARG
        ;;
    f)
        TOML_FILE_LOC=$OPTARG
        ;;
    r)
      RUN_NUM=$OPTARG
      if ! [[ $RUN_NUM =~ ^[\-0-9]+$ ]] || !(( RUN_NUM >= 0)); # Regex. Checks for an integer in first half, then checks if RUN_ID is at least 0
      then
        echo "Invalid Run ID: "$RUN_NUM
        exit
      fi
      ;;
    d)
      DRY_RUN="SET"
      ;;
    m)
      MCTRUTH="SET"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

IsSet TOML_SANITY_CHECK_LOC
IsSet TOML_FILE_LOC
IsSet RUN_NUM

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
IsSet GenData_MCTruth

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

if [ -z ${MCTRUTH} ]; 
then
cmd=(./gramsdetsim -s "$RUN_NUM" )
if [ -z ${DRY_RUN} ];
then 
    echo "${cmd[@]}"
    ./gramsdetsim -s "$RUN_NUM"
else
    echo "${cmd[@]}"
fi
fi

cd $CUR_DIR

if [ -z ${DRY_RUN} ];
then
if [ -z ${MCTRUTH} ]; 
then
    python CreateData.py $TOML_FILE_LOC -r "$RUN_NUM";
else
    python CreateData.py $TOML_FILE_LOC -r "$RUN_NUM" -m;
fi 
fi

: '
if [ -z ${DRY_RUN} ]; 
then
echo "Removing Temp Files"
cd $GenData_GramsSimWorkPath

OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/gramsg4.root"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC

OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/${GRAMSSKY_OUTPUT}"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC

OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/gramsdetsim.root"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC

OUTPUT_FILE_LOC="${GenData_GramsSimWorkPath}"
OUTPUT_FILE_LOC+="/gramsreadoutsim.root"
echo $OUTPUT_FILE_LOC
rm $OUTPUT_FILE_LOC
fi
'
conda deactivate
