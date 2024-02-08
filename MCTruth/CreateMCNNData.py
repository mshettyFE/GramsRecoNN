# Script that calls GramsSky and GramsG4 to generate data, extracts the Compton scatter series from the data, and then writes them out to a tuple file
# This is done by by using shlex and subprocess to construct terminal commands which gets processed by the OS
# The reason for this existing is that doing this manually was very tedious.

# This script will eventually need to expand this to encompass DetSim/ElecSim output
# Doing this should be relatively straightforward. You would just need to add to the GenData function to call gramsdetsim after gramsg4. You also might need
# to modify the .toml configuration. This process is repeated for GramsReadoutSim, and GramsElecSim
# You would also need to modify what you write to the .safetensors file. The output will remain the same (as in you still use gramsg4 output/truth-level-output for the output tensor)
# However, the input might become more involved (Need to generate multiple channels instead of 1 channel for ElecSim)

# This script can also run in another mode. By supplying -c on the command line, the Script will generate condor file so that you can submit it on the Nevis cluster.
# Work in progress. Should easily be modifed to run elsewhere.
# You don't need to interact with this aspect of the code if you don't want to. You can just run everything locally and it _SHOULD_ be fine

# To read config files
import toml
# Reading ROOT
from ROOT import gROOT
from ROOT import TFile
# Convert from numpy to pytorch tensor
from torch import from_numpy
# save to .safetensor file
from safetensors.torch import save_file
# command line parsing
import argparse
# terminal command running
import subprocess, shlex

# normal imports
import numpy as np
import sys,os, random,time

# Add the directory above MCTruth to Python path. Needed to access TomlSanityCheck, which does data  validation on Config file
# TomlSanityCheck.py is in the parent directory since I might want to use those commands in another directory in the repo
sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

# Run ROOT in batch mode
gROOT.SetBatch(True)

# For condor, stream output and error back to local machine. Why is this not a command line argument?
# Good question.
stream = True

class Position:
# Very bare bones R^3 vector. Supports addition, subtraction, L2 norm and dot product, which are the only operations I can about for Scatter Series Reconstruction
# Effectively a Wrapper around numpy, but converts gramsg4 truth tuple info into Python object
    def __init__(self,MCTruthNTuple):
    # Given the MCTruth ROOT NTuple, convert to x,y,z coordinates to np.array for vector maths
        self.pos = np.array([ getattr(MCTruthNTuple,"x")[0], getattr(MCTruthNTuple,"y")[0], getattr(MCTruthNTuple,"z")[0]])
    def __repr__(self):
    # If you do print(p) where p is of type Position, this is what is printed out
        output = ""
        output += str(self.pos[0])
        output += " "
        output += str(self.pos[1])
        output += " "
        output += str(self.pos[2])
        output += "\n"
        return output
    # __add__ and __sub__ are like operator overloads in C++
    def __add__(self, other):
        return self.pos + other.pos
    def __sub__(self, other):
        return self.pos - other.pos
    def norm(self):
        return self.pos.norm()
    def dot(self):
        return self.pos.dot()

class GramsG4Entry:
# Encapsulates a single entry in the MCTruth of GramsG4
# effectively just a C struct
    def __init__(self, MCTruthNTuple):
        self.position = Position(MCTruthNTuple)
        self.time = getattr(MCTruthNTuple, "t")[0]
        self.process = getattr(MCTruthNTuple, "ProcessName")
        self.energy = getattr(MCTruthNTuple, "Etot")[0]
        self.identifier = getattr(MCTruthNTuple,"identifier")[0]
    def __repr__(self):
        output = ""
        output += "\t"+self.position.__str__()
        output +="\t"+"Time: " +  str(self.time)+"\n"
        output +="\t"+"Process: " +  str(self.process)+"\n"
        output +="\t"+"Energy: " +  str(self.energy)+"\n"
        output +="\t"+"ID: " +  str(self.identifier)+"\n"
        return output
    def emit_positions(self):
        return self.position
    def emit_data(self):
        return (self.position, self.time, self.energy)

class ScatterSeries:
# Stores a list of GramsG4Entries, and determines if the series is reconstructable with Compton Cone Method
    def __init__(self):
        self.scatter_series = []
    def __repr__(self):
        count = 1
        output = ""
        for item  in self.scatter_series:
            label = "Hit "+str(count)+"\n"
            output += label
            output += item.__repr__()
            count += 1
        return output
    def __len__(self):
        return len(self.scatter_series)
    def add(self,scatter: GramsG4Entry):
    # Python lets you do type annotations now, which is neat, and useful in this case since we really only want GramsG4Entries here
        self.scatter_series.append(scatter)
    def sort(self):
    # Sort the scatter series by time
        self.scatter_series.sort(reverse=False, key= lambda scatter: scatter.time)
    def reconstructable(self):
        if(len(self.scatter_series) <3):
            return False
        valid_interactions = ["Primary", "phot", "compt"]
        for scatter in self.scatter_series:
            if scatter.process not in valid_interactions:
                return False
# Check that all interactions (excluding the Primary) are inside the LAr. An interaction inside LAr starts with 1.
# There are 7 digits in the identifier, so int(scatter.identifier/1000000) will specify what region in the detector the interaction took place in
        for scatter in self.scatter_series[1:]:
            if int(scatter.identifier/1000000)!=1:
                return False
        return True
    def escape_type(self):
        self.sort()
# If the last interaction was not a photoabsorbtion, then the photon escaped the detector
        if self.scatter_series[-1].process != "phot":
# 1 denotes that the series escaped
            return 1
# 0 denotes that all the series was recorded
        return 0

    def output_tuple(self):
    # Calculates the truth level energy and truth-level initial scattering angle (in radians)
        if(len(self.scatter_series) < 3):
            raise Exception("Scatter Series must be of at least length 3")
        e_type = self.escape_type()
        truth_angle = 0.0
        photon_to_first = self.scatter_series[0].position - self.scatter_series[1].position
        first_to_second = self.scatter_series[1].position - self.scatter_series[2].position
        truth_angle = np.dot(photon_to_first, first_to_second)/(np.linalg.norm(photon_to_first)*np.linalg.norm(first_to_second))
        return (self.scatter_series[0].energy,truth_angle,e_type)

def pixellate(xval,yval, xDim, yDim, PixelCountX, PixelCountY):
# Returns the x and y index on the anode plane for a given xval,yval pair
# Index (0,0) starts at bottom left. Both axes increase from left to right

# Yes, this is exactly what GramsRecoSim does. No, I could not have just used GramsRecoSim, since GramsRecoSim only accepts the .root file from GramsDetSim

# Since GramsG4 has coordinate system placed at center of anode plane, we need an Offset in x and y to ensure all values are positive
    xOffset = xDim/2.0
    yOffset = yDim/2.0

# Width of each bin
    dx = xDim/PixelCountX
    dy = yDim/PixelCountY
# edge case where xval equals xOffset. Clamp to highest pixel in the file
    if(xval==xOffset):
        nx = PixelCountX-1
    else:
# Binning
        nx = int((xval+xOffset)/dx)
# Last chance to catch out of bounds error
        if( (nx<0) or (nx>=PixelCountX)):
            raise Exception("xval outside TPC. Maybe not filtering series correctly?")
# Same for y
    if(yval==yOffset):
        ny = PixelCountY-1
    else:
        ny = int((yval+yOffset)/dy)
        if( (ny<0) or (ny>=PixelCountY)):
            raise Exception("yval outside TPC. Maybe not filtering series correctly?")
    return (nx,ny)

def CreateTensor(configuration, input_data, output_data, run):
# This assumes TomlSanityChecker has vetted configuration file
# Mass of electron in MeV. Need to subtract from each hit to get deposition energy
    mass_e =  0.51099895
    Geo = configuration["GenData"]["Geometry"]["value"].lower()
    if (Geo == "cube"):
        xDim = 70
        yDim = 70
    if (Geo == "flat"):
        xDim = 140
        yDim = 140
    PixelCountX = int(configuration["GenData"]["PixelCountX"]["value"])
    PixelCountY = int(configuration["GenData"]["PixelCountY"]["value"])
# 1 stands for number of channels (needed to be compatible with 2D conv layers)
# For Final version, will need to increase number of channels
    input_labels = np.zeros((len(input_data),1,PixelCountX, PixelCountY) )
# 1 and 3 stands for a row vector with energy, reconstruction angle, and escape type as columns
    output_labels = np.zeros((len(input_data), 1, 3))
    count = 0
    for series in input_data:
# Initialize empty anode plane with no depositions
        anode_grid = np.zeros((PixelCountX,PixelCountY))
# Drop photon from series, and skip series if only photon
        LArHits = input_data[series].scatter_series[1:]
        if(len(LArHits) ==0):
            continue
        for hit in LArHits:
            anode_indices = pixellate(hit.position.pos[0],hit.position.pos[1],xDim,yDim, PixelCountX, PixelCountY)
# Add energy of deposition to correct pixel
            anode_grid[anode_indices[0], anode_indices[1]] += (hit.energy-mass_e)
# At this point, have a 2D "image" of energy depositions. We get associate output data, and place data in big numpy array
        input_labels[count,0,:,:] = anode_grid
        output_labels[count,:,:] = output_data[series]
# Checking that energies are what we expect (a mix of all in events, were energies are close to each other, and escape events, where energies are not close)
#        agg = sum(sum(input_labels[count,1,:,:]))
#        truth = output_labels[count,0,0]
#        print(agg,truth, abs(agg-truth) )
        count += 1
    input_labels = from_numpy(input_labels)
    output_labels = from_numpy(output_labels)
    input_name = "input_anode_images_"+str(run)+"_"
    output_name = "output_anode_images_"+str(run)+"_"
    tensors = {
        input_name: input_labels,
        output_name: output_labels
    }
    return tensors

def ReadRoot(configuration, gramsg4_path):
    if (configuration["GenData"]["Debug"]["value"]):
    # See if gramsg4_path is valid
        print("Entering")
        print(gramsg4_path)
        print(os.path.exists(gramsg4_path))
    # Open up root file
    GramsG4file = TFile.Open ( gramsg4_path ,"READ")
    if (configuration["GenData"]["Debug"]["value"]):
    # Finished reading root file
        print("Read ROOT File")
    # Get TrackInfo tuple containing MCTruth data
    mctruth = GramsG4file.Get("TrackInfo")
    mctruth_series = {}
    output_energy_angle = {}
    if (configuration["GenData"]["Debug"]["value"]):
    # Start reading entries
        print("Pulling Data")
    nentries = mctruth.GetEntries()
    for entryNum in range (0 , nentries):
        if (configuration["GenData"]["Debug"]["value"]):
            print(entryNum, nentries)
        mctruth.GetEntry( entryNum )
        dict_key = (getattr(mctruth, "Run"), getattr(mctruth, "Event"))
# Pack MCTruth info into Entry object, then pack all associated Entry object into a ScatterSeries object
        scatter = GramsG4Entry(mctruth)
# Checking if Run/Event key is present. If not, add, then add new ScatterSeries object
        if dict_key in mctruth_series:
            mctruth_series[dict_key].add(scatter)
        else:
            s = ScatterSeries()
            s.add(scatter)
            mctruth_series[dict_key] = s
    if (configuration["GenData"]["Debug"]["value"]):
        print("Classifying Data")
    # Grab all the keys (ie. all the scatter series), and check if they are reconstructable. If they are, store in an seperate dictionary
    all_keys = list(mctruth_series.keys())
    output_mctruth_series = {}
    for key in all_keys:
        if mctruth_series[key].reconstructable():
            output_mctruth_series[key] = mctruth_series[key]
    if (configuration["GenData"]["Debug"]["value"]):
        print("Formatting Data")
    # Extract the necessary information from the reconstructable series and return that
    output_keys = list(output_mctruth_series.keys())
    for key in output_keys:
        output_tuple = output_mctruth_series[key].output_tuple()
        output_energy_angle[key] = output_tuple
    # Clean up
    if (configuration["GenData"]["Debug"]["value"]):
        print("Closing File")
    if (configuration["GenData"]["Debug"]["value"]):
        print("Exiting")
    GramsG4file.Close()
    return output_mctruth_series, output_energy_angle

def GenData(configuration, home_dir, rng_seed):
# Calls GramsSim executables to generate raw .root outputs, and then parses output to extract Compton Series
# To call an external executable, you do the following:
# 1. Create a list of command line arguments that you would typically pass to
# 2. Join the arguments such that there is one space between each argument (called command)
# 3. Run subprocess.run(shlex.split(command)). Don't know why you need shlex (Assuming some formatting stuff that the " ".join() doesn't do).
# In any case, you do the above for each external script you want to run. For MCTruth, we have GramsSky-> GramsG4
# For the final version, we would expect gramssky -> gramsg4 ->gramsdetsim -> gramsRecoSim -> gramsElecSim
    os.chdir(home_dir)
    home = os.getcwd()
    GramsSimLoc = os.path.join(home,"GramsSimWork")
    gramssky_loc = os.path.join(GramsSimLoc,"gramssky")
    gramsg4_loc = os.path.join(GramsSimLoc,"gramsg4")
    if not os.path.exists(gramssky_loc):
        raise Exception("Invalid GramsSim location")
    if not os.path.exists(gramsg4_loc):
        raise Exception("Invalid GramsSim location")
    os.chdir(GramsSimLoc)
    # Create temp.mac file to process gramssky output
    nparticles = configuration["GenData"]["nparticles"]["value"]
    temp_mac_loc = os.path.join(home,"GramsSimWork","mac","temp.mac")
    # This file effectively just reads in the output of GramsSky
    with open(temp_mac_loc,'w') as f:
        f.write("/run/initialize\n")
        f.write("/run/beamOn "+str(nparticles)+"\n")
    Geo = configuration["GenData"]["Geometry"]["value"]
    ## Gramssky part. See https://github.com/wgseligman/GramsSim/tree/master/GramsSky or ./gramssky -h for info on what the arguments are
    values = ["./gramssky", "-o","Events.hepmc3", "--RadiusSphere", "300"]
    values += ["--RadiusDisc", "100"]
    values += ["--PositionGeneration", "Iso", "-n", nparticles]
    # Generate across the entire sky
    values += ["--ThetaMinMax", "\"(-1.571, 1.571)\""]
    values += ["--PhiMinMax", "\"(0,6.283)\""]
    # Seed simulation with rng
    values += ["-s", str(rng_seed),"-r", str(rng_seed)]
    # Place origin depending on gmdl geometry
    if (Geo=="cube"):
        OriginSphere = "\"(0,0,-40.0)\""
    elif(Geo=="flat"):
        OriginSphere = "\"(0,0,-10.0)\""
    else:
        print("Unknown geometry")
        sys.exit()
    # Generate energies in a uniform distribution ranging from 0.1 to 10 MeV
    values += ["--OriginSphere", OriginSphere, "--EnergyGeneration", "Flat"]
    values += ["--EnergyMin", "0.1"]
    values += ["--EnergyMax", "10"]
    values += ["\n"]
    # Join all arguments, then run
    command = " ".join([str(v) for v in values])
    print(command)
    subprocess.run(shlex.split(command))
    # GramsG4. See https://github.com/wgseligman/GramsSim/tree/master/GramsG4 or ./gramsg4 -h
    values = []
    values += ["./gramsg4"]
    if(Geo=="cube"):
        gdml_path = os.path.join(home, "gdml", "ThinGrams.gdml")
    elif(Geo=="flat"):
        gdml_path = os.path.join(home, "gdml", "ThinFlatGrams.gdml")
    else:
        print("Unknown Geometry")
        sys.exit()
    values += ["-g",gdml_path]
    values += ["-i", "Events.hepmc3","-s", str(rng_seed), "-r" ,str(rng_seed)]
    values += ["-o","Source_"+str(rng_seed)+".root","-m",temp_mac_loc]
    values += ["\n"]
    command = " ".join([str(v) for v in values])
    print(command)
    subprocess.run(shlex.split(command))
    file_path = os.path.abspath("Source_"+str(rng_seed)+".root")
    # Return to directory where python script lives
    os.chdir(home)
    return file_path

def GenCondorFiles(config):
    # Generate .sh and .cmd such that this script can be run on condor
    base = os.getcwd()
    # Create the paths to all the input files
    shell_file_loc = os.path.join(base,"BatchGenData.sh")
    cmd_file_loc = os.path.join(base,"BatchGenData.cmd")
    python_script_loc = os.path.join(base, sys.argv[0])
    toml_file_loc = os.path.join(base,sys.argv[1])
    tar_file_name = "GramsSimWork.tar.gz"
    # Move up a directory to access TomlSanityChecker and tarball
    os.chdir("..")
    hm  = os.getcwd()
    toml_parser_name = "TomlSanityCheck.py"
    toml_parser_loc = os.path.join(hm, toml_parser_name)
    tar_file_loc = os.path.join(hm, tar_file_name)
    gdml_loc = os.path.join(hm,"gdml")
    # Write the shell script which will run on the batch farm
    with open(shell_file_loc,'w') as f:
        f.write("#!/bin/bash -l\n")
        # Set up conda enviornment.
        # Yes, I know I shouldn't hard code a path. But it's fine, since so far, the only people who would use condor on this live at Nevis.
        # If you ever run this out of Nevis, change the path to the correct version
        f.write("conda activate /nevis/riverside/share/ms6556/conda/envs/GramsDev\n")
        # Read in condor provided batch id
        f.write("process=$1\n")
        # Unzip GramsSim files
        f.write("tar -xzf "+tar_file_name +"\n")
        # set up correct file structure so that python script will run
        f.write("mkdir temp\n")
        f.write("mv "+str(sys.argv[0])+ " temp\n")
        f.write("mv "+str(sys.argv[1])+ " temp\n")
        f.write('cd temp\n')
        # Run this script for the given batch number
        if stream:
        # Turn off buffering if streaming data back to local machine (the -u argument to python)
            python_cmd = "python -u " +sys.argv[0]+ " " +sys.argv[1]+ " -b " + "${process}\n"
        else:
            python_cmd = "python " +sys.argv[0]+ " " +sys.argv[1]+ " -b " + "${process}\n"
        f.write(python_cmd)
    # 1. Store command line arguments which are seperated by spaces in a list
    values = ["tar", "-czf",tar_file_name,"GramsSimWork"]
    # 2. Join the arguments with a space
    command = " ".join([str(v) for v in values])
    # 3. Run via subprocess. Doesn't work if you don't use shlex for some reason...
    subprocess.run(shlex.split(command))
    # Write .cmd file to use with condor_submit
    with open(cmd_file_loc,"w") as f:
        f.write("executable = "+shell_file_loc+"\n")
        # Transfer over the tarball, the python script, the config file, the config file parser, and the gdml folder
        f.write("transfer_input_files = "+tar_file_loc+" , "+ python_script_loc + " , " + toml_file_loc+","+toml_parser_loc+","+gdml_loc+"\n")
        f.write("arguments = $(Process)\n")
        f.write("initialdir = "+ config["GenData"]["OutputFolderPath"]["value"]+"\n")
        f.write("universe = vanilla\n")
        f.write("should_transfer_files = YES\n")
        if stream:
            f.write("stream_output = True\n")
            f.write("stream_error = True\n")
        f.write("request_memory = 1024M\n")
        f.write("request_disk   = 102400K\n")
        f.write("when_to_transfer_output = ON_EXIT\n")
        f.write("requirements =  ( Arch == \"X86_64\" )\n")
        f.write("output = temp-$(Process).out\n")
        f.write("error = temp-$(Process).err\n")
        f.write("log = temp-$(Process).log\n")
        f.write("notification = Never\n")
        # Read number of batches to generate from command line
        f.write("queue "+ str(config["GenData"]["NBatches"]["value"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CreateMCNNData')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument("-b", '--BatchNo', help="Batch Job ID", type=int, default = 0)
    parser.add_argument("-c",'--GenCondor',help="Weather to generate condor files",action='store_true')
    parser.add_argument("-d", '--Debug',help="debug flag",action='store_true')
    args = parser.parse_args()
    sanity_checker = TomlSanityCheck(args.GenDataTOML)
    # Add Batch number to configuration dictionary. Read from command line for easier interfacing with condor
    sanity_checker.config_file["GenData"]["BatchNo"]=  {"value": int(args.BatchNo),  "constraint":"PosInt"}
    sanity_checker.config_file["GenData"]["GenCondor"]=  {"value": args.GenCondor,  "constraint":"Boolean"}
    sanity_checker.config_file["GenData"]["Debug"]=  {"value": args.Debug,  "constraint":"Boolean"}
    # Make sure that config file has sane parameters that satisfy the constraints
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    GramsConfig = sanity_checker.return_config()
    # Condor script generation branch
    if(GramsConfig["GenData"]["GenCondor"]["value"]):
        GenCondorFiles(GramsConfig)
    else:
        if GramsConfig["GenData"]["Debug"]["value"]:
            print("Generating Batch"+str(args.BatchNo))
        # I assume that this script is run in some folder that is parallel to another folder where ./gramssky and gramsg4 are located at
        # So like
        # .
        #   /GramsSimWork
        #   /MCTruth
        #       CreateMCNNData.py

        # Move up to parent directory
        os.chdir("..")
        hm = os.getcwd()
        random.seed(time.time())
        output_tensor = {}
        max_runs = int(GramsConfig["GenData"]["nruns"]["value"])
        meta = {}
        for run in range(max_runs):
    # Properly seed the simulation RNG
            rng_seed  = max_runs*int(GramsConfig["GenData"]["BatchNo"]["value"])+run
            gramsg4_file = GenData(GramsConfig, hm, rng_seed )
            if (GramsConfig["GenData"]["Debug"]["value"]):
                print(gramsg4_file)
            # Run simulation, and generate tensors to pass to PyTorch
            input_data, output_data = ReadRoot(GramsConfig, gramsg4_file)
            new = CreateTensor(GramsConfig, input_data, output_data,run)
            output_tensor.update(new)
    # Add to meta data how many images were generated in each run for a given batch
            first = list(new.keys())[0]
            meta[str(run)] = str(new[first].shape[0])
            if (GramsConfig["GenData"]["Debug"]["value"]):
                print(meta)
            os.remove(gramsg4_file)
        fname = os.path.join(GramsConfig["GenData"]["OutputFolderPath"]["value"],GramsConfig["GenData"]["OutputFileBaseName"]["value"]+"_"+str(GramsConfig["GenData"]["BatchNo"]["value"])+".safetensors")
        save_file(output_tensor,fname, metadata=meta)