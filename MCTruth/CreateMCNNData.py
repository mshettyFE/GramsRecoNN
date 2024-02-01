import toml
from ROOT import gROOT
from ROOT import TFile
from torch import from_numpy
from safetensors.torch import save_file
import argparse
import subprocess, shlex
import numpy as np

import sys,os, random,time

sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

gROOT.SetBatch(True)

class Position:
# Given the MCTruth ROOT NTuple, convert to x,y,z coordinates to np.array for vector maths
    def __init__(self,MCTruthNTuple):
        self.pos = np.array([getattr(MCTruthNTuple,"x")[0], getattr(MCTruthNTuple,"y")[0], getattr(MCTruthNTuple,"z")[0]])
    def __repr__(self):
        output = ""
        output += str(self.pos[0])
        output += " "
        output += str(self.pos[1])
        output += " "
        output += str(self.pos[2])
        output += "\n"
        return output

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
    def __init__(self, MCTruthNTuple):
        self.position = Position(MCTruthNTuple)
        self.time = getattr(MCTruthNTuple, "t")[0]
        self.process = getattr(MCTruthNTuple, "ProcessName")
        self.energy = getattr(MCTruthNTuple, "Etot")[0]
        self.identifier = getattr(MCTruthNTuple,"identifier")[0]
    def __repr__(self):
        return self.__str__()
    def __str__(self):
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
    def add(self,scatter):
# Assumes that scatter is of type GramsG4Entry
        self.scatter_series.append(scatter)
    def sort(self):
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
        nx = int((xval+xOffset)/dx)
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
    input_labels = np.zeros((len(input_data),PixelCountX, PixelCountY) )
# 1 and 2 stands for a row vector with energy and reconstruction angle as columns
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
        input_labels[count,:,:] = anode_grid
        output_labels[count,:,:] = output_data[series]
# Checking that energies are what we expect (a mix of all in events, were energies are close to each other, and escape events, where energies are not close)
#        agg = sum(sum(input_labels[count,:,:]))
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
    GramsG4file = TFile.Open ( gramsg4_path ,"READ")
    mctruth = GramsG4file.Get("TrackInfo")
    output_mctruth_series = {}
    output_energy_angle = {}
    for entryNum in range (0 , mctruth.GetEntries()):
        mctruth.GetEntry( entryNum )
        dict_key = (getattr(mctruth, "Run"), getattr(mctruth, "Event"))
        scatter = GramsG4Entry(mctruth)
        if dict_key in output_mctruth_series:
            output_mctruth_series[dict_key].add(scatter)
        else:
            s = ScatterSeries()
            s.add(scatter)
            output_mctruth_series[dict_key] = s
    keys = list(output_mctruth_series.keys())
    for key in keys:
        if not (output_mctruth_series[key].reconstructable()):
            del output_mctruth_series[key]
    keys = list(output_mctruth_series.keys())
    for key in keys:
        output_tuple = output_mctruth_series[key].output_tuple()
        output_energy_angle[key] = output_tuple
    GramsG4file.Close()
    return output_mctruth_series, output_energy_angle

def GenData(configuration, home_dir, rng_seed):
    os.chdir(home_dir)
    home = os.getcwd()
    os.chdir(os.path.join(home,"GramsSimWork"))
    # Create .mac file to process gramssky
    nparticles = configuration["GenData"]["nparticles"]["value"]
    temp_mac_loc = os.path.join(home,"GenData","mac","temp.mac")
    with open(temp_mac_loc,'w') as f:
        f.write("/run/initialize\n")
        f.write("/run/beamOn "+str(nparticles)+"\n")
    Geo = configuration["GenData"]["Geometry"]["value"]
    ## Gramssky part
    values = ["./gramssky", "-o","Events.hepmc3", "--RadiusSphere", "300"]
    values += ["--RadiusDisc", "100"]
    values += ["--PositionGeneration", "Iso", "-n", nparticles]
    values += ["--ThetaMinMax", "\"(-1.571, 1.571)\""]
    values += ["--PhiMinMax", "\"(0,6.283)\""]
    values += ["-s", str(rng_seed),"-r", str(rng_seed)]
    if (Geo=="cube"):
        OriginSphere = "\"(0,0,-40.0)\""
    elif(Geo=="flat"):
        OriginSphere = "\"(0,0,-10.0)\""
    else:
        print("Unknown geometry")
        sys.exit()
    values += ["--OriginSphere", OriginSphere, "--EnergyGeneration", "Flat"]
    values += ["--EnergyMin", "0.1"]
    values += ["--EnergyMax", "10"]
    values += ["\n"]
    command = " ".join([str(v) for v in values])
    print(command)
    subprocess.run(shlex.split(command))
    # GramsG4
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
    file_path = os.path.join(home,"GenData","Source_"+str(rng_seed)+".root")
    os.chdir(home)
    return file_path

def GenCondorFiles(config):
    base = os.getcwd()
    shell_file_loc = os.path.join(base,"BatchGenData.sh")
    cmd_file_loc = os.path.join(base,"BatchGenData.cmd")
    python_script_loc = os.path.join(base, sys.argv[0])
    toml_file_loc = os.path.join(base,sys.argv[1])
    tar_file_name = "GramsSimWork.tar.gz"
    os.chdir("..")
    hm  = os.getcwd()
    tar_file_loc = os.path.join(hm, tar_file_name)
    file_to_tar = os.path.join(hm,"GramsSimWork")
    with open(shell_file_loc,'w') as f:
        f.write("#!/bin/bash -l\n")
        f.write("conda activate /nevis/riverside/share/ms6556/conda/envs/GramsDev\n")
        f.write("process=$1\n")
        f.write("tar -xzf "+tar_file_name +"\n")
        f.write("mkdir temp\n")
        f.write("mv "+str(sys.argv[0])+ " temp\n")
        f.write("mv "+str(sys.argv[1])+ " temp\n")
        f.write('chdir temp\n')
        python_cmd = "python " +sys.argv[0]+ " " +sys.argv[1]+ " -b " + "${process}\n"
        f.write(python_cmd)
    values = ["tar", "-cvzf",tar_file_name,"GramsSimWork"]
    command = " ".join([str(v) for v in values])
    print(command)
    subprocess.run(shlex.split(command))
    with open(cmd_file_loc,"w") as f:
        f.write("executable = "+shell_file_loc+"\n")
        f.write("transfer_input_files = "+tar_file_loc+" , "+ python_script_loc + " , " + toml_file_loc+"\n")
        f.write("arguments = $(Process)\n")
        f.write("initialdir = "+ config["GenData"]["OutputFolderPath"]["value"]+"\n")
        f.write("universe = vanilla\n")
        f.write("should_transfer_files = YES\n")
        f.write("when_to_transfer_files = ON_EXIT\n")
        f.write("requirements =  ( Arch == \"X86_64\" )\n")
        f.write("output = temp-$(Process).out\n")
        f.write("error = temp-$(Process).err\n")
        f.write("log = temp-$(Process).log\n")
        f.write("notification = Never\n")
        f.write("queue "+ str(config["GenData"]["NBatches"]["value"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CreateMCNNData')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument("-b", '--BatchNo', help="Batch Job ID", type=int, default = 0)
    parser.add_argument("-c",'--GenCondor',help="Weather to generate condor files", default = False)
    args = parser.parse_args()
    sanity_checker = TomlSanityCheck(args.GenDataTOML)
# Add Batch number to configuration dictionary. Read from command line for easier interfacing with condor
    sanity_checker.config_file["GenData"]["BatchNo"]=  {"value": int(args.BatchNo),  "constraint":"PosInt"}
    sanity_checker.config_file["GenData"]["GenCondor"]=  {"value": bool(args.GenCondor),  "constraint":"Boolean"}
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    GramsConfig = sanity_checker.return_config()
    if(GramsConfig["GenData"]["GenCondor"]):
        GenCondorFiles(GramsConfig)
    else:
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
            input_data, output_data = ReadRoot(GramsConfig, gramsg4_file)
            new = CreateTensor(GramsConfig, input_data, output_data,run)
            output_tensor.update(new)
    # Add to meta data on how many images were generated in each run for a given batch
            for k in new.keys():
                meta[str(run)] = str(new[k].shape[0])
                break
            os.remove(gramsg4_file)
        fname = os.path.join(GramsConfig["GenData"]["OutputFolderPath"]["value"],GramsConfig["GenData"]["OutputFileBaseName"]["value"]+"_"+str(GramsConfig["GenData"]["BatchNo"]["value"])+".safetensors")
        save_file(output_tensor,fname, metadata=meta)
