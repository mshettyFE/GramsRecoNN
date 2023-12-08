import toml
import ROOT
import h5py
import argparse
import subprocess, shlex
import numpy as np

import sys,os

ROOT.gROOT.SetBatch(True)

class Position:
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
    def __init__(self, MCTruthNTuple):
        self.position = Position(MCTruthNTuple)
        self.time = getattr(MCTruthNTuple, "t")[0]
        self.process = getattr(MCTruthNTuple, "ProcessName")
        self.energy = getattr(MCTruthNTuple, "Etot")[0]
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        output = ""
        output += "\t"+self.position.__str__()
        output +="\t"+"Time: " +  str(self.time)+"\n"
        output +="\t"+"Process: " +  str(self.process)+"\n"
        output +="\t"+"Energy: " +  str(self.energy)+"\n"
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
            label = "Series "+str(count)+"\n"
            output += label
            output += item.__repr__()
            count += 1
        return output

    def __len__(self):
        return len(self.scatter_series)

    def add(self,scatter):
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
        return True

    def output_tuple(self):
        if(len(self.scatter_series) < 3):
            raise Exception("Scatter Series must be of at least length 3")
        self.sort()
        truth_angle = 0.0
        photon_to_first = self.scatter_series[0].position - self.scatter_series[1].position
        first_to_second = self.scatter_series[1].position - self.scatter_series[2].position
        truth_angle = np.dot(photon_to_first, first_to_second)/(np.linalg.norm(photon_to_first)*np.linalg.norm(first_to_second))
        return (self.scatter_series[0].energy,truth_angle)

def ConvertToHDF5(configuration, input_data, output_data):
    # Do pixellation algorithm stuff here
    pass

def ReadRoot(configuration, gramsg4_path):
    GramsG4file = ROOT.TFile.Open ( gramsg4_path ," READ ")
    mctruth = GramsG4file.Get("TrackInfo")
    LArHits = GramsG4file.Get("LArHits")
    output_mctruth_series = {}
    output_LAr_series = {}
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

def GenData(configuration, rng_seed):
    home = os.getcwd()
    os.chdir(os.path.join(home,"GenData"))
    # Create .mac file to process gramssky
    nparticles = configuration["GenData"]["nparticles"]
    temp_mac_loc = os.path.join(home,"GenData","mac","temp.mac")
    with open(temp_mac_loc,'w') as f:
        f.write("/run/initialize\n")
        f.write("/run/beamOn "+str(nparticles)+"\n")
    Geo = configuration["GenData"]["Geometry"]
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
    return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CreateMCNNData')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument("rng",type=int,help="RNG seed to randomize simulation")
    args = parser.parse_args()
    try:
        GramsConfig = toml.load(args.GenDataTOML)
    except:
        print("Couldn't read"+ args.GenDataTOML)
        sys.exit()
    gramsg4_file = GenData(GramsConfig, args.rng)
    input_data, output_data = ReadRoot(GramsConfig, gramsg4_file)

#    with h5py.File(output_name, "a") as f:
    # Meta data on total number of training data pairs
#        meta_data = f.create_group("meta")
#        meta_data.attrs["samples"] = len(Scatter_paths)
#        inpt_labels = f.create_group("input")
#        output_labels = f.create_group("output")