# Takes in GramsG4 File, extracts Compton scatters from data, suppresses z axis information and pixellates data on AnodePlane to form image, and then write to disk
# To read config files
import toml
# Reading ROOT
import uproot
# Convert from numpy to pytorch tensor
from torch import from_numpy
# save to .safetensor file
from safetensors.torch import save_file
# command line parsing
import argparse

# normal imports
import numpy as np
import sys,os, random,time

# Add the directory above MCTruth to Python path. Needed to access TomlSanityCheck, which does data  validation on Config file
# TomlSanityCheck.py is in the parent directory since I might want to use those commands in another directory in the repo
sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

# sorted Branches of Gramsg4 root tuple, along with map
TrackInfo_keys = ['Etot', 'Event', 'PDGCode', 'ParentID', 'ProcessName', 'Run', 'TrackID', 'identifier', 'px', 'py', 'pz', 't', 'x', 'y', 'z']
TrackInfo_keys_map = {k:v for k,v in zip(TrackInfo_keys,range(len(TrackInfo_keys)))}


class Position:
# Very bare bones R^3 vector. Supports addition, subtraction, L2 norm and dot product, which are the only operations I can about for Scatter Series Reconstruction
# Effectively a Wrapper around numpy, but converts gramsg4 truth tuple info into Python object
    def __init__(self,x,y,z):
        self.pos = np.array([x,y,z])
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
    def __init__(self,dataframe: np.ndarray):
        if (dataframe.shape[0] != 15):
            raise Exception("This hit cannot be reconstructed")
        self.position = Position(dataframe[TrackInfoIndex('x')],dataframe[TrackInfoIndex('y')],dataframe[TrackInfoIndex('z')])
        self.time = dataframe[TrackInfoIndex('t')]
        self.process = dataframe[TrackInfoIndex('ProcessName')]
        self.energy = dataframe[TrackInfoIndex('Etot')]
        self.identifier = dataframe[TrackInfoIndex('identifier')]

    def __repr__(self):
        output = ""
        output += "\tPosition:"+self.position.__str__()
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
    input_name = "input_anode_images"
    output_name = "output_anode_images"
    tensors = {
        input_name: input_labels,
        output_name: output_labels
    }
    return tensors

def TrackInfoIndex(label:str):
    global TrackInfo_keys_map
    if label not in TrackInfo_keys_map:
        raise Exception("invalid gramsg4 label")
    out = TrackInfo_keys_map[label]
    return out


def ReadRoot(configuration, gramsg4_path):
    output_mctruth_series = {}
    output_energy_angle = {}
    # Open up root file and get TrackInfo TTree
    with uproot.open(gramsg4_path+":TrackInfo") as mctruth:
        keys = mctruth.keys()
        keys  = list(keys)
        keys.sort()
        data =  mctruth.arrays(library='np')
        # Doing this since I couldn't find any syntax that allows you to index a numpy array with entries of variable length
        # We only care about the first entry when there are multiple deposits in a hit
        for i in range(data['t'].shape[0]):
            data['t'][i] = data['t'][i][0]
            data['x'][i] = data['x'][i][0]
            data['y'][i] = data['y'][i][0]
            data['z'][i] = data['z'][i][0]
            data['px'][i] = data['px'][i][0]
            data['py'][i] = data['py'][i][0]
            data['pz'][i] = data['pz'][i][0]
            data['Etot'][i] = data['Etot'][i][0]
            data['identifier'][i] = data['identifier'][i][0]
        unique_event_mask = np.unique(data["Event"]) # Grab all the unique events
        unique_run_mask = np.unique(data["Run"]) # Grab all unique runs
        for run in unique_run_mask:
            for event in unique_event_mask:
                event_mask = np.logical_and((data["Run"]==run), ( data["Event"]==event)) # Grab all hits coming from the same initial gamma ray
                check_for_reconstructable_mask = np.logical_and(event_mask,data["ProcessName"]!="Primary") # From the hits, get all the non-photon events
                primary_mask = np.logical_and(event_mask,(data["ProcessName"]=="Primary")) # get the gamma ray for this particular event
                # get the data for this event
                scatters = [data[str(key)][check_for_reconstructable_mask] for key in keys]
                gamma = [data[str(key)][primary_mask] for key in keys]
                # package the data into a single numpy array
                scatters = np.stack(scatters)
                gamma = np.concatenate(gamma)
                # Generate the scatter series for this event 
                cur_series = ScatterSeries()
                cur_series.add(GramsG4Entry(gamma))
                for hit in range(scatters.shape[1]):
                    cur_series.add(GramsG4Entry(scatters[:,hit]))
                # If this series can be reconstructed, then add to the output
                if (cur_series.reconstructable()):
                    output_mctruth_series[(run,event)] = cur_series

        output_keys = list(output_mctruth_series.keys())
        for key in output_keys:
            output_tuple = output_mctruth_series[key].output_tuple()
            output_energy_angle[key] = output_tuple
    return output_mctruth_series, output_energy_angle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='CreateMCNNData')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument("-r", '--RunID', help="Job ID. Seeds RNG", type=int, default = 0)
    parser.add_argument("-d", '--Debug',help="debug flag",action='store_true')
    args = parser.parse_args()
    sanity_checker = TomlSanityCheck(args.GenDataTOML)
    # Add Batch number to configuration dictionary. Read from command line for easier interfacing with condor
    sanity_checker.config_file["GenData"]["RunID"]=  {"value": int(args.RunID),  "constraint":"PosInt"}
    sanity_checker.config_file["GenData"]["Debug"]=  {"value": args.Debug,  "constraint":"Boolean"}
    # Make sure that config file has sane parameters that satisfy the constraints
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    GramsConfig = sanity_checker.return_config()
    if GramsConfig["GenData"]["Debug"]["value"]:
        print("Generating Batch"+str(args.RunID))
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
    meta = {}
    # Run simulation, and generate tensors to pass to PyTorch
    gramsg4_file = os.path.join(hm,"GramsSimWork","gramsg4.root")
    input_data, output_data = ReadRoot(GramsConfig, gramsg4_file)
    new = CreateTensor(GramsConfig, input_data, output_data,GramsConfig["GenData"]["RunID"]["value"])
    output_tensor.update(new)
# Add to meta data how many images were generated in each run for a given batch
    first = list(new.keys())[0]
    meta[str(GramsConfig["GenData"]["RunID"]["value"])] = str(new[first].shape[0])
    if (GramsConfig["GenData"]["Debug"]["value"]):
        print(meta)
    fname = os.path.join(GramsConfig["Condor"]["OutputFolderPath"]["value"],GramsConfig["GenData"]["OutputFileBaseName"]["value"]+"_"+str(GramsConfig["GenData"]["RunID"]["value"])+".safetensors")
    save_file(output_tensor,fname, metadata=meta)