# Takes in GramsG4 File, extracts Compton scatters from data, suppresses z axis information and pixellates data on AnodePlane to form image, and then write to disk
# To read config files
import toml
# Reading ROOT
import uproot
from ROOT import TH1F
# command line parsing
import argparse

# normal imports
import numpy as np
import matplotlib.pyplot as plt
import sys,os, random,time

# Add the directory above MCTruth to Python path. Needed to access TomlSanityCheck, which does data  validation on Config file
# TomlSanityCheck.py is in the parent directory since I might want to use those commands in another directory in the repo
sys.path.append('../..')
from  TomlSanityCheck import TomlSanityCheck

# sorted Branches of Gramsg4 root tuple, along with map
TrackInfo_keys = ('Etot', 'Event', 'PDGCode', 'ParentID', 'ProcessName', 'Run', 'TrackID', 'identifier', 'px', 'py', 'pz', 't', 'x', 'y', 'z')
TrackInfo_keys_map = {k:v for k,v in zip(TrackInfo_keys,range(len(TrackInfo_keys)))}

# Interactions to look out for in data
gamma_ray_process_name = "Primary"
valid_interactions = ("phot", "compt")
valid_escape_flags= ("AllIn","Escape")

# rest mass of electron in MeV
rest_mass_e = 0.511


class Position:
# Very bare bones R^3 vector. Supports addition, subtraction, L2 norm and dot product, which are the only operations I can about for Scatter Series Reconstruction
# Effectively a Wrapper around numpy, lets one turn gramsg4 x,y,z leaves into a Python object that is easier to work with
    def __init__(self,x,y,z):
        self.pos = np.array([x,y,z])
    def __repr__(self):
    # If you do print(p) where p is of type Position, this is what is pushed to stdout
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
    def emit_process(self):
        return self.process
    def emit_data(self):
        return {"position":self.position, "time":self.time, "energy":self.energy}

class ScatterSeries:
# Stores a list of GramsG4Entries, and determines if the series is reconstructable with Compton Cone Method
    def __init__(self):
        self.current_index = -1 # This is so that we can treat ScatterSeries as an iterator
        self.scatter_series = [] # the actual scatter series data
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
    def __iter__(self):
        self.current_index = -1 # Reset the counter to the begining when iterator is constructed
        return self

    def __next__(self):
        self.current_index += 1 # Advance counter, and check if it is out of bounds or not
        if self.current_index < len(self.scatter_series):
            return self.scatter_series[self.current_index]
        raise StopIteration

    def add(self,scatter: GramsG4Entry):
        self.scatter_series.append(scatter)
    def sort(self):
    # Sort the scatter series by time
        self.scatter_series.sort(reverse=False, key= lambda scatter: scatter.time)
    def reconstructable(self):
        if(len(self.scatter_series) <3):
            return False
    # Sort by time to get definite ordering, where we expect photon to be first
        self.sort()
        global valid_interactions
        global gamma_ray_process_name
        if(self.scatter_series[0].process != gamma_ray_process_name):
            return False
        for scatter in self.scatter_series[1:]:
            # Check if all scatters after first gamma ray are either Compton or Photoabsoption
            if scatter.process not in valid_interactions:
                return False
            # Check that all interactions (excluding the Primary) are inside the LAr. An interaction inside LAr starts with 1.
            # There are 7 digits in the identifier, so int(scatter.identifier/1000000) will specify what region in the detector the interaction took place in
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
    # Grab summed deposited energy
        global rest_mass_e
        summed_deposited_energy = sum([scatter.energy-rest_mass_e for scatter in self.scatter_series[1:]])
        return {"truth_energy":self.scatter_series[0].energy,
                 "truth_angle":truth_angle, 
                 "escape_type": e_type,
                   "dep_energy": summed_deposited_energy, "n_scatters": len(self)-1}
    
def TrackInfoIndex(label:str): # This function is needed since Python dictionaries have no defined order on the keys
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
                reconstructable_process_check = np.logical_and(event_mask, data["ProcessName"]!="Primary") # Check that scatters are not a gamma ray
                intermediate_mask = np.logical_and(reconstructable_process_check, data["ParentID"]==1) # Check that all scatters came immediately from ParentID
                check_for_reconstructable_mask = np.logical_and(intermediate_mask, data["PDGCode"]==11) # From the hits, get all the non-photon events
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
                # If this series can be reconstructed, then add to the output. Shame you had to do all that computation just to throw away 99% of it...
                if (cur_series.reconstructable()):
                    output_mctruth_series[(run,event)] = cur_series
        output_keys = list(output_mctruth_series.keys())
        for key in output_keys:
            output_tuple = output_mctruth_series[key].output_tuple()
            output_energy_angle[key] = output_tuple
    return output_mctruth_series, output_energy_angle

def histogram_compton_energy(scatter_series, out_dir, event_type="compt"):
    global valid_interactions
    if(event_type not in valid_interactions):
        raise Exception("event_type is not valid")
    keys = list(scatter_series.keys())
    all_in_outputs = []
    escape_outputs = []
    global rest_mass_e
    for key in keys:
        scatters = scatter_series[key]
        escape_type = scatters.escape_type()
        for hit in scatters:
            if (hit.process == event_type):
                if(escape_type == 0):
                    all_in_outputs.append(hit.energy-rest_mass_e) # subtract rest mass of electron
                else:
                    escape_outputs.append(hit.energy-rest_mass_e)
    plt.hist(all_in_outputs, bins=100, color='skyblue', edgecolor='black')
    if (event_type=="compt"):
        plt.title("All In Compton for All In Events")
    if (event_type=="phot"):
        plt.title("All In Phot for All In Events")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Count")
    home = os.getcwd()
    os.chdir(out_dir)
    if (event_type=="compt"):
        plt.savefig("AllInComptInHistogram.png")
    if (event_type=="phot"):
        plt.savefig("AllInPhotInHistogram.png")
    os.chdir(home)
    plt.clf()

    plt.hist(escape_outputs, bins=100, color='skyblue', edgecolor='black')
    if (event_type=="compt"):
        plt.title("Escape Compton for All In Events")
    if (event_type=="phot"):
        plt.title("Escape Phot for All In Events")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Count")
    home = os.getcwd()
    os.chdir(out_dir)
    if (event_type=="compt"):
        plt.savefig("EscapeComptInHistogram.png")
    if (event_type=="phot"):
        plt.savefig("EscapePhotInHistogram.png")
    os.chdir(home)

def scatter_truth(scatter_series, out_dir, y_para_name = "dep_energy"):
    keys = list(scatter_series.keys())
    truth_energy_all_in = []
    truth_energy_escape = []
    y_val_all_in = []
    y_val_escape = []
    for key in keys:
        scatters = scatter_series[key]
        data = scatters.output_tuple()
        
        if(data["escape_type"] == 0):
            truth_energy_all_in.append(data["truth_energy"])
            y_val_all_in.append(data[y_para_name])
        if(data["escape_type"] == 1):
            truth_energy_escape.append(data["truth_energy"])
            y_val_escape.append(data[y_para_name])
    plt.clf()
    plt.scatter(truth_energy_all_in,y_val_all_in, color='b')
    plt.scatter(truth_energy_escape,y_val_escape, color='r')
    plt.title("Truth versus "+y_para_name+" "+str(len(y_val_all_in))+" "+str(len(y_val_escape)))
    plt.xlabel("Truth Energy (MeV)")
    plt.ylabel(y_para_name)
    home = os.getcwd()
    os.chdir(out_dir)
    plt.savefig("TruthVs_"+y_para_name+".png")
    os.chdir(home)
    plt.clf()

def hist_data(scatter_series, out_dir, y_para_name = "n_scatters", escape_type=None):
    keys = list(scatter_series.keys())
    title = y_para_name+" Histogram"
    if(escape_type):
        title = title + "_"+ escape_type
    title = title.replace(" ","_")
    values = []
    for key in keys:
        scatters = scatter_series[key]
        data = scatters.output_tuple()
        if(escape_type):
            if((escape_type=="AllIn") and (data["escape_type"] == 0)):
                values.append(data[y_para_name])
            if((escape_type=="Escape") and (data["escape_type"] == 1)):
                values.append(data[y_para_name])
        else:
            values.append(data[y_para_name])
    plt.clf()
    home = os.getcwd()
    os.chdir(out_dir)
    root_hist = TH1F("hist", title, 100, min(values), max(values))
    root_hist.GetXaxis().SetTitle(y_para_name)
    root_hist.GetYaxis().SetTitle("Count")
    for val in values:
        root_hist.Fill(val)
    with  uproot.recreate(title+".root") as file:
        file["hist"] = root_hist
    os.chdir(home)
    plt.clf()

def ThreeDLastScatterPlot(scatter_series, out_dir,escape_type="AllIn"):
    keys = list(scatter_series.keys())
    global valid_escape_flags
    if(escape_type not in valid_escape_flags):
        raise Exception("Invalid escape value")
    title = escape_type+"LastScatter_3D"
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_vals = []
    y_vals = []
    z_vals = []
    for key in keys:
        scatters = scatter_series[key]
        escape_flag = scatters.escape_type()
        last_hit = scatters.scatter_series[-1]
        x = last_hit.position.pos[0]
        y = last_hit.position.pos[1]
        z = last_hit.position.pos[2]
        if(((escape_type=="AllIn")) and (escape_flag == 0)):
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
        if(((escape_type=="Escape")) and (escape_flag == 1)):
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
# 3d Plot
    ax.scatter(x_vals, y_vals,z_vals, marker = 'o', color='red')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title(title)
    home = os.getcwd()
    os.chdir(out_dir)
    fig.savefig(title+".png")
# X Distribution
    plt.clf()
    plt.hist(x_vals, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("X (cm)")
    plt.ylabel("Count")
    plt.savefig(title+"Histogram_X.png")
    plt.clf()
# Y Distribution
    plt.hist(y_vals, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Y (cm)")
    plt.ylabel("Count")
    plt.savefig(title+"Histogram_Y.png")
    plt.clf()
# Z Distribution
    plt.hist(z_vals, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Z (cm)")
    plt.ylabel("Count")
    plt.savefig(title+"Histogram_Z.png")
    plt.clf()
    os.chdir(home)

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
    #       this_script.py

    # Move up to parent directory
    home = os.getcwd()
    os.chdir("../..")
    hm = os.getcwd()
    output_tensor = {}
    meta = {}
    # Run simulation, and generate tensors to pass to PyTorch
    gramsg4_file = os.path.join(hm,"GramsSimWork","gramsg4.root")
    input_data, output_data = ReadRoot(GramsConfig, gramsg4_file)
    histogram_compton_energy(input_data,home,"phot")
    histogram_compton_energy(input_data,home,"compt")
    scatter_truth(input_data,home)
    hist_data(input_data, home)
    hist_data(input_data, home,escape_type="AllIn")
    hist_data(input_data, home,escape_type="Escape")
    hist_data(input_data, home,y_para_name="truth_energy")
    hist_data(input_data, home,escape_type="AllIn",y_para_name="truth_energy")
    hist_data(input_data, home,escape_type="Escape",y_para_name="truth_energy")
    hist_data(input_data, home,escape_type="AllIn",y_para_name="dep_energy")
    hist_data(input_data, home,escape_type="Escape",y_para_name="dep_energy")
    hist_data(input_data, home,y_para_name="dep_energy")
    ThreeDLastScatterPlot(input_data, home, escape_type="Escape")
    ThreeDLastScatterPlot(input_data, home, escape_type="AllIn")
