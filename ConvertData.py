import toml
import ROOT
from ROOT import RDataFrame
import h5py
import argparse
import sys,os, re

def pair_up(Scatter_paths, Cone_paths):
    if(len(Scatter_paths) != len(Cone_paths)):
        print("No root files detected")
        sys.exit()
    if(len(Scatter_paths) == 0):
        print("No root files detected")
        sys.exit()
    pairs = []
    for file1 in Scatter_paths:
        regex_search = re.search("_[0-9]+.root",file1)
        if not regex_search:
            print("Invalid Extraction File "+ file1)
            sys.exit()
        id = regex_search.group(0)
        flag = False
        for file2 in Cone_paths:
            if(file1.endswith(id) and file2.endswith(id)):
                pairs.append((file1, file2))
                flag=  True
                break
        if not flag:
            print("Couldn't find matching file for:" + file1)
            sys.exit()
    return pairs

def print_entries(run, event):
    print(run, event)

def convert(pair):
    # root file containing scatters
    scatter_df = RDataFrame("FilteredSeries",pair[0])
    # root file containing cones
    out_f = ROOT.TFile(pair[1],"read")
    outputs = out_f.Get("Cones")
    # Get Event and Run of each cone
    for cone in outputs:
        run = cone.Run
        event = cone.Event
        print(run,event)
        scatter_df.Filter()
    out_f.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ROOTtoSafetensor')
    parser.add_argument('GenDataTOML',help="Path to .toml config file to generate data")
    parser.add_argument('PixelTOML',help="Path to .toml config file with pixellation data")
    args = parser.parse_args()

    try:
        GramsConfig = toml.load(args.GenDataTOML)
    except:
        print("Couldn't read"+ args.GenDataTOML)
        sys.exit()

    try:
        PixelConfig = toml.load(args.PixelTOML)
    except:
        print("Couldn't read"+ args.PixelTOML)
        sys.exit()

# Check if root files exist
    base_path = GramsConfig["General"]["output_directory"]
    Data_path = os.path.join(base_path, "Background", "Cones")
#    Data_path = os.path.join(base_path, "Background","Cones")
#    if not os.path.exists(Data_path):
#        print("Couldn't find" + Data_path)
#        sys.exit()
    Scatter_paths = []
    Cone_paths = []
    Extract_ID = GramsConfig["Background"]["Extract"]["ExtractOutput"]
    Recon_ID = GramsConfig["Background"]["Reconstruct"]["ReconstructOutput"]
    for file in os.listdir(Data_path):
        if (file.endswith(".root")) and (Extract_ID in file):
            Scatter_paths.append(os.path.join(Data_path, file))
        if (file.endswith(".root")) and (Recon_ID in file):
            Cone_paths.append(os.path.join(Data_path, file))
    pairs = pair_up(Scatter_paths, Cone_paths)
    output_name = PixelConfig["Pixel"]["OutputName"]
    convert(pairs[9])
#    with h5py.File(output_name, "a") as f:
    # Meta data on total number of training data pairs
#        meta_data = f.create_group("meta")
#        meta_data.attrs["samples"] = len(Scatter_paths)
#        inpt_labels = f.create_group("input")
#        output_labels = f.create_group("output")