import tomllib
import ROOT
import h5py
import argparse
import sys,os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ROOTtoSafetensor')
    parser.add_argument('GramsSimSensitivityConfigTOML',help="Path to .toml config file In GramsSimSensitivity")
    parser.add_argument('PixelConfigTOML',help="Path to .toml config file containing pixellation info")
    args = parser.parse_args()

    try:
        with open(args.GramsSimSensitivityConfigTOML,'rb') as f:
            GramsConfig = tomllib.load(f)
    except:
        print("Couldn't read"+ args.GramsSimSensitivityConfigTOML)
        sys.exit()

    try:
        with open(args.PixelConfigTOML,'rb') as f:
            PixelConfig = tomllib.load(f)
    except:
        print("Couldn't read"+ args.PixelConfigTOML)
        sys.exit()

# Check if root files exist
    base_path = GramsConfig["General"]["output_directory"]
    Data_path = os.path.join(base_path, "Background","Cones")
    if not os.path.exists(Data_path):
        print("Couldn't find" + Data_path)
        sys.exit()
    Scatter_paths = []
    Cone_paths = []
    Extract_ID = GramsConfig["Background"]["Extract"]["ExtractOutput"]
    Recon_ID = GramsConfig["Background"]["Reconstruct"]["ReconstructOutput"]
    for file in os.listdir(Data_path):
        if (file.endswith(".root")) and (Extract_ID in file):
            Scatter_paths.append(os.path.join(Data_path, file))
        if (file.endswith(".root")) and (Recon_ID in file):
            Cone_paths.append(os.path.join(Data_path, file))

    if(len(Scatter_paths) != len(Cone_paths)):
        print("No root files detected")
        sys.exit()
    if(len(Scatter_paths) == 0):
        print("No root files detected")
        sys.exit()
    
