import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import safe_open

def extract_safetensor_paths(folder_name):
# helper function to get paths to all .safetensors files in a folder
    if not os.path.exists(folder_name):
        raise Exception("Invalid folder")
    for file in os.listdir(folder_name):
        if os.path.isfile(os.path.join(folder_name, file)):
            if file.endswith(".safetensors"):
                yield os.path.join(folder_name,file)

class AnodePlaneDataset(Dataset):
    def __init__(self,dataset_folder, verbose = False, max_files=None):
# dataset_folder (filepath): a folder consisting entirely of .safetensor files. Need 3 seperate folders for Test, Train, and Validation
# verbose (bool): print out stuff
# max_files (positive int): set a limit on the number of files to read. Used to select the first N files in a folder

# prefixes for tensor names in .safetensors file
        self.input_preappend = "input_anode_images"
        self.output_preappend = "output_anode_images"
# mapping between global index to relavent data to pull correct tensor (batch_filepath, input_key, output_key,  event_id)
        self.index_mapping = {}
# identifier for each event in each run in each file
        global_index = 0
# For each .safetensors file in the folder
        for fpath in extract_safetensor_paths(dataset_folder):
            with safe_open(fpath, framework="pt") as f:
# Grab the names of all of the tensors stored
                key_names = [run_id for run_id in f.keys()]
# Validate that there is an even number of tensors (1 input, 1 output)
                total_runs = int(len(key_names)/2)
                if verbose:
                    print(key_names)
                if ((len(key_names)%2 != 0) or (total_runs==0)):
                    raise Exception("Uneven number of keys. Number of input and output images don't match")
# For each run, check if input/output name is present in the keys. If they aren't throw
                for run_id in range(total_runs):
                    if max_files:
# If max_files is set, check if the current file number is greater than or equal to max_files.
                        if(run_id >= max_files):
                            break
                    input_name = self.input_preappend+"_"+str(run_id)+"_"
                    output_name = self.output_preappend+"_"+str(run_id)+"_"
                    if input_name in key_names:
                        key_names.remove(input_name)
                    else:
                        raise Exception(input_name+" not found")
                    if output_name in key_names:
                        key_names.remove(output_name)
                    else:
                        raise Exception(output_name+" not found")
# If you get here, we know that input_name and output_name are valid tensors within the file. Grab them
                    input_data = f.get_tensor(input_name)
                    output_data = f.get_tensor(output_name)
# Verify that there are the same number of events in each file
                    input_data_n_events = input_data.shape[0]
                    output_data_n_events = output_data.shape[0]
                    if input_data_n_events != output_data_n_events:
                        raise Exception("Number of events don't match between "+input_name+" and "+output_name)
# Add the fpath, input tensor name, output tensor name, and event number to the map
                    for event in range(input_data_n_events):
                        self.index_mapping[global_index] = (fpath, input_name, output_name, event)
                        global_index += 1
        self.total_images = global_index
    def __len__(self):
        return self.total_images
    def __getitem__(self,index):
# the result of this function is a tuple, where the first value is the image of the anode plane, and the second value is the truth level output variables
# The output variables are, from left to right: incident energy of gamma ray, first scattering angle, and an escape flag
# (1 for photon escaped, 0 for last scatter was a photoabsorption)
        if(index >= self.total_images):
            raise Exception("Index larger than number of images")
        target_path, target_input, target_output, target_event = self.index_mapping[index]
        input_data = None
        output_data = None
        with safe_open(target_path, framework="pt") as f:
            input_data = f.get_tensor(target_input)[target_event,:,:]
            output_data = f.get_tensor(target_output)[target_event,:,:]
        return (input_data, output_data)
    def plot(self,index):
# Plot an input anode plane image
        data = self.__getitem__(index)
        plt.pcolormesh(data[0])
        plt.colorbar()
        TotalE = data[1][0,0].item()
        RecordedE = sum(sum(data[0])).item()
        title = "Init. E: "+str(round(TotalE,3))+" Rec. E: "+str(round(RecordedE,3))
        plt.title(title)
        plt.show()

def truth_level_class_pics(anodeDataset,verbose = True, plot=True, display_escape = True):
# Aggregate all events into a single image, sorted by class (one for "All In", on for "Escape"
    total = len(anodeDataset)
    AllIn = torch.zeros(anodeDataset[0][0].shape)
    Escape = torch.zeros(anodeDataset[0][0].shape)
    for i in range(total):
        if verbose:
            if(i%1000 ==0):
                print(i,total)
        inpt, output = anodeDataset[i]
        if int(output[0,2]) == 0:
            AllIn += inpt
        else:
            Escape += inpt
    if display_escape:
        plt.pcolormesh(Escape)
        plt.title("Agg. energy deps (Escape)")
    else:
        plt.pcolormesh(AllIn)
        plt.title("Agg. energy deps (AllIN)")
    plt.colorbar()
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    if(plot):
        plt.show()
    return (AllIn, Escape)

if __name__ == "__main__":
    batch_size = 100
    ad = AnodePlaneDataset("/nevis/milne/files/ms6556/BleekerData/GramsMLRecoData/Train")
    truth_level_class_pics(ad,True, True, False)
