# Creating Training Data (Python)
* Run Extract and Reconstruction with MCTruth turned on.
* Each sucessful reconstruction series counts as 1 image and 1 output
* Have output be Truth Energy and Truth Angle. 
* Project x,y,z,E vector onto x_pix, y_pix, E. 
    * Use Options in GramsReadoutSim to do conversion to image. Use image as input
    * Write images to one directory, and write label pairs to a parallel directory
* Write Custom Dataset python class to:
    * __init__
        * Read in output labels
        * Specify image directory
    * __len__
        * length of output
    * __getitem__
        * read in image
        * return image, label 
