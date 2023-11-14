# Building

If at Nevis, run the following:
```
conda activate /nevis/riverside/data/ms6556/NNGramsReco
```

use following to create conda enviornment if outside Nevis:
```
name="NNGrams"
file="env.yaml"
conda create -c conda-forge --name $name --file $file
```

run 
```
./config.sh
```

to build GramsSim and SparseConvolutionNet package.


