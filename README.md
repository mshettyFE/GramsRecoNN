# Building

If at Nevis, run the following:
```
conda activate /nevis/riverside/data/ms6556/conda/envs/GramsDev
```

use following to create conda enviornment if outside Nevis:
```
name="GramsDev"
file="enviornment.yml"
conda env update --name $name --file $file
```

run
```
./config.sh
```

in base repo directory to build GramsSim and SparseConvolutionNet package.


# Running
