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
./config_NNReco.sh
```

in base repo directory to build GramsSim and SparseConvolutionNet package.


# Generating Data
If you want to generate more MCTruth training data run the following command in the MCTruth folder
```
./DataGen.sh ../TomlSanityCheck.py Config.toml $num
```
where ```$num``` is the unique identifier for the output file. You can change the output directory in the GenData header in Config.toml file under the GenData directory. You can also fiddle with other parameters if you like.

In fact, for your first run, you WILL need to fiddle with the Config.toml since some value are specific to Nevis/myself. It should be well-documented enough to make changes easily?

If you are at Nevis, you can run the following in the MCTruth folder
```
./SubmitMCJob.sh ../TomlSanityCheck.py Config.toml
```

Once again for your first batch run, you WILL need to fiddle with Config.toml.