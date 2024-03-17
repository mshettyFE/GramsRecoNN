# Building

If at Nevis, run the following:
```
export CONDA_ENV_PATH="/nevis/riverside/share/ms6556/conda/envs/GramsDev"
./config_NNReco.sh
```

## Outside Nevis
If you are not at Nevis, then use the following to create a local conda enviornment:
```
name="GramsDev"
file="enviornment.yml"
conda env update --name $name --file $file
```

Then, run ```conda env list``` and replace the CONDA_ENV_PATH with your own GramsDev path.

Once you have the enviornment set up, run the following to finish the setup:
```
export CONDA_ENV_PATH=""YOUR_CONDA_ENV_PATH_HERE"
./config_NNReco.sh
```

# Generating Data
If you want to generate more MCTruth training data run the following command in the MCTruth folder
```
./DataGen.sh ../TomlSanityCheck.py Config.toml $num
```
where ```$num``` is the unique run number for the output file. You can change the output directory in the GenData header in Config.toml file under the GenData directory. You can also fiddle with other parameters if you like.

In fact, for your first run, you WILL need to fiddle with the Config.toml since some value are specific to Nevis/myself. It should be well-documented enough to make changes easily?

If you are at Nevis, you can run the following in the MCTruth folder
```
./SubmitMCJob.sh ../TomlSanityCheck.py Config.toml
```

Once again for your first batch run, you WILL need to fiddle with Config.toml.
