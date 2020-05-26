
# File structure:
|File/Directory		|	Description																				|
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|../data         	|	The Penn Treebank data is expected to be placed in a directory 'data' at the same level as the git repository directory.						|
|dataloader.py		|	Contains a function to load and preprocess the Penn Treebank data.													|
|rnnlm.py		|	the torch.nn.Module that implements the RNNLM baseline.															|
|sentence-vae.py	|	Contains the torch.nn.Module that implements the VAE-Encoder, and the Module that implements the SentenceVAE (which combines the VAE-Encoder and RNNLM)			|
|train.py		|	Contains the procedures for training either the RNNLM or the SentenceVAE												|
|hparamsearch.py	|	Contains procedures that we used to call the training procedures for different sets of hyperparameters.									|
|final_models.py	|	Contains the procedures that invoke training for the final models, and performs the interpolation and sampling experiments.						|
|environment.yml	|	Dump of the conda environment used for development.															|


# How to run:
 - Install required packages, either by using the environment.yml or otherwise.
 - To train a model use the train.py script, without commandline options it will use all default settings, to see a list of possible options use --help.
 - To train our final model configurations and perform the experiments, simply run the final_models.py, no commandline options.

