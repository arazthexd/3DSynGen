# 3DSynGen
This is my thesis project and it's super incomplete. It will hopefully be much more documented and things will make more sense some time in the future :D

# TODO
Primary model results satisfactory. The next steps:
- [ ] 3D geometry as conditions for the model
- [ ] Dimensionality reduction and preprocessing of molecule/reaction featurizers
- [ ] Discuss potential transfer learning from relevant models
- [ ] Use of rxnfp or drfp for featurizing reactions (not reaction templates!)
- [ ] Clean up and push the rest of the code to the repo
- [ ] Use searchers for dataset to be able to calculate more practical metrics

# Repo Structure
1) data: This folder contains a small dataset of different molecular fragments (the full dataset won't be uploaded on GitHub due to size limits)
2) src: This folder contains the main code used for running most experiments. It consists of a utils module, a data module, and a model module.

# Environment Setup
```bash
mamba create -n syngen rdkit numpy=1 pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 pyg ipykernel ipywidgets py3dmol pytorch-cluster pytorch-scatter pytorch-sparse seaborn anaconda::prince lightning tensorboard -c pyg -c pytorch -c nvidia
mamba activate syngen
pip install dill mordredcommunity[full] drfp ipyml mpire
```

# Code from other repos
I'm using code from 
- https://github.com/FlyingGiraffe/vnn
- https://github.com/drorlab/gvp-pytorch

for experimentation and those are copied here for convenience.