# 3DSynGen
Just a new public repository for my thesis. Like, who would ever want to steal my code, right? :D

# TODO
Primary model results satisfactory. The next steps:
- [ ] 3D geometry as conditions for the model
- [ ] Dimensionality reduction and preprocessing of molecule/reaction featurizers
- [ ] Discuss potential transfer learning from relevant models
- [ ] Use of rxnfp or drfp for featurizing reactions (not reaction templates!)
- [ ] Clean up and push the rest of the code to the repo
- [ ] Use searchers for dataset to be able to calculate more practical metrics

# Env
```bash
mamba create -n thesis2 rdkit numpy=1 pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 pyg ipykernel ipywidgets py3dmol pytorch-cluster pytorch-scatter pytorch-sparse seaborn anaconda::prince lightning tensorboard -c pyg -c pytorch -c nvidia
pip install dill mordredcommunity[full] drfp ipyml mpire
```

# Code from other repos
I'm using code from 
- https://github.com/FlyingGiraffe/vnn
- https://github.com/drorlab/gvp-pytorch
for experimentation and those are copied here for convenience.