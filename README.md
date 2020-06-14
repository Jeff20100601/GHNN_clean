## Graph Hawkes Neural Network for Forecasting on Temporal Knowledge Graphs

This repository contains code for the reprsentation proposed in [Graph Hawkes Neural Network for Forecasting on Temporal Knowledge Graphs](https://openreview.net/pdf?id=kXVazet_cB) paper.
## Installation
- Create a conda environment:
```
$ conda create -n ghnn python=3.6 anaconda
```
```
$ source activate ghnn
```
- Change directory to the GHNN folder
- Install PyTorch (>= 0.4.0)
## How to use
After installing the requirements, run the following command to preprocess datasets.:
```
$ python3 data/DATA_NAME/get_history.py
$ python3 data/DATA_NAME/get_history_tpre_appro.py
```
To train and test the model.
```
$ python3 co-train.py -d DATA_NAME  
```
Only evaluating the model.
```
$ python3 co-train.py -d DATA_NAME --only_eva true --eva_dir MODEL_DIR
```
## Citation
If you use the codes, please cite the following paper:
```
@inproceedings{han2020graph,
  title={Graph Hawkes Neural Network for Forecasting on Temporal Knowledge Graphs},
  author={Han, Zhen and Ma, Yunpu and Wang, Yuyi and Günnemann, Stephan and Tresp, Volker},
  booktitle={AKBC},
  year={2020}
}
```
## License
Copyright (c) 2020-present, Siemens AG.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
