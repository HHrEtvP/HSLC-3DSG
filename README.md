# Exploring Hierarchical Spatial Layout Cues for 3D Point-based Scene Graph Prediction
This repository is the code implementation of our paper: 《Exploring Hierarchical Spatial Layout Cues for 3D Point-based Scene Graph Prediction》  
Our code base contains these contents:
* Detailed implementation of our HSLC-3DSG model using Pytorch.
* Our adopted version of Neural-Motif and aGCN model, made suitable for 3D Point-based Scene Graph Generation.
* Modified VoteNet code capable of using 3RScan as train and eval dataset.  

## Dependencies
```python
matplotlib
networkx
nltk
numpy
opencv-python
plyfile
scipy
six
tensorboard
tensorflow-estimator
tensorflow-gpu
torch
tqdm
trimesh
yacs
```
Our code is tested with Ubuntu 16.04.6 LTS, Python3.7.9, Pytorch v1.1.0, TensorFlow v1.14.0, CUDA 10.0 and cuDNN v7.4.

## Installation
To run our code, you need to first download the 3RScan and 3DSSG dataset and extract the corresponding data from 3RScan dataset. Please visit the dataset's main page to gain access to the datasets:  
[3RScan](https://github.com/WaldJohannaU/3RScan)  
[3DSSG](https://3dssg.github.io/)  
You need to follow the instruction in the link above correctly before doing the following steps. The final 3RScan dataset is a collection of scans, each scan is labeled a scan_id(the file's name), each scan also containing at least one .ply file and semseg.json file, which contains the point cloud data and object data of this scan respectfully.

### Pre-processe dataset
Due to the raw point cloud data and the relation data being in different datasets(3DSSG is based on 3RScan but without point cloud data). it's required to first pre-process both datasets so our model can load it correctly.
To properly load both datasets, you first need to extract the point cloud and object data from 3RScan dataset, the point cloud can be found in .ply file, and object data can be found in semseg.json in each scan's directory.We recommend you store object data and point cloud using .npy file.  
For relation data, since they are stored in a single .json file, we don't need to process them.  
Since our model is an end-to-end model which starts with VoteNet as the detector, our model's dataset format is similar to that of VoteNet so you can start from [there](https://github.com/facebookresearch/votenet/blob/main/doc/tips.md).  
our model requires 3 parts of input:
#### object&scan information
Stored in dict, this dict contains the basic information about a single 3D in-door scan, including point cloud, object centers, object sizes, etc.
#### relation matrix
A matrix containing the relation information of a scan, matrix's rows(and columns) represent objects while the elements represent relation, For example, if mat[0][4] is 5, then this means object 0 and object 4 has relation and its id is 5.
#### object id & matrix element index
Since the relation matrix come from 3DSSG and object information comes from 3RScan, there is some inconsistency in object indexes between 3DSSG and 3RScan. For example, in 3RScan, a table might be the 5th object in the scan's file, but in 3DSSG, it might be the 7th. The relation matrix uses indexes in 3DSSG's file but the object information uses indexes in 3RScan's file, so we need to tell which object corresponds to which row(or column) in the relation matrix.  
### Pre-training votenet
Our code shares the same technique of loading 3RScan dataset, thus you can use the newly-generated 3RScan dataset above to directly train VoteNet minus the relation part. Additionally, our pre-trained VoteNet checkpoint can be found here:  
[link]

## Running our model
For PredCLS:  
Train:
```python
python train_predcls.py
```
Test:
```python
python eval_predcls.py
```
For SGCLS:  
Train:
```python
python train_sgcls.py
```
Test:
```python
python eval_sgcls.py
```
For SGDet:  
Train:
```python
python train_sgdet.py
```
Test:
```python
python eval_sgdet.py
```
Note:you can change parameters like learning rate and weight decay via Argparse

## Running other models
We also provide our adoption of other models like Neural-Motif and aGCN, their script can be found in /other_models

## Pretraining support detector
Our model can be trained both jointly and separately. Since our model's relation detection module consists of 2 modules, we can first pre-train the first module, then use the trained first module to train the second module and fine-tune the first module, the pre-trained first module's checkpoint can be found here:[link]. Additionally, you can also train the first module using your own configuration.
