# Exploring Hierarchical Spatial Layout Cues for 3D Point-based Scene Graph Prediction
This repository is the code implementation of our paper: 《Exploring Hierarchical Spatial Layout Cues for 3D Point-based Scene Graph Prediction》  
Our code base contains these contents:
* Detailed implementation of our HSLC-3DSG model using Pytorch.
* Our adopted version of Neural-Motif and aGCN model, made suitable for 3D Point-based Scene Graph Generation.
* Modified VoteNet code capable of using 3RScan as train and eval dataset.
* Pre-processing and load script of 3RScan and 3DSSG dataset. 
* Filtered clean 3RScan and 3DSSG dataset.  

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
You need to follow the instruction in the link above correctly before doing the following steps.The final 3RScan dataset is a collection of scan, each scan is labeled a scan_id(the file's name), each scan also containing at least one .ply file and semseg.json file, which containing the point cloud data and object data of this scan.

### Pre-processe dataset
Due to the raw point cloud data and the relation data is in different datasets(3DSSG is based on 3RScan but without point cloud data). it's required to first pre-process the dataset so our model can load it correctly.
First, run /model/dataset/PlyReader.py and /model/dataset/SemReader.py to extract point cloud and object bounding box data:
```
python PlyReader.py
python SemReader.py
```
After this, the structure in /model/dataset/3RScan_trainval should look like this:
```
|--model
     |--dataset
           |--3RScan_trainval
                |--depth 
                |--label 
                |--train_data_idx 
                |--val_data_idx
```
Note:There are several scans in the original 3RScan and 3DSSG dataset that are either missing data or are corrupted, thus we provided the filtered data indexes of 3RScan dataset.  

Luckily for relation data, we don't need to do too much work, just copy the 3DSSG file to /model/dataset and we are set.

Lastly run /model/dataset/rscan_data.py to compress the data into .npy files:
```python
python rscan_data.py
```

### Pre-training votenet
Our code shares the same technique of loading 3RScan dataset, thus you can use the newly-generated 3RScan dataset above to directly train VoteNet minus the relation part, additionally, our pre-trained VoteNet checkpoint can be found here:  
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
Our model can be trained both jointly and separately. Since our model's relation detection module consists of 2 modules, we can first pre-train the first module, then use the trained first module to train the second module and fine tune the first module, the pretrained first module's checkpoint can be found here:[link].Additional you can train the first module using your own configuration.
