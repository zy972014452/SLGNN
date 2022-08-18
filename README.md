# SLGNN：Synthetic lethality prediction in human cancers based on  factor-aware knowledge graph neural network
![image](https://github.com/zy972014452/SLGNN/blob/main/framework.png)

# Installation
SLGNN is based on Pytorch and Python
## Requirements
You will need the following packages to run the code:
* python==3.9.0
* torch==1.10.0
* dgl==0.7.2
* numpy==1.21.4
* pandas==1.3.5
* scikit_learn==1.0.1
* torch_scatter==2.0.9
# Data Description
The './data' folder contains the data used by our paper. The '/data/SL/fold' folder contains the SL interaction data, and we pre-divide the SL interactions in the raw data.

The '/data/SL/raw' folder contains all raw data.
# Usage
First, you need to clone the repository or download source codes and data files. 

    $ git clone https://github.com/zy972014452/SLGNN.git

Then go to the folder '/src'

    $ cd src

You can directly run the following code to train the model:
  
    python train.py   --epoch 100 \
                      --batch_size 1024 \
                      --dim 64 \
                      --l2 0.0001 \
                      --lr 0.003 \
                      --sim_regularity 0.001 \
                      --inverse_r True \
                      --node_dropout_rate 0.5 \
                              
The rest of the hyperparameters can be viewed in the code
