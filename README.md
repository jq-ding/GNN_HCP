# GNN_HCP

## Description
This project is about using GCN, GIN and GAT to do the task classification on HCP-task Dataset. There are two distinct scans for the same participant, i.e. LR scan and RL scan. Basically, we use one for training and the other for validation and testing, but you can choose your own way to use them.

## Installation
```
argparse
torch
numpy
scikit-learn
torch-geometric
```

Also, you can use `requirement.txt` to create conda environment by ```conda create --name <env> --file requirement.txt```

## Data

Download the [LR](https://drive.google.com/file/d/10O3nF2_IRDPoSdZ1EGWcUOum2mHnEJ64/view?usp=sharing) and [RL](https://drive.google.com/file/d/1vRvOMbHoN1bk3KEpk22k80zLxkOgHaUP/view?usp=sharing) dataset, unzip them, create a `data` folder and put them into it.

## Usage
To run this project, follow this example:

```bash
python GNNs_HCP.py --model GCN --hidden_channels 128 --epochs 200 --gpu 0
