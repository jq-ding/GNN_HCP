# GNN-HCP: Graph Neural Networks for HCP Task Decoding

This repository contains code and benchmarks for applying graph neural networks (GNNs) to brain functional connectivity data from the **Human Connectome Project (HCP)**, with a focus on task-state classification and reproducible benchmarking.

## Contents

This repository hosts two related projects:

### 1. HCP Task Classification with GNNs

A benchmark of standard graph neural network architectures (GCN, GIN, GAT) on HCP task-state fMRI data. The HCP scanning protocol provides two phase-encoding directions for each subject (LR and RL) — we use one for training and the other for evaluation, providing a clean cross-acquisition validation setup.

### 2. Replicability Benchmark (NeurIPS 2024)

Code in `Benchmark_neurips24/` extends the basic GNN setup into a systematic **replicability benchmark** that evaluates how well deep models for fMRI generalize across scanners, sites, and acquisition protocols — addressing one of the central practical barriers in neuroimaging-based brain health research.

## Repository Structure

```
GNN_HCP/
├── GNNs_HCP.py                # Main training script for HCP task classification
├── Benchmark_neurips24/       # Replicability benchmark (NeurIPS 2024 follow-up)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
conda create --name gnn-hcp python=3.10
conda activate gnn-hcp
pip install -r requirements.txt
```

Required packages: `torch`, `torch-geometric`, `numpy`, `scikit-learn`, `argparse`.

## Data

Download the preprocessed HCP task-state data:

- **LR scan**: [Google Drive link](https://drive.google.com/file/d/10O3nF2_IRDPoSdZ1EGWcUOum2mHnEJ64/view?usp=sharing)
- **RL scan**: [Google Drive link](https://drive.google.com/file/d/1vRvOMbHoN1bk3KEpk22k80zLxkOgHaUP/view?usp=sharing)

Unzip and place both into a `data/` folder at the repository root.

## Usage

Train and evaluate a basic GNN architecture:

```bash
python GNNs_HCP.py \
    --model GCN \
    --hidden_channels 256 \
    --epochs 200 \
    --gpu 0
```

Supported model options: `GCN`, `GIN`, `GAT`.

For the replicability benchmark, see the README inside `Benchmark_neurips24/`.

## Related Publications

This codebase supports work in the following publications on graph-based modeling and replicability for neuroimaging:

- **Re-think and Re-design Graph Neural Networks in Spaces of Continuous Graph Diffusion Functionals** — NeurIPS 2023
- **Scanning the Horizon of Replicability in Neuroscience** — IEEE TBME 2025

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
