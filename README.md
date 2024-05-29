# HalfedgeCNN

The Pytorch implementation for our paper: **HalfEdgeFormer: A High-efficient network for Half-edge Structure with Transformer**

## Framework

<p align="center">
  <img width="750" src="https://github.com/AlvinHan123/HalfedgeFormer/blob/main/framework.png"> 
</p>

Overview of the proposed framework, HalfEdgeFormer. The process begins with "Mesh Simplification," where complex 3D models are reduced to simplified geometric representations using methods like QEM (Quadric Error Metric). These simplified meshes are then processed in the "Graph Transformer" component of the framework. Here, positional embeddings derived from the Laplacian linear projections of vertex and edge features form the input to a series of graph transformer layers. These layers utilize multi-head attention mechanisms, normalization, and MLP (Multi-Layer Perceptron) blocks to process the graph data iteratively. The graph transformer outputs feature vectors which are then classified using an MLP head into various categories leverages modern neural network techniques tailored for graph data, aiming to efficiently classify 3D shapes by learning from their geometric and topological features.

## Getting Started

### Install dependencies
To set up your development environment, follow these steps:

1. Create a new Anaconda environment:
    ```bash
    conda create --name halfedgeformer python=3.8
    ```
2. Activate the new environment:
    ```bash
    conda activate halfedgeformer    
    ```
3. Install Pytorch with the correct CUDA version (if your system supports it):
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
4. Optionally, install tensorboardX for training visualization:
    ```bash
    pip install tensorboard
    conda install -c conda-forge tensorboardx

    % install other dependencies
    % pip install six 
    ```

**Note:** Depending on your system, you may need to install additional packages.

### 3D Shape Classification on SHREC

To begin training for SHREC classification, follow these steps:

1. Download and unzip the SHREC dataset:
    ```bash
    bash scripts/get_shrec_data.sh
    ```
2. Start the training process with:
    ```bash
    python scripts/train_with_settings.py shrec_16  
    ```
3. To view the training loss and accuracy plots, run the following in another terminal:
    ```bash
    tensorboard --logdir runs
    ```

## Results on SHREC11 dataset
| Method                           | Split 16  |
|----------------------------------|-----------|
| MeshCNN [Hanocka et al., 2019]   | 98.6%     |
| HalfedgeCNN [Ludwig et al., 2023]| 99.5%     |
| PD-MeshNet [Milano et al., 2020] | 99.7%     |
| MeshWalker [Lahav et al., 2020]  | 98.6%     |
| HodgeNet [Smirnov et al., 2021]  | 99.2%     |
| DiffusionNet [Sharp et al., 2022]| 99.7%     |
| SubdivNet [Hu et al., 2022]      | **100%**  |
| **HalfedgeFormer (Ours)**        | 98.4%     |

Table: The best numbers are in **bold**.

Code references: \
[HalfedgeCNN](https://graphics.cs.uos.de/)


### Appendix
In the processing of the SHREC11 dataset, edge contraction using Quadric Error Metrics (QEM) is omitted due to the dataset's inherently small size. For larger, more complex mesh structures, preprocessing steps from Homework 3 should be applied to manage and reduce mesh complexity effectively.
