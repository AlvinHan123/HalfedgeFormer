# HalfedgeCNN

The Pytorch implementation for our paper: **HalfEdgeFormer: A High-efficient Half-edge Structure with Transformer**

## Framework

<p align="center">
  <img width="750" src="https://github.com/AlvinHan123/LDMLR/blob/main/assets/framework.png"> 
</p>

Overview of the proposed framework, LDMLR. The figure describes the training of the framework: (a) obtain encoded features by a pre-training convolutional neural network on the long-tailed training set, (b) Generate pseudo-features by the diffusion model using encoded features, and (c) Train the fully connected layers using encoded and pseudo-features. The encoder from (a) and the classifier from (c) are used to predict long-tailed data in the evaluation stage.

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
   Visit [http://localhost:6006](http://
