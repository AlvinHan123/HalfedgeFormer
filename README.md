# HalfedgeCNN

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
