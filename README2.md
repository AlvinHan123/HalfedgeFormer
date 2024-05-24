# HalfEdgeTransformer for Mesh Segmentation

Version 2.0

Note: This code is based on a modification of the code from .

### Overview
This repository contains the implementation of HalfEdgeTransformer, an advanced mesh segmentation model that integrates the robust feature extraction capabilities of Transformers with the HalfEdgeCNN architecture. This model is designed to handle high-resolution 3D mesh data, providing precise and efficient semantic segmentation by leveraging the global contextual information captured by Transformers.
### Features
1. Enhanced Global Context Awareness: Utilizes the self-attention mechanism of Transformers to capture long-range dependencies within the mesh data.
2. Precision in Segmentation: Achieves more accurate segmentation, especially in complex geometrical structures, by understanding both local and global mesh features.
3. Scalability and Flexibility: Efficiently processes variable-sized meshes and adapts to different levels of mesh complexities.

### Contributions
1. Introduction of HalfEdgeFormer: A novel method that incorporates transformer structures into the HalfEdgeCNN framework, replacing traditional convolutions.
2. Performance Validation: Demonstrates the effectiveness of HalfEdgeFormer on the SHREC dataset, showing comparable or superior performance to HalfEdgeCNN.
3. Exploration of Transformer-Based Mesh Processing: Opens new possibilities for applying transformers in geometric deep learning by demonstrating their utility in mesh processing tasks.

# Getting Started
