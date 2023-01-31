# Unsupervised Superpixel Generation using Edge-Sparse Embedding

This repository provides code to reproduce the results from:

Geusen, J., Bredell, G., Zhou, T., & Konukoglu, E. (2022). Unsupervised Superpixel Generation using Edge-Sparse Embedding. arXiv preprint [arXiv:2211.15474](https://arxiv.org/abs/2211.15474).

Please install following packages before usage.
```bat
conda create --name DeepDecoder python=3.8.5
conda activate DeepDecoder
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install scikit-image scikit-learn matplotlib
pip install opencv-python
```

[demo.py](demo.py) includes an example usage.

[FittingConfiguration.py](FittingConfiguration.py) includes all configuration parameters.

[fitting](fitting) includes all the code related to the fitting procedure of the deep decoders.

[models](models) includes all the code to generate the Deep Decoder in pytorch.

[utils](utils) includes a few image processing/loading functions and functions related to the clustering step.