# FDAPNet
This repo is an implementation of the following paper: Frequency-domain attention pruning network for recognizing hepatic echinococcosis in ultrasound images. The paper is currently under review at Information Sciences.

The detailed structure of the FDAPNet model is shown in the following figure:
<p align="center">
  <img src="model/figs/segment_model.png" width="700"/>
</p>

The detailed structure of the classification part is shown as follows:
<p align="center">
  <img src="model/figs/classify_model.png" width="500"/>
</p>

# Requirements
+ CUDA/CUDNN
+ pytorch>=1.10.1
+ torchvision>=0.11.2
