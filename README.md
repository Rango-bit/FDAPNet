# FDAPNet
This repo is an implementation of the following paper: A Frequency-Domain Attention Pruning Model for Segmentation and Classification of Hepatic Echinococcosis in Ultrasound Images. The paper is currently under review at IEEE Journal of Biomedical and Health Informatics. We first open-sourced the FDAPNet model code and will release the full project code after the paper is accepted.

The detailed structure of the FDAPNet model is shown in the following figure:
<p align="center">
  <img src="model/figs/segment_model.png" width="700"/>
</p>

The detailed structure of the classification part is shown as follows:
<p align="center">
  <img src="model/figs/classify_model.png" width="500"/>
</p>

The segmentation performance of FDAPNet on ultrasound images of hepatic encopresis is as follows:
<p align="center">
  <img src="model/figs/segment_results.jpg" width="600"/>
</p>

The following shows the classification performance of FDAPNet on ultrasound images of hepatic encopresis:
<p align="center">
  <img src="model/figs/classify_results.jpg" width="500"/>
</p>

# Requirements
+ CUDA/CUDNN
+ pytorch>=1.10.1
+ torchvision>=0.11.2
