# Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection


This repository contains the code for the paper "Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection" by Ruiying Lu · YuJie Wu · Long Tian · Dongsheng Wang · Bo Chen · Xiyang Liu · Ruimin Hu.

## Training
Begin your training from '\experiments\MVTec-AD\train.sh'

## Testing
Using '\experiments\MVTec-AD\eval.sh' to evaluate the results.

## Noting
We have also released our well-trained checkpoint on MVTec at: https://pan.baidu.com/s/1vPWwcWLAHINmE8_RsxW9Hw?pwd=2PME, Extract code：2PME. It is free to download the checkpoint and put it at "\experiments\MVTec-AD\checkpoints\HVQ_TR_switch\best_\"

"models.HVQ_TR_switch_OT" refers to the model with OT strategy and "models.HVQ_TR_switch" refers to the model without OT strategy. We have offered both choices for users. Note that the SOTA performances in our paper are abtained with "models.HVQ_TR_switch_OT".

## Welcome to discuss with us (luruiying@xidian.edu.cn) and cite our paper:

Lu R, Wu Y J, Tian L, et al. Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection[C]//Thirty-seventh Conference on Neural Information Processing Systems. 2023.
