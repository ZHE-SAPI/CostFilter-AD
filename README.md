# CostFilter-AD

[ICML2025] Official Implementation of CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering

| [📄 Paper](https://arxiv.org/abs/2505.01476)              | [📑 Slide](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/Materials/CostFilter-AD_slide_ICML2025.pdf)                                     | [🖼️ Poster](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/Materials/CostFilter-AD_poster_ICML2025.pdf) |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [🌐 ICML Site](https://icml.cc/virtual/2025/poster/46359) | [🔍 OpenReview](https://openreview.net/forum?id=6p2wsBeYSs&referrer=%5Bthe%20profile%20of%20Mingxiu%20Cai%5D(%2Fprofile%3Fid%3D~Mingxiu_Cai1)) |                                                                                                             |

### **Update (07/10/2025):**

We have open-sourced  **[CostFilter-AD-HVQ-Trans](https://github.com/ZHE-SAPI/CostFilter-AD/tree/main/Costfilter_HVQtrans)** , built upon the [HVQ-Trans](https://github.com/RuiyingLu/HVQ-Trans) (NeurIPS '23) baseline.

It also delivers significant gains on  **MVTec-AD** ,  **VisA** ,  **MPDD** , and  **BTAD** .

👉 Please see the [`Costfilter_HVQtrans`](https://github.com/ZHE-SAPI/CostFilter-AD/tree/main/Costfilter_HVQtrans) subfolder for code and details.

### **Update (07/06/2025):**

We have open-sourced  **[CostFilter-AD-Dinomaly](https://github.com/ZHE-SAPI/CostFilter-AD/tree/main/Costfilter_Dinomaly)** , built upon the [Dinomaly](https://github.com/guojiajeremy/Dinomaly/tree/master) (CVPR '25) baseline.

It delivers significant gains on  **MVTec-AD** ,  **VisA** ,  **MPDD** , and  **BTAD** .

👉 Please see the [`Costfilter_Dinomaly`](https://github.com/ZHE-SAPI/CostFilter-AD/tree/main/Costfilter_Dinomaly) subfolder for code and details.

# **Abstract:**

Unsupervised anomaly detection (UAD) seeks to localize the anomaly mask of an input image with respect to normal samples. Either by reconstructing normal counterparts (reconstruction-based) or by learning an image feature embedding space (embedding-based), existing approaches fundamentally rely on image-level or feature-level matching to derive anomaly scores. Often, such a matching process is inaccurate yet overlooked, leading to sub-optimal detection. To address this issue, we introduce the concept of cost filtering, borrowed from classical matching tasks, such as depth and flow estimation, into the UAD problem. We call this approach CostFilter-AD.
Specifically, we first construct a matching cost volume between the input and normal samples, comprising two spatial dimensions and one matching dimension that encodes potential matches. To refine this, we propose a cost volume filtering network, guided by the input observation as an attention query across multiple feature layers, which effectively suppresses matching noise while preserving edge structures and capturing subtle anomalies.
Designed as a generic post-processing plug-in,
CostFilter-AD can be integrated with either reconstruction-based or embedding-based methods.
Extensive experiments on MVTec-AD and VisA benchmarks validate the generic benefits of CostFilter-AD for both single- and multi-class UAD tasks. Code and models will be released.

# Motivation:

![Motivation](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/Materials/motivation.png)
**Comparison of Multi-class UAD Results.** We present the visualization results and kernel density estimation curves (Parzen, 1962) of image- and pixel-level logits. Baseline results are highlighted in yellow, while ours are shown in green. Our model achieves superior performance by detecting anomalies with less noise and offering a clearer separation between normal and abnormal logits.

# Overview:

![Overview](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/Materials/overview.png)
**Overview of CostFilter-AD.** We reformulate UAD as a matching cost filtering process:
**(i)** First, we employ a pre-trained encoder to extract features from both the input image and the templates (reconstructed normal images or randomly selected normal samples).
**(ii)** Second, we construct an anomaly cost volume based on global similarity matching.
**(iii)** Thirdly, we learn a cost volume filtering network, guided by attention queries derived from the input features and an initial anomaly map, to refine the volume and generate the final detection results.
**(iv)** Further, we integrate a class-aware adaptor to tackle class imbalance and enhance the ability to deal with multiple anomaly classes simultaneously.

# Environments

Create a new conda environment and install required packages.

```
conda create -n cosfilterad python=3.8.20
conda activate cosfilterad 
pip install -r requirements.txt
# This environment is fully compatible with CostFilter-AD + one of five  baselines discussed in this paper.
```

Experiments can be conducted on A100 (80GB or 40GB) and GV100 (32GB); results are consistent across GPUs.

# Welcome to discuss

🚀 We are very happy to release the codes and model weights.

💬 We are currently finalizing them, so please stay tuned and feel free to discuss!

# 📚 Cite


## 📚 Cite

If you find our work helpful, please consider **citing our paper** and giving us a ⭐. Thank you!

```
  @inproceedings{zhang2025costfilter,
  author    = {Zhang, Zhe and Cai, Mingxiu and Wang, Hanxiao and Wu, Gaochang and Chai, Tianyou and Zhu, Xiatian},
  title     = {CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering},
  booktitle = {42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  month     = {July},
  location  = {Vancouver, Canada},
  note      = {arXiv preprint arXiv:2505.01476}
}
```
