# CostFilter-AD

[ICML2025] Official Implementation of CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering


| [📄 Paper](https://arxiv.org/abs/2505.01476)              | [📑 Slide](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/CostFilter-AD_slide_ICML2025.pdf)                                               | [🖼️ Poster](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/CostFilter-AD_poster_ICML2025.pdf) |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [🌐 ICML Site](https://icml.cc/virtual/2025/poster/46359) | [🔍 OpenReview](https://openreview.net/forum?id=6p2wsBeYSs&referrer=%5Bthe%20profile%20of%20Mingxiu%20Cai%5D(%2Fprofile%3Fid%3D~Mingxiu_Cai1)) |                                                                                                   |

# **Update (07/06/2025):**

We have open-sourced  **CostFilter-AD** , built upon the [Dinomaly](https://github.com/guojiajeremy/Dinomaly/tree/master) (CVPR '25) baseline.

It delivers significant gains on  **MVTec-AD** ,  **VisA** ,  **MPDD** , and  **BTAD** .

👉 Please see the [`Costfilter_Dinomaly`](https://github.com/ZHE-SAPI/CostFilter-AD/tree/main/Costfilter_Dinomaly) subfolder for code and details.


# **Abstract:**

Unsupervised anomaly detection (UAD) seeks to localize the anomaly mask of an input image with respect to normal samples. Either by reconstructing normal counterparts (reconstruction-based) or by learning an image feature embedding space (embedding-based), existing approaches fundamentally rely on image-level or feature-level matching to derive anomaly scores. Often, such a matching process is inaccurate yet overlooked, leading to sub-optimal detection. To address this issue, we introduce the concept of cost filtering, borrowed from classical matching tasks, such as depth and flow estimation, into the UAD problem. We call this approach CostFilter-AD.
Specifically, we first construct a matching cost volume between the input and normal samples, comprising two spatial dimensions and one matching dimension that encodes potential matches. To refine this, we propose a cost volume filtering network, guided by the input observation as an attention query across multiple feature layers, which effectively suppresses matching noise while preserving edge structures and capturing subtle anomalies.
Designed as a generic post-processing plug-in,
CostFilter-AD can be integrated with either reconstruction-based or embedding-based methods.
Extensive experiments on MVTec-AD and VisA benchmarks validate the generic benefits of CostFilter-AD for both single- and multi-class UAD tasks. Code and models will be released.


![ICML8276poster](https://github.com/user-attachments/assets/89ee19e0-f2d3-44ae-9e18-12fa78514414)


# Welcome to discuss

🚀 We are very happy to release the codes and model weights.

💬 We are currently finalizing them, so please stay tuned and feel free to discuss!


# 📚 Cite

If you think our Costfilter-AD helpful, please cite it using the following reference:@inproceedings{zhang2025costfilter,

```
  author    = {Zhang, Zhe and Cai, Mingxiu and Wang, Hanxiao and Wu, Gaochang and Chai, Tianyou and Zhu, Xiatian},
  title     = {CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering},
  booktitle = {42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  month     = {July},
  location  = {Vancouver, Canada},
  note      = {arXiv preprint arXiv:2505.01476}
}
```
