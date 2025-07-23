# CostFilter-AD Enhances the Baseline [HVQ-Trans](https://github.com/RuiyingLu/HVQ-Trans)

## Benchmark Results

### MVTec-AD

|  Method   | Image AUROC | Image AP | Image F1max | Pixel AUROC | Pixel AP | Pixel F1max | AUPRO    |
| :-------: | ----------- | -------- | ----------- | ----------- | -------- | ----------- | -------- |
| HVQ-Trans | 97.9        | 99.3     | 97.4        | 97.4        | 49.4     | 54.3        | 91.5     |
|  + Ours   | **99.0**    | **99.7** | **98.6**    | **98.0**    | **58.1** | **61.2**    | **93.3** |

### VisA

| Method    | Image AUROC | Image AP | Image F1max | Pixel AUROC | Pixel AP | Pixel F1max | AUPRO    |
| --------- | ----------- | -------- | ----------- | ----------- | -------- | ----------- | -------- |
| HVQ-Trans | 91.5        | 93.4     | 88.1        | 98.5        | 35.5     | 39.6        | 86.4     |
| + Ours    | **98.6**    | **95.3** | **89.3**    | **98.6**    | **41.4** | **45.0**    | **86.8** |

### MPDD

| Method    | Image AUROC | Image AP | Image F1max | Pixel AUROC | Pixel AP | Pixel F1max | AUPRO    |
| --------- | ----------- | -------- | ----------- | ----------- | -------- | ----------- | -------- |
| HVQ-Trans | 86.5        | 87.9     | 85.6        | 96.9        | 26.4     | 30.5        | **88.0** |
| + Ours    | **93.1**    | **95.4** | **90.3**    | **97.5**    | **34.1** | **37.0**    | 82.9     |

### BTAD

| Method    | Image AUROC | Image AP | Image F1max | Pixel AUROC | Pixel AP | Pixel F1max | AUPRO    |
| --------- | ----------- | -------- | ----------- | ----------- | -------- | ----------- | -------- |
| HVQ-Trans | 90.9        | 97.8     | 94.8        | 96.7        | 43.2     | 48.7        | **75.6** |
| + Ours    | **93.4**    | **98.6** | **96.0**    | **97.3**    | **47.0** | **50.2**    | 74.4     |



## ⭐⭐ **Flexible Extensibility**

This paper proposes a plug-in method. However, as long as you have the following:

1. If you have **train/test images** and either **reconstructed normal images** (in reconstruction-based methods) or randomly selected **normal train template images** (in embedding-based methods), you can extract **features** using any feature extractor such as DINO, SAM, or ViT;

**or**

2. If you already have **features** of the **train/test images** and those of the **reconstructed/normal template images**,

then you can perform global matching (e.g., cosine similarity) or local matching (e.g., L2 loss) to obtain the matching results (referred to as the cost volume in this paper). 

These results can be then fed into our proposed cost filtering network to denoise the matching output. 

**Extensive experiments demonstrate the general effectiveness of our cost filtering method.**



## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n cosfilterad python=3.8.20
conda activate cosfilterad 
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# This environment is also compatible with CostFilter-AD + other baselines
```

Experiments can be conducted on A100 (80GB or 40GB) and GV100 (32GB); results are consistent across GPUs.

## 2. Dataset Preparation

Modify the `data_path` in code as needed to point to your local dataset.


###  DTD dataset （Describable Textures Dataset） 
DTD dataset is essential！
Firstly, please download the [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and place it under your custom `data_dtd_path`.
Then, edit line anomaly_source_paths = sorted(...) of  `Costfilter_HVQtrans/Costfilter_HVQ_xxxx/datasets/custom_dataset.py` as follows:

```python
anomaly_source_paths = sorted(glob.glob("/your/data_dtd_path/images" + "/*/*.jpg")) 

```

> Replace `"/your/data_dtd_path"` with the actual directory where DTD is located.

Costfilter_HVQ_xxxx denotes Costfilter_HVQ_mvtec, Costfilter_HVQ_visa, etc.

### MVTec AD

Download from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip to: `./hdd/Datasets/MVTec-AD/`

```
|-- mvtec_anomaly_detection
    |-- bottle
	|-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
    |-- cable
    |-- ....
```

### VisA

Download [URL](https://github.com/amazon-science/spot-diff).
Unzip to `./hdd/Datasets/VisA/`. Preprocess the dataset to `./hdd/Datasets/VisA_pytorch/` in 1-class mode by their official splitting
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```

`./hdd/Datasets/VisA_pytorch/` will be like:

```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```

### MPDD and BTAD

Download the MPDD dataset from [URL](https://github.com/stepanje/MPDD). Unzip to `./hdd/Datasets/MPDD/`.

Download the BTAD dataset from [URL](https://github.com/pankajmishra000/VT-ADL). Unzip to `./hdd/Datasets/BTAD1/`.

General structure for all datasets:

```
|-- Name_of_Dataset
    |-- Category1
	|-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
    |-- Category2
    |-- ....
```

## 3. Checkpoints

---

### 🔹 HVQ-Trans Baseline Weights

| Dataset  | Download Link                                                                                                    | Save Path                                                  |
| :------: | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| MVTec-AD | [HVQ-Trans on MVTec-AD](https://drive.google.com/drive/folders/1hIXu-szb7MpkZsxGTWaqd3A_fvJVuatk?usp=drive_link) | `./Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoints/` |
|   VisA   | [HVQ-Trans on VisA](https://drive.google.com/drive/folders/1ziM-4a1aNSiGEGBGB5jYpqfPJlUJDi1q?usp=drive_link)     | `./Costfilter_HVQ_visa/experiments/VISA/checkpoints/`      |
|   MPDD   | [HVQ-Trans on MPDD](https://drive.google.com/drive/folders/1ZB7wRiXxEAfA_D9ND1_llATJ4Dwc5Stl?usp=drive_link)     | `./Costfilter_HVQ_mpdd/experiments/MVTec-AD/checkpoints/`  |
|   BTAD   | [HVQ-Trans on BTAD](https://drive.google.com/drive/folders/1GARfapkAgPd48ZQUAbVzml56r3jbknNL?usp=drive_link)     | `./Costfilter_HVQ_btad/experiments/MVTec-AD/checkpoints/`  |

---

### 🔹 [CostFilter-AD Weights (Ours)](https://drive.google.com/drive/folders/1GtSaI93i_ZxCre83kqcgu85TxMxvsHwf?usp=sharing)

| Dataset  | Download Link                                                                                                        | Save Path                                                       |
| :------: | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| MVTec-AD | [CostFilterAD-HVQ on MVTec-AD](https://drive.google.com/drive/folders/1vp-Sp0LUFsRYHlz2KlQnQ03gh6DyZRO8?usp=sharing) | `./Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoint_paths/` |
|   VisA   | [CostFilterAD-HVQ on VisA](https://drive.google.com/drive/folders/1HBblSDN25vj_x8PVSIOdiXwZUJZbWnlu?usp=drive_link)  | `./Costfilter_HVQ_visa/experiments/VISA/checkpoint_paths/`      |
|   MPDD   | [CostFilterAD-HVQ on MPDD](https://drive.google.com/drive/folders/1NXy18oGZVYx0TubiZQwneM6WEM9ThWPc?usp=drive_link)  | `./Costfilter_HVQ_mpdd/experiments/MVTec-AD/checkpoint_paths/`  |
|   BTAD   | [CostFilterAD-HVQ on BTAD](https://drive.google.com/drive/folders/1qsMpD47XuaSx1Tjf1f8UOQmU_tGOyEhJ?usp=drive_link)  | `./Costfilter_HVQ_btad/experiments/MVTec-AD/checkpoint_paths/`  |

## 4. Run Experiments

We study the Multi-Class anomaly detection.

### **🔹 Testing (using pretrained weights)**

**MVTec-AD:**

```
cd /CostFilterAD/Costfilter_HVQ_trans/Costfilter_HVQ_mvtec/experiments/MVTec-AD
bash ./eval.sh
```

**VisA**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_visa/experiments/VISA
bash ./eval.sh
```

**MPDD**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_mpdd/experiments/MVTec-AD
bash ./eva_mpdd.sh
```

**BTAD**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_btad/experiments/MVTec-AD
bash ./eva_btad.sh
```

### **🔹 Train: you can train the cost filtering model as follows.**

**MVTec-AD:**

```
cd /CostFilterAD/Costfilter_HVQ_trans/Costfilter_HVQ_mvtec/experiments/MVTec-AD
bash ./train.sh
```

**VisA**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_visa/experiments/VISA
bash ./train.sh
```

**MPDD**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_mpdd/experiments/MVTec-AD
bash ./train_mpdd.sh
```

**BTAD**

```
cd /CostFilter-AD-main/Costfilter_HVQ_trans/Costfilter_HVQ_btad/experiments/MVTec-AD
bash ./train_btad.sh
```

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
