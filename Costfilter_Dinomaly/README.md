# CostFilter-AD Enhances the Baseline Dinomaly

## Benchmark Results

### MVTec-AD

| Method   | Image AUROC    | Image AP       | Image F1max    | Pixel AUROC    | Pixel AP       | Pixel F1max    | AUPRO          |
| -------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| UniAD    | 97.5           | 99.1           | 97.0           | 96.9           | 44.5           | 50.5           | 90.6           |
| + Ours   | 99.0           | 99.7           | 98.1           | 97.5           | 60.5           | 59.9           | 91.3           |
| Dinomaly | **99.6** | **99.8** | 99.0           | 98.3           | 68.7           | 68.7           | 94.6           |
| + Ours   | **99.6** | **99.8** | **99.1** | **98.7** | **75.6** | **72.9** | **95.6** |

### VisA

| Method   | Image AUROC    | Image AP       | Image F1max    | Pixel AUROC    | Pixel AP       | Pixel F1max    | AUPRO          |
| -------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| UniAD    | 91.5           | 93.6           | 88.5           | 98.0           | 32.7           | 38.4           | 76.1           |
| + Ours   | 92.1           | 94.0           | 88.9           | 98.6           | 34.0           | 39.0           | 86.4           |
| Dinomaly | 98.7           | **98.9** | 96.1           | 98.7           | 52.5           | 55.4           | **94.5** |
| + Ours   | **98.8** | 98.8           | **96.5** | **98.9** | **59.9** | **59.9** | 94.4           |

### MPDD

| Method    | Image AUROC    | Image AP       | Image F1max    | Pixel AUROC    | Pixel AP       | Pixel F1max    | AUPRO          |
| --------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| HVQ-Trans | 86.5           | 87.9           | 85.6           | 96.9           | 26.4           | 30.5           | 88.0           |
| + Ours    | 93.1           | 95.4           | 90.3           | 97.5           | 34.1           | 37.0           | 82.9           |
| Dinomaly  | 97.2           | 98.4           | **96.0** | **99.1** | 59.5           | **59.4** | **96.6** |
| + Ours    | **97.4** | **98.5** | **96.0** | **99.1** | **59.6** | 59.3           | **96.6** |

### BTAD

| Method    | Image AUROC    | Image AP       | Image F1max    | Pixel AUROC    | Pixel AP       | Pixel F1max    | AUPRO          |
| --------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| HVQ-Trans | 90.9           | 97.8           | 94.8           | 96.7           | 43.2           | 48.7           | 75.6           |
| + Ours    | 93.3           | 98.6           | 96.0           | 97.3           | 47.0           | 50.2           | 76.2           |
| Dinomaly  | 95.4           | 98.5           | 95.5           | 97.9           | 70.1           | 68.0           | 76.5           |
| + Ours    | **96.2** | **98.9** | **96.3** | **98.2** | **74.8** | **70.0** | **81.0** |

## **Note:**

CostFilter-AD yields comparable or greater gains when integrated with other baselines such as [GLAD](https://github.com/hyao1/GLAD/tree/main) (ECCV'24), [UniAD](https://github.com/zhiyuanyou/UniAD) (NeurIPS'22), [HVQ-Trans](https://github.com/RuiyingLu/HVQ-Trans) (NeurIPS'23), and [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (WACV'25). Codes are coming.

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n cosfilterad python=3.8.20
conda activate cosfilterad 
pip install -r requirements.txt```
# This environment is fully compatible with CostFilter-AD + one of five  baselines discussed in this paper.
```

```

```

Experiments can be conducted on A100 (80GB or 40GB) and GV100 (32GB); results are consistent across GPUs.

## 2. Dataset Preparation

Modify the `data_path` in code as needed to point to your local dataset.

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

### 🔹 Dino Model Weights

|         Description         | Download Link                                                                                     | Save Path       |
| :-------------------------: | ------------------------------------------------------------------------------------------------- | --------------- |
| dinov2_vitb14_reg4_pretrain | [Google Drive](https://drive.google.com/drive/folders/1nhb-mAO_0eyl3pY6-vqlBLJK2rRXoR17?usp=sharing) | `./backbones` |

---

### 🔹 Dinomaly Baseline Weights

| Dataset | Download Link                                                                                             | Save Path                 |
| :------: | --------------------------------------------------------------------------------------------------------- | ------------------------- |
| MVTec-AD | [Dinomaly on MVTec-AD](https://drive.google.com/drive/folders/1Obnx0wCHMWu8lE8M9pXhQOuduJol2kPJ?usp=sharing) | `./saved_results_mvtec` |
|   VisA   | [Dinomaly on VisA](https://drive.google.com/drive/folders/1COemGeV62HUZ9P4E1JDyF0-x0a0REi7-?usp=sharing)     | `./saved_results_visa`  |
|   MPDD   | [Dinomaly on MPDD](https://drive.google.com/drive/folders/17dTiCXsW0zghsHBORz2HzvvZQgGDfFnP?usp=sharing)     | `./saved_results_mpdd`  |
|   BTAD   | [Dinomaly on BTAD](https://drive.google.com/drive/folders/1BBdy_imjjN_T8y55cjE5CHxIehAn25F9?usp=sharing)     | `./saved_results_btad`  |

---

### 🔹 [CostFilter-AD Weights (Ours)](https://drive.google.com/drive/folders/1uFSY8O-nQ_Ji4pT5lNM0CTuI0ZB7L9Ql?usp=sharing)

| Dataset | Download Link                                                                                                 | Save Path                            |
| :------: | ------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| MVTec-AD | [CostFilterAD on MVTec-AD](https://drive.google.com/drive/folders/1tMVlel5UMhYzQFT-E7_FZJapIMHyCdNz?usp=sharing) | `./checkpoint_paths/mvtecad0.0001` |
|   VisA   | [CostFilterAD on VisA](https://drive.google.com/drive/folders/1NzjZPYlLExfVht5fAuoU9f8RlXuPDUnk?usp=sharing)     | `./checkpoint_paths/visaad0.0001`  |
|   MPDD   | [CostFilterAD on MPDD](https://drive.google.com/drive/folders/1xZSTo8f-C6JUc2g909BFIzZtMYbXZEM3?usp=sharing)     | `./checkpoint_paths/mpddad0.0001`  |
|   BTAD   | [CostFilterAD on BTAD](https://drive.google.com/drive/folders/1QIdzCuivQvxwHci-A-cFIyr79L8h2Tce?usp=sharing)     | `./checkpoint_paths/btadad0.0001`  |

## 4. Run Experiments

We study the Multi-Class anomaly detection.

### **🔹 Testing (using pretrained weights)**

**MVTec-AD:**

```
cd /CostFilterAD/Costfilter_Dinomaly
python costfilter_dinomaly_mvtec_uni_my_train_test.py
```

**VisA**

```
python costfilter_dinomaly_visa_uni_my_train_test.py
```

**MPDD**

```
python costfilter_dinomaly_mpdd_uni_my_train_test.py
```

**BTAD**

```
python costfilter_dinomaly_btad_uni_my_train_test.py
```

### **🔹 Train: you can train the cost filtering model as follows.**

**MVTec-AD:**

```
cd /CostFilterAD/Costfilter_Dinomaly
python costfilter_dinomaly_mvtec_uni_my_train_test.py --train
```

**VisA**

```
python costfilter_dinomaly_visa_uni_my_train_test.py --train
```

**MPDD**

```
python costfilter_dinomaly_mpdd_uni_my_train_test.py --train
```

**BTAD**

```
python costfilter_dinomaly_btad_uni_my_train_test.py --train
```

## 📚 Cite

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
