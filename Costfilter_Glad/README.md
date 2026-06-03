# CostFilter-AD Plug-in for GLAD

This repository provides the CostFilter-AD plug-in implementation for GLAD.

CostFilter-AD is trained as a lightweight filtering network on top of GLAD. Running GLAD diffusion inference again during every training epoch would be inefficient. Therefore, we pre-compute the GLAD reconstruction features for the training and test images and save them as `.pt` files. During training and evaluation, CostFilter-AD directly loads these saved `.pt` files.

## 1. Pre-computed GLAD Features

Download the pre-computed `.pt` feature archives from Quark Drive:

```text
Link: https://pan.quark.cn/s/8e28955c3012?pwd=wBi3
Password: wBi3
```

| Dataset | Split | Archive files | Extract to |
|---|---|---|---|
| MVTec | Train | `train_save_path_Glad_mvtecad1.zip`<br>`train_save_path_Glad_mvtecad2.zip` | `./Costfilter_Glad_Mvtecad/train_save_path_Glad_mvtecad/` |
| MVTec | Test | `test_save_path_Glad_mvtecad.zip` | `./Costfilter_Glad_Mvtecad/test_save_path/` |
| VisA | Train | `train_save_path_1234_VlSA1.zip`<br>`train_save_path_1234_VlSA2.zip` | `./Costfilter_Glad_VisA/train_save_path_1234_VlSA/` |
| VisA | Test | `test_save_path_1234_VlSA.zip` | `./Costfilter_Glad_VisA/test_save_path_1234_VlSA/` |

For the training splits, extract both ZIP files into the same target directory.

## 2. Pre-trained Weights

Download the pre-trained weights from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1ujsll7yinB-HaJMT0G-mzw?pwd=pjk5
Password: pjk5
```

| Folder or file | Role | Place to |
|---|---|---|
| `checkpoints_path_mvtec/` | CostFilter-AD weights for MVTec | `./Costfilter_Glad_Mvtecad/checkpoints_path/` |
| `checkpoints_path_visa/` | CostFilter-AD weights for VisA | `./Costfilter_Glad_VisA/checkpoints_path/` |
| `CompVis/` | GLAD open-source weights | `./Costfilter_Glad_Mvtecad/` and `./Costfilter_Glad_VisA/` |
| `model/` | GLAD open-source weights | `./Costfilter_Glad_Mvtecad/`  |
| `model_Glad/` | GLAD open-source weights | `./Costfilter_Glad_VisA/` |

`checkpoints_path` contains the `.pth` weights trained by CostFilter-AD. `CompVis`, `model`, and `model_Glad` contain the GLAD-related open-source weights. Since CostFilter-AD is a plug-in method for GLAD, all the above weights are required.

## 3. Expected Directory Layout

After downloading and extracting the files, the directory layout should be similar to the following:

```text
Costfilter_Glad_Mvtecad/
├── checkpoints_path/
├── CompVis/
├── model/
├── train_save_path_Glad_mvtecad/
└── test_save_path/

Costfilter_Glad_VisA/
├── checkpoints_path/
├── CompVis/
├── model_Glad/
├── train_save_path_1234_VlSA/
└── test_save_path_1234_VlSA/
```

Please keep the folder names consistent with the paths used in the training and testing scripts.

Since we provide the pre-trained `.pth` weights, you can directly run the testing scripts after placing the required feature files and weights in the directories above.

## 4. Testing and Training
```bash
conda activate cosfilterad 
```

### MVTec

```bash
cd xxxx/Costfilter_Glad_Mvtecad
```

Test:

```bash
bash test_multi_dino_gauss_3d_qianghua.sh
```

Train:

```bash
bash train_multi_dino_3d.sh
```

### VisA

```bash
cd xxxx/Costfilter_Glad_VisA
```

Test:

```bash
bash test_multi_dino_gauss_3d_qianghua.sh
```

Train:

```bash
bash train_multi_dino_3d.sh
```


