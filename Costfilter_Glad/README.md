# CostFilterAD + GLAD Integration
This is the core code of CostFilterAD + [GLAD](https://github.com/hyao1/GLAD/tree/main) Integration. 

The key is line 544-612 for anomaly volume construction and line 615-773 for anomaly volume filtering of [main_multi_glad_dino_3d.py](https://github.com/ZHE-SAPI/CostFilter-AD/blob/main/Costfilter_Glad/Costfilter_Glad_Mvtecad/main_multi_glad_dino_3d.py)


The complete code can be released at the end of August. Thank you.

##  Setup

Navigate to the working directory before running any scripts:

```bash
cd your_path/anomaly/CostFilterAD/Costfilter_Glad/Costfilter_Glad_Mvtecad
```

---

##  Training

To train the integrated model:

```bash
nohup bash train_multi_dino_3d.sh > train_costfilter_glad_mvtecad.log 2>&1 &
```

---

##  Testing

To evaluate the model:

```bash
nohup bash test_multi_dino_gauss_3d_qianghua.sh > test_costfilter_glad_mvtecad.log 2>&1 &
```


---
