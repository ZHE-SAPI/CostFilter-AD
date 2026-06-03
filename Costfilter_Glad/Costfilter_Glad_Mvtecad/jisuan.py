import re
import numpy as np

# 原始文本（请替换为你自己的字符串）
text = """
Model_qian: /home/customer/Desktop/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume/models_grid_test_bs16_triloss_gai_anomalydino_num0_rand4/epoch_28_0_fscratch_anomalydino_num0_rand4.pth
Class: hazelnut
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9958
P-AP: 0.8543
P-F1-max: 0.7563
PRO: 0.9656

Class: screw
I-AUROC: 0.9798
I-AP: 0.9911
I-F1-max: 0.9694
P-AUROC: 0.9932
P-AP: 0.5425
P-F1-max: 0.5249
PRO: 0.9614

Class: capsule
I-AUROC: 0.9856
I-AP: 0.9969
I-F1-max: 0.977
P-AUROC: 0.9923
P-AP: 0.5324
P-F1-max: 0.5472
PRO: 0.9688

Class: cable
I-AUROC: 0.9955
I-AP: 0.9974
I-F1-max: 0.9881
P-AUROC: 0.9945
P-AP: 0.7819
P-F1-max: 0.6976
PRO: 0.9588

Class: transistor
I-AUROC: 0.9954
I-AP: 0.9991
I-F1-max: 0.9686
P-AUROC: 0.9819
P-AP: 0.7485
P-F1-max: 0.6984
PRO: 0.8998

Class: wood
I-AUROC: 0.9963
I-AP: 0.9989
I-F1-max: 0.9836
P-AUROC: 0.9828
P-AP: 0.8585
P-F1-max: 0.774
PRO: 0.9756

Class: grid
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9966
P-AP: 0.6275
P-F1-max: 0.6261
PRO: 0.9799

Class: carpet
I-AUROC: 0.9996
I-AP: 0.9999
I-F1-max: 0.9944
P-AUROC: 0.9964
P-AP: 0.8191
P-F1-max: 0.712
PRO: 0.9842

Class: leather
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9986
P-AP: 0.7098
P-F1-max: 0.6576
PRO: 0.9914

Class: tile
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9956
P-AP: 0.9544
P-F1-max: 0.8768
PRO: 0.9798

Class: bottle
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9915
P-AP: 0.8754
P-F1-max: 0.7898
PRO: 0.9628

Class: metal_nut
I-AUROC: 1.0
I-AP: 1.0
I-F1-max: 1.0
P-AUROC: 0.9937
P-AP: 0.9399
P-F1-max: 0.8854
PRO: 0.9769

Class: pill
I-AUROC: 0.9958
I-AP: 0.9985
I-F1-max: 0.9892
P-AUROC: 0.9855
P-AP: 0.7894
P-F1-max: 0.7303
PRO: 0.9685

Class: toothbrush
I-AUROC: 0.9957
I-AP: 0.9987
I-F1-max: 0.9836
P-AUROC: 0.9946
P-AP: 0.5244
P-F1-max: 0.628
PRO: 0.9698

Class: zipper
I-AUROC: 0.9907
I-AP: 0.9978
I-F1-max: 0.9886
P-AUROC: 0.9697
P-AP: 0.5956
P-F1-max: 0.5499
PRO: 0.8964


AUC Image mean: nan
AP Image mean: nan
AUC Pixel mean: nan
AP Pixel mean: nan
Model_hou_glad: /home/customer/Desktop/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume/models_grid_test_bs16_triloss_gai_anomalydino_num0_rand4/epoch_28_0_fscratch_anomalydino_num0_rand4.pth
Mean Performance_glad: [99.27466667 99.648      98.164      98.73066667 73.586      69.212
 95.514     ]



"""

# 待提取的指标名
metrics = ["I-AUROC", "I-AP", "I-F1-max", "P-AUROC", "P-AP", "P-F1-max", "PRO"]

# 用于存储每个指标的值
results = {metric: [] for metric in metrics}

# 提取每类的指标
for metric in metrics:
    pattern = rf"{metric}:\s*([0-9.]+)"
    matches = re.findall(pattern, text)
    values = list(map(float, matches))
    results[metric].extend(values)

# 计算平均值
mean_results = {metric: np.mean(vals) for metric, vals in results.items()}

# 打印结果
print("Mean Performance Across Classes:")
for metric, mean_val in mean_results.items():
    print(f"{metric}: {mean_val*100:.8f}")


# I-AUROC: 99.56266667
# I-AP: 99.85533333
# I-F1-max: 98.95000000
# P-AUROC: 99.08466667
# P-AP: 74.35733333
# P-F1-max: 69.69533333
# PRO: 96.26466667
