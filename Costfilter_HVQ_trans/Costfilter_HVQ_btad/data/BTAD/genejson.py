import os
import json

# # 数据集根目录
# dataset_root = "/home/ZZ/anomaly/GLAD-main/hdd/Datasets/BTAD"

# # 输出 JSON 文件路径
# train_json_path = "/home/ZZ/anomaly/HVQ-Trans-master/data/BTAD/train.json"
# test_json_path = "/home/ZZ/anomaly/HVQ-Trans-master/data/BTAD/test.json"

# def generate_json_for_all_categories():
#     train_data = []
#     test_data = []

#     # 遍历所有类别文件夹
#     for clsname in os.listdir(dataset_root):
#         cls_path = os.path.join(dataset_root, clsname)
#         if not os.path.isdir(cls_path):
#             continue

#         # 处理 train/good 文件夹
#         train_good_path = os.path.join(cls_path, "train", "ok")
#         if os.path.exists(train_good_path):
#             for filename in os.listdir(train_good_path):
#                 train_data.append({
#                     "filename": f"{clsname}/train/ok/{filename}",
#                     "label": 0,  # good 类别
#                     "label_name": "good",
#                     "clsname": clsname
#                 })

#         # 处理 test 文件夹
#         test_path = os.path.join(cls_path, "test")
#         ground_truth_path = os.path.join(cls_path, "ground_truth")
#         if os.path.exists(test_path):
#             for category in os.listdir(test_path):
#                 category_path = os.path.join(test_path, category)
#                 if not os.path.isdir(category_path):
#                     continue

#                 # 遍历每个类别（good, combined, rough, ...）
#                 for filename in os.listdir(category_path):
#                     entry = {
#                         "filename": f"{clsname}/test/{category}/{filename}",
#                         "label": 0 if category == "ok" else 1,  # ok 为正常类别，其他为异常类别
#                         "label_name": category,  # 使用具体的类别名称
#                         "clsname": clsname
#                     }

#                     # 如果是异常类别，添加对应的掩码路径
#                     if category != "ok":
#                         maskname = filename.replace(".png", "_mask.png")  # 根据规则生成掩码文件名
#                         entry["maskname"] = f"{clsname}/ground_truth/{category}/{maskname}"

#                     test_data.append(entry)

#     # 保存 train.json
#     with open(train_json_path, "w") as train_file:
#         for entry in train_data:
#             train_file.write(json.dumps(entry) + "\n")

#     # 保存 test.json
#     with open(test_json_path, "w") as test_file:
#         for entry in test_data:
#             test_file.write(json.dumps(entry) + "\n")

#     print(f"train.json and test.json for all categories generated successfully!")

# # 执行生成过程
# generate_json_for_all_categories()















import os
import json

# 数据集根目录
dataset_root = "/home/ZZ/anomaly/GLAD-main/hdd/Datasets/BTAD"


test_json_path = "/home/ZZ/anomaly/HVQ-Trans-master/data/BTAD/test.json"



def generate_test_json():
    test_data = []

    # 遍历所有类别文件夹
    for clsname in os.listdir(dataset_root):
        cls_path = os.path.join(dataset_root, clsname)
        if not os.path.isdir(cls_path):
            continue

        # 处理 test 文件夹
        test_path = os.path.join(cls_path, "test")
        ground_truth_path = os.path.join(cls_path, "ground_truth")
        if os.path.exists(test_path):
            for category in os.listdir(test_path):
                category_path = os.path.join(test_path, category)
                if not os.path.isdir(category_path):
                    continue

                # 遍历每个类别（good, hole, combined, rough, ...）
                for filename in os.listdir(category_path):
                    entry = {
                        "filename": f"{clsname}/test/{category}/{filename}",
                        "label": 0 if category == "ok" else 1,
                        "label_name": "good" if category == "ok" else "defective",
                        "clsname": clsname
                    }

                    # 如果是异常类别，添加对应的掩码路径
                    if category != "ok":
                        maskname = filename.replace(".JPG", ".png").replace(".jpg", ".png").replace(".bmp", ".png")  # 根据规则生成掩码文件名
                        entry["maskname"] = f"{clsname}/ground_truth/{category}/{maskname}"

                    test_data.append(entry)

    # 保存 test.json
    with open(test_json_path, "w") as test_file:
        for entry in test_data:
            test_file.write(json.dumps(entry) + "\n")

    print(f"test.json for all categories generated successfully!")

# 执行生成过程
generate_test_json()
