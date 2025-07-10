import torch

def compare_models(path1, path2, key='model_state_dict'):
    model1 = torch.load(path1, map_location='cpu')
    model2 = torch.load(path2, map_location='cpu')

    # 提取 state_dict
    state_dict1 = model1[key] if key in model1 else model1
    state_dict2 = model2[key] if key in model2 else model2

    # 检查键是否一致
    if state_dict1.keys() != state_dict2.keys():
        print("❌ 参数名不一致")
        print("仅存在于 model1 的参数：", set(state_dict1.keys()) - set(state_dict2.keys()))
        print("仅存在于 model2 的参数：", set(state_dict2.keys()) - set(state_dict1.keys()))
        return

    all_same = True
    for name in state_dict1:
        tensor1 = state_dict1[name]
        tensor2 = state_dict2[name]
        if not torch.equal(tensor1, tensor2):
            all_same = False
            diff = torch.norm(tensor1.float() - tensor2.float()).item()
            print(f"⚠️ 参数不同: {name}，L2 差异: {diff:.6f}")
        else:
            # 可选打印一致项
            pass

    if all_same:
        print("✅ 两个模型的所有参数完全一致")
    else:
        print("🔍 模型参数存在差异")

# 示例用法
compare_models('/home/ZZ/anomaly/MY_HVQeriment_mpdd/experiments/MVTec-AD/checkpoint_paths/epoch_5_hvq_my_mpdd.pth', '/home/ZZ/anomaly/MY_HVQeriment_mpdd/experiments/MVTec-AD/checkpoint_paths/epoch_8_hvq_my_mpdd.pth')  # 替换为你的文件路径
