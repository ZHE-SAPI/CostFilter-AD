import torch

def remove_optimizer_from_checkpoint(pth_path, output_path=None):
    """
    删除 .pth 文件中的 optimizer_state_dict 并重新保存。
    
    参数：
        pth_path (str): 原始 .pth 文件路径
        output_path (str, optional): 处理后保存的新路径。如果为 None，则覆盖原文件
    """
    # 1. 加载模型字典
    checkpoint = torch.load(pth_path, map_location='cpu')

    # 2. 移除 optimizer_state_dict
    if 'optimizer_state_dict' in checkpoint:
        print("Removing 'optimizer_state_dict'...")
        del checkpoint['optimizer_state_dict']
    else:
        print("No 'optimizer_state_dict' found in checkpoint.")

    # 3. 确定输出路径
    if output_path is None:
        output_path = pth_path  # 覆盖原文件

    # 4. 保存
    torch.save(checkpoint, output_path)
    print(f"Cleaned checkpoint saved to: {output_path}")



remove_optimizer_from_checkpoint(
    "/home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoint_paths/epoch_23_hvq_my_mvtecad.pth",
    "/home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoint_paths/costfilter_hvqtrans_mvtecad.pth"
)
ckpt = torch.load("/home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoint_paths/costfilter_hvqtrans_mvtecad.pth")
print(ckpt.keys())



# cd /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_mvtec/experiments/MVTec-AD/checkpoint_paths

#  python convert_model_pth.py 
