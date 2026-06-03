#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from torch.utils.data import DataLoader
from pipeline import StableDiffusionPipeline
from ddim_scheduling import DDIMScheduler
import copy
import numpy as np
import argparse
import logging
import os
import time
import torch
import torch.utils.checkpoint
from torchvision import transforms, utils
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import torch.optim as optim
import torch.nn as nn

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb

from dataset.dataset_multiclass import MVTecDataset
# from dataset.dataset_multiclass1 import MVTecDataset1
# from dataset.dataset_multiclass2 import MVTecDataset2

from creat_model import model
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from utilize.utilize import normalize, fix_seeds, compute_pro, reconstruction
from tensorboardX import SummaryWriter

from creat_model import get_vit_encoder
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import cv2

import warnings
from loss import FocalLoss, SSIM

# from unet3d_att_dino_channel_test import DiscriminativeSubNetwork_3d_att_dino_channel
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel


from unet2d_att_dino_cost import DiscriminativeSubNetwork_2d_att_dino_cost

from torch.cuda.amp import autocast, GradScaler
from PIL.ImageOps import exif_transpose
from PIL import Image
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import faiss

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

import threading
import faiss

from scipy.ndimage import gaussian_filter

# 使用全局锁和字典管理 FAISS 资源
# faiss_resources_lock = threading.Lock()
faiss_resources = {}

def get_faiss_resources(process_id):
    """
    获取进程安全的 FAISS 资源
    :param process_id: 当前进程 ID，用于区分不同进程的资源
    """
    if process_id not in faiss_resources:
        # print(f"Initializing FAISS resources for process {process_id}")
        res_3 = faiss.StandardGpuResources()
        res_6 = faiss.StandardGpuResources()
        res_9 = faiss.StandardGpuResources()
        res_12 = faiss.StandardGpuResources()
        faiss_resources[process_id] = {
            "res_3": res_3,
            "knn_index_3": faiss.GpuIndexFlatL2(res_3, 768),
            "res_6": res_6,
            "knn_index_6": faiss.GpuIndexFlatL2(res_6, 768),
            "res_9": res_9,
            "knn_index_9": faiss.GpuIndexFlatL2(res_9, 768),
            "res_12": res_12,
            "knn_index_12": faiss.GpuIndexFlatL2(res_12, 768),
        }
    return faiss_resources[process_id]

def worker_init_fn(worker_id):
    """
    初始化 DataLoader worker 时调用，确保每个进程有自己的资源
    """
    process_id = os.getpid()  # 当前进程 ID
    # print(f"[Worker {worker_id}] process_id: {process_id}")
    get_faiss_resources(process_id)  # 确保资源已初始化
    # print(f"Worker {worker_id} initialized in process {process_id}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--train", type=bool)

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="Path to pretrained model or model identifier.", )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose"
                            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."),
                        )

    # dataset setting
    parser.add_argument("--instance_data_dir", type=str, default="/hdd/Datasets/MVTec-AD")
    parser.add_argument("--anomaly_data_dir", type=str, default="/hdd/Datasets/dtd")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--class_name", default="", type=str)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--denoise_step", type=int, default=500)
    parser.add_argument("--min_step", type=int, default=350)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--dino_resolution", type=int, default=512, )
    parser.add_argument("--v", type=int, default=0, )
    parser.add_argument("--input_threshold", type=float, default=0.0, )
    parser.add_argument("--dino_save_path", default=None, type=str)

    parser.add_argument("--inference_step", type=int, default=25, )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size (per device) for sampling images.")

    # train setting
    parser.add_argument("--checkpointing_steps", type=int, default=200, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]'), )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."), )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
                        )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def compute_text_embeddings(prompt, tokenizer, text_encoder):
    with torch.no_grad():
        text_inputs = tokenizer(prompt,
                                truncation=True,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                return_tensors="pt",
                                )
        prompt_embeds = encode_prompt(text_encoder, text_inputs.input_ids, text_encoder.device)
    return prompt_embeds


def encode_prompt(text_encoder, text_inputs_ids, device):
    with torch.no_grad():
        text_encoder.to(device)
        prompt_embeds = text_encoder(
            text_inputs_ids.to(device),
            attention_mask=None,
        )[0]
    return prompt_embeds


def predict_eps(
        alphas_cumprod,
        x_0_anomaly,
        x_0_normal,
        timesteps,
        noise
):
    x_0_anomaly = x_0_anomaly.to(torch.double)
    noise = noise.to(torch.double)
    x_0_normal = x_0_normal.to(torch.double)

    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = alphas_cumprod.to(device=x_0_anomaly.device, dtype=x_0_anomaly.dtype)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(x_0_anomaly.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(x_0_anomaly.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    eps = (sqrt_alpha_prod * x_0_anomaly + sqrt_one_minus_alpha_prod * noise - sqrt_alpha_prod * x_0_normal) / sqrt_one_minus_alpha_prod
    return eps.to(torch.float32)

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

# def weights_init_2d(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.normal_(m.weight, mean=0.0, std=0.02)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.normal_(m.weight, mean=1.0, std=0.02)
#         nn.init.constant_(m.bias, 0)


# def weights_init_3d(m):
#     if isinstance(m, nn.Conv3d):
#         nn.init.normal_(m.weight, mean=0.0, std=0.02)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm3d):
#         nn.init.normal_(m.weight, mean=1.0, std=0.02)
#         nn.init.constant_(m.bias, 0)


def weights_init_2d(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init_3d(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def reverse_normalization(normalized_image):

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)

    # 如果 normalized_image 是 numpy 数组，先转换为 PyTorch 张量
    if isinstance(normalized_image, np.ndarray):
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32)

    # 反向归一化
    original_image = normalized_image * std[None, :, None, None] + mean[None, :, None, None]
    
    return original_image



class CustomDataset_dict_test(Dataset):
    def __init__(self, directory, device, resolution):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        self.device = device
        self.resize = resolution
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 加载.pt文件
        file_path = os.path.join(self.directory, self.files[idx])
        data_dict = torch.load(file_path, map_location='cpu')
        
        # 提取需要的数据
        # latents_all = data_dict["latents_all"].to(self.device)

        image_input = data_dict["batch"]["image_input"]
        anomaly_mask = data_dict["batch"]["anomaly_mask"]
        object_mask = data_dict["batch"]["object_mask"]
        instance_path = data_dict["batch"]["instance_path"]
        instance_label = data_dict["batch"]["instance_label"]

        idx_hou = self.pad_tensor(torch.tensor(data_dict["batch"]["idx_hou"]).squeeze()).unsqueeze(0)
        reconstruct_latent = data_dict["batch"]["reconstruct_latent"]
        # latents_all = [torch.tensor(latent) for latent in data_dict["latents_all"]]
        # for i in range(len(data_dict["latents_all"])):
        #     print(data_dict["latents_all"][i].shape)
        # print(idx_hou.shape)
        latents_all = [torch.tensor(latent) for latent in data_dict["latents_all"]]
        # latents_all = torch.stack([torch.tensor(latent) for latent in data_dict["latents_all"]])
        # print(f"instance_path type: {type(instance_path)}, value: {instance_path}")

        # instance_image = Image.open(data_dict["batch"]["instance_path"][0])
        # instance_image = exif_transpose(instance_image)
        # if not instance_image.mode == "RGB":
        #     instance_image = instance_image.convert("RGB")
        # instance_image = self.transformer_compose(instance_image)
        
        # print('instance_image == image_input', instance_image == image_input)
        instance_image = image_input
        # latents_all = [torch.tensor(latent).to(self.device) for latent in data_dict["latents_all"]]
        

        return latents_all, instance_image, image_input, anomaly_mask, instance_label, instance_path, idx_hou, reconstruct_latent, object_mask

    def pad_tensor(self, tensor, target_length=25):
        # 如果张量的长度小于目标长度
        # print('tensor0', tensor)
        if tensor.size(0) < target_length:
            padding = target_length - tensor.size(0)  # 计算需要填充的长度
            tensor = F.pad(tensor, (padding, 0))  # 在前面填充，后面不填充
        # print('tensor1', tensor)
        return tensor

    def transformer_compose(self, image, mask=None):
        transforms_resize = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.BILINEAR)
        # transforms_center_crop = transforms.CenterCrop(self.img_size)
        # transforms_random_horizontal_flip = T.RandomHFlip()
        # transforms_random_vertical_flip = T.RandomVFlip()
        # transforms_random_rotate1 = T.RandomRotation([0, 90, 180, 270])
        # transforms_random_rotate2 = transforms.RandomRotation(15.0)
        transforms_to_tensor = transforms.ToTensor()
        # mean = (0.48145466, 0.4578275, 0.40821073)
        # std = (0.26862954, 0.26130258, 0.27577711)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transforms_normalize = transforms.Normalize(mean, std)

        image = transforms_resize(image)
        # if self.type == "train":
        #     image, mask = transforms_random_horizontal_flip(image, mask)
        #     image, mask = transforms_random_vertical_flip(image, mask)
        #     image, mask = transforms_random_rotate(image, mask)
        image = transforms_to_tensor(image)
        image = transforms_normalize(image)

        if mask:
            mask = transforms_resize(mask)
            mask = transforms_to_tensor(mask)
            # mask = torch.where(mask > 0, 1.0, 0.0)
            # image_r = transforms_resize(image_r),
            # image_r = transforms_to_tensor(image_r)
            return image, mask
        return image

 
# def get_class_labels(instance_path, device):
#     """
#     根据路径从 CLSNAMES 中匹配类别，并返回类别索引的 Tensor。
    
#     :param instance_path: list 或 iterable，包含图像路径
#     :param CLSNAMES: list，包含类别名称的列表
#     :param clsname_to_index: dict，类别名称到索引的映射
#     :param device: 设备（'cpu' 或 'cuda'），用来指定输出张量所在的设备
#     :return: Tensor，包含每个路径对应的类别索引
#     """
#     # 初始化分类标签
#     cls_labels = []
#     # print('clsname_to_index', clsname_to_index)

#     for path in instance_path:  # 遍历所有路径
#         matched = False  # 标记是否找到匹配类别
#         # print('path', path)
#         for cls_name in CLSNAMES:
#             # print('cls_name', cls_name)

#             if cls_name in path:  # 判断类别是否出现在路径中

#                 cls_labels.append(clsname_to_index[cls_name])
#                 matched = True
#                 break
        
#         if not matched:
#             print(f"Warning: No matching class found for path {path}")
#             cls_labels.append(-1)  # 如果没有匹配类别，用 -1 表示未知类别

#     # 将类别标签转换为 Tensor 并转移到指定设备
#     cls_labels_tensor = torch.tensor(cls_labels, dtype=torch.long).to(device)
    
#     return cls_labels_tensor




 




def augment_image(img_ref, augmentation="rotate", angles = [0, 45, 90, 135, 180, 225, 270, 315]): # 
    """
    Augmentation of images, currently just rotation.
    Supports input as PyTorch tensor with shape (3, H, W).
    """
    imgs = []

    # 检查输入是否为 PyTorch 张量，并转换为 NumPy 格式
    if isinstance(img_ref, torch.Tensor):
        img_ref_np = tensor_to_numpy(img_ref)
    else:
        raise ValueError("Input img_ref must be a PyTorch Tensor with shape (3, H, W).")

    # 执行旋转增强
    if augmentation == "rotate":
        for angle in angles:
            rotated_img = rotate_image(img_ref_np, angle)  # 执行旋转

            rotated_img_tensor = numpy_to_tensor(rotated_img).to(device) # 转回 PyTorch Tensor

            # rotated_img_tensor = numpy_to_tensor(img_ref_np).to(device)  # 转回 PyTorch Tensor
            imgs.append(rotated_img_tensor)
            # print('torch.sum(rotated_img_tensor==img_ref)', torch.sum(rotated_img_tensor==img_ref))
    return imgs


def rotate_image(image, angle):
    """
    Rotate an image using OpenCV.
    Input image should have shape (H, W, C).
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)  # 图像中心点
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return result


def tensor_to_numpy(tensor):
    """
    Converts a PyTorch tensor (3, H, W) to a NumPy array (H, W, C) with uint8 values.
    Tensor values are expected to be in [0.0, 1.0].
    """
    tensor = tensor.cpu().clone()  # 确保在 CPU 上
    # print('tensor_to_numpy', torch.max(tensor))
    # print('tensor_to_numpy', torch.min(tensor))
    # tensor = tensor * 255.0  # 将范围 [0, 1] 转换到 [0, 255]
    # tensor = tensor.permute(1, 2, 0).numpy().astype(np.uint8)  # 转换为 HWC 格式和 uint8 类型
    tensor = tensor.permute(1, 2, 0).numpy() # 这样计算更精确
    return tensor


def numpy_to_tensor(image):
    """
    Converts a NumPy array (H, W, C) with uint8 values to a PyTorch tensor (3, H, W).
    Image values are converted to [0.0, 1.0].
    """
    # image = torch.tensor(image).float() / 255.0  # 转换为 float32 并归一化到 [0, 1]
    image = torch.tensor(image).float() # 转换为 float32 并归一化到 [0, 1]
    image = image.permute(2, 0, 1)  # 转换为 CHW 格式
    # print('numpy_to_tensor', torch.max(image))
    # print('numpy_to_tensor', torch.min(image))
    return image

def compute_dynamic_gamma(cls_out, target, dino_resolution, num_classes=15):
    """
    Compute gamma dynamically for each image based on classification output (cls_out).
    The idea is to increase gamma (focus on hard examples) when classification confidence is low,
    and decrease gamma (focus on easy examples) when confidence is high.
    
    Additionally, if the classification is wrong, increase gamma to focus more on hard examples.
    """
    # cls_out has shape [batch_size, num_classes]

    # Define difficult classes that should always have high gamma (set to 4)
    difficult_classes = ['hazelnut', 'capsule', 'wood', 'cable', 'pill', 'screw', 'transistor', 'zipper']
    
    # Create a mapping of class index to class name (reverse the clsname_to_index mapping)
    clsname_to_index = {name: idx for idx, name in enumerate([
        'hazelnut', 'capsule', 'grid', 'carpet', 'leather', 'tile', 'wood', 'bottle', 
        'cable', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ])}


    batch_size = cls_out.shape[0]
    # print('cls_out.shape', cls_out.shape)
    # Find the predicted class for each sample
    predicted_classes = cls_out.max(dim=1)[1]  # shape: [batch_size]
    
    # Check if the prediction is correct
    correct_prediction = (predicted_classes == target)  # shape: [batch_size]
    # print('correct_prediction', correct_prediction)
    # Calculate gamma based on classification confidence
    # Decrease gamma as classification confidence increases

    # cls_out = torch.softmax(cls_out, dim=1)

    gamma = 3.0 - cls_out.max(dim=1).values  # Max prob from cls_out to determine focus level
    # gamma = gamma.unsqueeze(1).expand(-1, num_classes)  # Expand gamma to the shape [batch_size, num_classes]
    gamma = torch.clamp(gamma, min=1.5)
    # print('gamma0', gamma)

    # Set gamma to 4 for the difficult classes
    for i in range(batch_size):
        predicted_class_idx = predicted_classes[i].item()  # Get the predicted class index for the i-th sample
        predicted_class_name = list(clsname_to_index.keys())[predicted_class_idx]  # Get the class name
        
        # If the predicted class is one of the difficult ones, set gamma to 4
        if predicted_class_name in difficult_classes:
            gamma[i] = 3.5  # Set all gamma values for this sample to 4 (you can customize it per class if needed)
            # print('predicted_class_name', predicted_class_name)

    # print('gamma1', gamma)

    # If the prediction is incorrect, increase gamma for that image
    # For simplicity, let's set gamma to a higher value if the prediction is wrong
    gamma[~correct_prediction] = 3.5  # Increase gamma for misclassified images (e.g., 4.0)
    # print('gamma2', gamma)

   
    return gamma  # Shape [batch_size]




def compute_background_mask(img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        if isinstance(img_features, torch.Tensor):
            img_features = img_features.cpu().detach().numpy()
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))

        # print('first_pc values:', first_pc[:10])  # 查看前 10 个主成分值

        # print('first_pc.shape', first_pc.shape)

        # print('Threshold:', threshold)
        # print('First PC max:', first_pc.max())
        # print('First PC min:', first_pc.min())

        if masking_type == True:
            mask = first_pc > threshold
            # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            # print('m.shape', m.shape)
            # print('m sum:', m.sum())
            # print('m.size * 0.35', m.size * 0.35)

            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)

        # mask = torch.tensor(mask.squeeze(), dtype=torch.bool)
        return mask.squeeze()



def get_object_name_from_path(instance_images_paths, masking_default):
    """
    从图像路径中提取 object_name，匹配 masking_default 字典中的键。

    Args:
        instance_images_paths (list): 包含图像路径的列表。
        masking_default (dict): 包含 object_name 及其对应 masking 值的字典。

    Returns:
        list: 包含匹配 object_name 的列表。
    """
    object_names = []
    # 遍历所有路径
    for path in instance_images_paths:
        matched = False
        for key in masking_default.keys():
            if f"/{key}/" in path:  # 判断路径中是否包含 key
                object_names.append(key)
                matched = True
                break  # 找到匹配的 key 就退出当前循环
        if not matched:
            object_names.append(None)  # 如果没有匹配项，追加 None

    return object_names


            
def test_2dunet_dino_DRAEM(model, dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step, log_file, model_num, iftest, len_dataset, a_, beta):
    print(f"test:{class_name}")
 
    img_dim = 256
    
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

    with torch.no_grad():
        val_pipe.to(device)
        val_pipe.set_progress_bar_config(disable=True)
        val_pipe.unet.eval()
        val_pipe.vae.eval()
        val_pipe.text_encoder.eval()
        model.eval()

        transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        preds = []
        masks = []
        scores = []
        labels = []


        # print('len_dataset', len_dataset)
        total_pixel_scores = np.zeros((img_dim * img_dim * len_dataset))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len_dataset))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        index_mapping = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

        masking_default = {
                'bottle': False,
                'cable': False,
                'capsule': True,
                'carpet': False,
                'grid': False,
                'hazelnut': True,
                'leather': False,
                'metal_nut': False,
                'pill': True,
                'screw': True,
                'tile': False,
                'toothbrush': True,
                'transistor': False,
                'wood': False,
                'zipper': False
            }

        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
        # for idx_, batch in enumerate(val_dataloader):
            model.eval()
            a_ += 1
            # latents_all, _, image_input, anomaly_mask, instance_label, instance_path, denoise_s_hou1, reconstruct_latent, object_mask = batch

            # image_input = image_input.squeeze().to(device)
            # anomaly_mask = anomaly_mask.to(device)
            # instance_label = instance_label.to(device)
            # denoise_s_hou1 = denoise_s_hou1.to(device)
            # reconstruct_latent = reconstruct_latent.squeeze().to(device)
            # object_mask = object_mask.to(device)





            instance_images = batch["instance_images"].to(device)
            instance_masks = batch["instance_masks"].to(device)
            instance_path = batch["instance_path"]
            instance_label = batch["instance_label"].to(device)

            features_ref_3_bat = batch["features_ref_3"].numpy().astype(np.float32) 
            features_ref_6_bat = batch["features_ref_6"].numpy().astype(np.float32)
            features_ref_9_bat = batch["features_ref_9"].numpy().astype(np.float32)
            features_ref_12_bat = batch["features_ref_12"].numpy().astype(np.float32)
            features_ref_bat = batch["features_ref_path"]


            object_mask = batch["object_mask"].to(device)
            object_mask = object_mask.unsqueeze(1)

            batch_size = instance_images.shape[0]  
            # denoise_s_hou  = [None] * batch_size



            is_normal = instance_label.cpu().numpy()
            for i in range(is_normal.shape[0]):
                anomaly_score_gt.append(is_normal[i])
            true_mask = instance_masks.float()
            print('instance_masks.shape', instance_masks.shape) # [2, 1, 256, 256]
            true_mask_cv = true_mask.detach().cpu().numpy()[:, :, :, :].transpose((0,2,3,1)) # .transpose((1, 2, 0))
            print('true_mask_cv.shape', true_mask_cv.shape) # (2, 256, 256, 1)
            # print('denoise_s_hou1.shape', denoise_s_hou1.shape)
            # print('denoise_s_hou1', denoise_s_hou1)



            # class_labels = get_class_labels(instance_path, device)

            # out_dir = convert_path(instance_path[0], save_root)

            # denoise_s_hou  = [None] * batch_size

            # print('class_labels', class_labels) # [ 0,  7,  3,  9, 14,  4,  1,  2]
            



            object_name = get_object_name_from_path(instance_path, masking_default)
            # print('object_name', object_name) # ['hazelnut', 'bottle', 'carpet', 'metal_nut', 'zipper', 'leather', 'capsule', 'grid']

            mask_ref_images = False
            # masking = masking_default[object_name] 
            # print('masking_default[object_name[0]]', masking_default[object_name[0]])
            # print('masking_default[object_name[1]]', masking_default[object_name[1]])
            # print('masking_default[object_name[2]]', masking_default[object_name[2]])
            # print('masking_default[object_name[3]]', masking_default[object_name[3]])
            # print('masking_default[object_name[4]]', masking_default[object_name[4]])
            # print('masking_default[object_name[5]]', masking_default[object_name[5]])
            # print('masking_default[object_name[6]]', masking_default[object_name[6]])
            # print('masking_default[object_name[7]]', masking_default[object_name[7]])

            min_similarity_map_all_3 = []
            anomaly_map_all_bat_3 = [] 

            min_similarity_map_all_6 = []
            anomaly_map_all_bat_6 = [] 

            min_similarity_map_all_9 = []
            anomaly_map_all_bat_9 = [] 

            min_similarity_map_all_12 = []
            anomaly_map_all_bat_12 = [] 

            dino_feature_all = []

            
            anomaly_maps1g12 = torch.zeros((1*batch_size, 1, args.dino_resolution, args.dino_resolution)).to(device)
            anomaly_maps2g12 = torch.zeros((1*batch_size, 1, args.dino_resolution, args.dino_resolution)).to(device)


            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = 1, 1024, 768
            h = int(n ** 0.5)


            for i_tem in range(batch_size):


                process_id = os.getpid()
                # print('process_id_worker_init_fn', process_id)
                resources = get_faiss_resources(process_id)
                knn_index_3 = resources["knn_index_3"]
                knn_index_6 = resources["knn_index_6"]
                knn_index_9 = resources["knn_index_9"]
                knn_index_12 = resources["knn_index_12"]

                # data_dict = torch.load(get_pt(object_name[i_tem])) 


                # data_dict = {
                # "instance_images": torch.tensor(instance_images).to(dtype=torch.float16),  # 初始化一个包含25个None的列表，用于存储不同iter的latents
                
                # "instance_masks": torch.tensor(instance_masks).to(dtype=torch.float16),  # 预留位置，稍后可以填充具体数据
                # "instance_path": instance_path,
                # "instance_label": torch.tensor(instance_label).to(dtype=torch.float16),

                # # "img_augmented": img_augmented,


                # "features_ref_3": torch.tensor(features_ref_3).to(dtype=torch.float16),
                # "features_ref_6": torch.tensor(features_ref_6).to(dtype=torch.float16),
                # "features_ref_9": torch.tensor(features_ref_9).to(dtype=torch.float16),
                # "features_ref_12": torch.tensor(features_ref_12).to(dtype=torch.float16),}

                # # 你可以添加其他需要的键
               


                # print('features_ref_3_bat.shape', features_ref_3_bat.shape) # [8, 8192, 768]
                # print('features_ref_6_bat.shape', features_ref_6_bat.shape) # [8, 8192, 768]


                start_time = time.time()
                features_ref_3 = features_ref_3_bat[i_tem]
                features_ref_6 = features_ref_6_bat[i_tem]
                features_ref_9 = features_ref_9_bat[i_tem]
                features_ref_12 = features_ref_12_bat[i_tem]

                # print('features_ref_9.shape', features_ref_9.shape) # [8192, 768]
                # print('features_ref_12.shape', features_ref_12.shape) # [8192, 768]

                # img_augmented = augment_image(instance_images[i_tem])
                # print('len(img_augmented)', len(img_augmented)) # 8
                # print(torch.stack(img_augmented).shape) # [8, 3, 256, 256]
                # img_augmented = torch.nn.functional.interpolate(torch.stack(img_augmented), size=args.dino_resolution, mode='bilinear', align_corners=True)
                # print('img_augmented.shape', img_augmented.shape) # [8, 3, 256, 256]
                # img_augmented = transform(img_augmented)
                # print('img_augmented.shape', img_augmented.shape) # [8, 3, 256, 256]
                # with torch.no_grad():
                #     for i_aug in range(len(img_augmented)):
                        
                #         print('img_augmented[i_aug].shape', img_augmented[i_aug].shape) # [3, 256, 256]
                #         # print('dino_model', dino_model)

                #         _, patch_tokens_i_tem_aug = dino_model(img_augmented[i_aug].unsqueeze(0).to(dtype=weight_dtype))
                #         patch_tokens_i_tem_aug  = [feature[:, 1:, :].squeeze(0).float().cpu().numpy() for feature in patch_tokens_i_tem_aug]  
                #         print('patch_tokens_i_tem_aug[-1].shape', patch_tokens_i_tem_aug[-1].shape) # [1, 1024, 768]

                #         mask_ref = compute_background_mask(patch_tokens_i_tem_aug[-1].squeeze(), (32,32), threshold=3, masking_type=(mask_ref_images and masking_default[object_name[i_tem]]))
                #         print('mask_ref.shape', mask_ref.shape) # 
                #         features_ref_3.append(patch_tokens_i_tem_aug[0][mask_ref])
                #         features_ref_6.append(patch_tokens_i_tem_aug[1][mask_ref])
                #         features_ref_9.append(patch_tokens_i_tem_aug[2][mask_ref])
                #         features_ref_12.append(patch_tokens_i_tem_aug[3][mask_ref])

                #     features_ref_3 = np.concatenate(features_ref_3, axis=0)
                #     features_ref_6 = np.concatenate(features_ref_6, axis=0)
                #     features_ref_9 = np.concatenate(features_ref_9, axis=0)
                #     features_ref_12 = np.concatenate(features_ref_12, axis=0)

                elapsed_time = time.time()- start_time
                # print(f"代码运行时间0: {elapsed_time:.4f}秒")
                start_time = time.time()

                # FAISS GPU 资源和索引
                
                knn_index_3.reset()
                # print('knn_index_3.ntotal', knn_index_3.ntotal)  # 应为 0
                # print(f"Thread ID: {threading.get_ident()}")

                faiss.normalize_L2(features_ref_3)  # L2 归一化
                # print('features_ref_3.shape', features_ref_3.shape) # (8192, 768) 8 rotate images 
                knn_index_3.add(features_ref_3)  # 添加到索引

                
                knn_index_6.reset()
                # print('knn_index_6.ntotal', knn_index_6.ntotal)  # 应为 0
                # print(f"Thread ID: {threading.get_ident()}")
                faiss.normalize_L2(features_ref_6)
                knn_index_6.add(features_ref_6)

                
                knn_index_9.reset()
                # print('knn_index_9.ntotal', knn_index_9.ntotal)  # 应为 0
                # print(f"Thread ID: {threading.get_ident()}")
                faiss.normalize_L2(features_ref_9)
                knn_index_9.add(features_ref_9)

                
                knn_index_12.reset()
                # print('knn_index_12.ntotal', knn_index_12.ntotal)  # 应为 0
                # print(f"Thread ID: {threading.get_ident()}")
                faiss.normalize_L2(features_ref_12)
                knn_index_12.add(features_ref_12)


                knn_neighbors = 1024
                grid_size2_chn = (32, 32, knn_neighbors)
                grid_size2 = (32, 32)





                img_2 = torch.nn.functional.interpolate(instance_images[i_tem].unsqueeze(0), size=args.dino_resolution, mode='bilinear', align_corners=True)
                img_2 = transform(img_2)
                with torch.no_grad():
                    _, patch_tokens_features2 = dino_model(img_2.to(dtype=weight_dtype))

                    patch_tokens_features2  = [feature[:, 1:, :].squeeze(0).cpu().numpy() for feature in patch_tokens_features2]  
                    dino_feature_all.append(torch.tensor(patch_tokens_features2).to(device))

                    masking = masking_default[object_name[i_tem]]
                    # print('masking', masking)
                    # print('object_name[i_tem]', object_name[i_tem])
                    # print('object_name', object_name)
                    if masking:
                        mask2 = compute_background_mask(patch_tokens_features2[-1].squeeze(), (32,32), threshold=3, masking_type=masking)
                    else:
                        mask2 = np.ones(patch_tokens_features2[-1].squeeze().shape[0], dtype=bool)
                    # print('patch_tokens_features2[0].shape', patch_tokens_features2[0].shape) # (1024, 768)
                    # print('mask2.shape', mask2.shape) # (1024,)
                
                # print('type(patch_tokens_features2[0])', type(patch_tokens_features2[0]))
                # print('type(mask2)', type(mask2))
                features_anomal_3 = patch_tokens_features2[0][mask2].astype(np.float32)
                features_anomal_6 = patch_tokens_features2[1][mask2].astype(np.float32)
                features_anomal_9 = patch_tokens_features2[2][mask2].astype(np.float32)
                features_anomal_12 = patch_tokens_features2[3][mask2].astype(np.float32)
                # print('type(features_anomal_12)', type(features_anomal_12))
                

                # print('features_anomal_3.shape0', features_anomal_3.shape) # (666 768)
                faiss.normalize_L2(features_anomal_3) 
                # print('features_anomal_3.shape1', features_anomal_3.shape) # (666 768)
                distances_3, _ = knn_index_3.search(features_anomal_3, k = knn_neighbors)
                # if knn_neighbors > 1:
                #     distances = distances.mean(axis=1)
                distances_3 = distances_3 / 2
                output_distances_3 = np.zeros((1024, knn_neighbors), dtype=np.float32)
                # print('distances_3.shape', distances_3.shape) # (666 1024)
                # print('mask2.shape', mask2.shape) # (1024,)
                # print('output_distances_3.shape', output_distances_3.shape) # (1024, 1024)

                output_distances_3[mask2] = distances_3.squeeze()
                d_masked_3 = torch.tensor(output_distances_3.reshape(grid_size2_chn))
                # print('d_masked_3.shape', d_masked_3.shape) # 
                anomaly_map_all_bat_3.append(d_masked_3)
                print("d_masked_3.dtype")
                print(d_masked_3.dtype)


                distances_3, _ = knn_index_3.search(features_anomal_3, k = 1)
                distances_3 = distances_3 / 2
                output_distances_3 = np.zeros_like(mask2, dtype=np.float32)
                output_distances_3[mask2] = distances_3.squeeze()
                d_masked_3 = torch.tensor(output_distances_3.reshape(grid_size2))
                # print('d_masked_3.shape', d_masked_3.shape) # 
                min_similarity_map_all_3.append(d_masked_3)





                faiss.normalize_L2(features_anomal_6) 
                distances_6, _ = knn_index_6.search(features_anomal_6, k = knn_neighbors)
                # if knn_neighbors > 1:
                #     distances = distances.mean(axis=1)
                distances_6 = distances_6 / 2
                output_distances_6 = np.zeros((1024, knn_neighbors), dtype=np.float32)
                output_distances_6[mask2] = distances_6.squeeze()
                d_masked_6 = torch.tensor(output_distances_6.reshape(grid_size2_chn))
                # print('d_masked_6.shape', d_masked_6.shape) # 
                anomaly_map_all_bat_6.append(d_masked_6)

                distances_6, _ = knn_index_6.search(features_anomal_6, k = 1)
                distances_6 = distances_6 / 2
                output_distances_6 = np.zeros_like(mask2, dtype=np.float32)
                output_distances_6[mask2] = distances_6.squeeze()
                d_masked_6 = torch.tensor(output_distances_6.reshape(grid_size2))
                # print('d_masked_6.shape', d_masked_6.shape) # 
                min_similarity_map_all_6.append(d_masked_6)



                faiss.normalize_L2(features_anomal_9) 
                distances_9, _ = knn_index_9.search(features_anomal_9, k = knn_neighbors)
                # if knn_neighbors > 1:
                #     distances = distances.mean(axis=1)
                distances_9 = distances_9 / 2
                output_distances_9 = np.zeros((1024, knn_neighbors), dtype=np.float32)
                output_distances_9[mask2] = distances_9.squeeze()
                d_masked_9 = torch.tensor(output_distances_9.reshape(grid_size2_chn))
                # print('d_masked_9.shape', d_masked_9.shape) # 
                anomaly_map_all_bat_9.append(d_masked_9)

                distances_9, _ = knn_index_9.search(features_anomal_9, k = 1)
                distances_9 = distances_9 / 2
                output_distances_9 = np.zeros_like(mask2, dtype=np.float32)
                output_distances_9[mask2] = distances_9.squeeze()
                d_masked_9 = torch.tensor(output_distances_9.reshape(grid_size2))
                # print('d_masked_9.shape', d_masked_9.shape) # 
                min_similarity_map_all_9.append(d_masked_9)




                faiss.normalize_L2(features_anomal_12) 
                distances_12, _ = knn_index_12.search(features_anomal_12, k = knn_neighbors)
                # if knn_neighbors > 1:
                #     distances = distances.mean(axis=1)
                distances_12 = distances_12 / 2
                output_distances_12 = np.zeros((1024, knn_neighbors), dtype=np.float32)
                output_distances_12[mask2] = distances_12.squeeze()
                d_masked_12 = torch.tensor(output_distances_12.reshape(grid_size2_chn))
                # print('d_masked_12.shape', d_masked_12.shape) # 
                anomaly_map_all_bat_12.append(d_masked_12)

                distances_12, _ = knn_index_12.search(features_anomal_12, k = 1)
                distances_12 = distances_12 / 2
                output_distances_12 = np.zeros_like(mask2, dtype=np.float32)
                output_distances_12[mask2] = distances_12.squeeze()
                d_masked_12 = torch.tensor(output_distances_12.reshape(grid_size2))
                min_similarity_map_all_12.append(d_masked_12)

                # print('d_masked_12.shape', d_masked_12.shape) # 

                elapsed_time = time.time()- start_time
                # print(f"代码运行时间1: {elapsed_time:.4f}秒")
                start_time = time.time()


                anomaly_maps1g = torch.zeros((1, 1, 256, 256)).to(device)
                anomaly_maps1g += F.interpolate(d_masked_3.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True).to(device)
                anomaly_maps1g += F.interpolate(d_masked_6.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True).to(device)
                anomaly_maps1g += F.interpolate(d_masked_9.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True).to(device)
                anomaly_maps1g += F.interpolate(d_masked_12.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True).to(device)

                

                anomaly_maps1g = anomaly_maps1g/4
                print('sum_anomaly_maps1g')
                

                anomaly_maps1g = gaussian_blur2d(anomaly_maps1g, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
                anomaly_maps1g12[i_tem] = anomaly_maps1g.unsqueeze(0)



            #     if idx_ % 5 == 0:
            #         sample = F.interpolate(torch.tensor(mask2, dtype=torch.float32).reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True)
            #         sample = np.squeeze(gaussian_filter(sample, sigma=4) * 255).astype(np.uint8)
            #         cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/mask2_{a_}_{i_tem}_{model_num}.png", sample)

            #         sample = F.interpolate(d_masked_12.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True)
            #         sample = np.squeeze(gaussian_filter(sample, sigma=4) * 255).astype(np.uint8)
            #         cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/d_masked_12+_{a_}_{i_tem}_{model_num}.png", sample)

            #         sample = F.interpolate(d_masked_9.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True)
            #         sample = np.squeeze(gaussian_filter(sample, sigma=4) * 255).astype(np.uint8)
            #         cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/d_masked_9+_{a_}_{i_tem}_{model_num}.png", sample)

            #         sample = F.interpolate(d_masked_6.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True)
            #         sample = np.squeeze(gaussian_filter(sample, sigma=4) * 255).astype(np.uint8)
            #         cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/d_masked_6+_{a_}_{i_tem}_{model_num}.png", sample)

            #         sample = F.interpolate(d_masked_3.reshape(-1, 1, 32, 32), size=256, mode='bilinear', align_corners=True)
            #         sample = np.squeeze(gaussian_filter(sample, sigma=4) * 255).astype(np.uint8)
            #         cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/d_masked_3+_{a_}_{i_tem}_{model_num}.png", sample)

            
            # elapsed_time = time.time()- start_time
            # # print(f"代码运行时间2: {elapsed_time:.4f}秒")
            # start_time = time.time()

            # min_similarity_map_all_3 = [torch.tensor(arr) for arr in min_similarity_map_all_3]
            # anomaly_map_all_bat_3 = [torch.tensor(arr) for arr in anomaly_map_all_bat_3]
            # min_similarity_map_all_6 = [torch.tensor(arr) for arr in min_similarity_map_all_6]
            # anomaly_map_all_bat_6 = [torch.tensor(arr) for arr in anomaly_map_all_bat_6]
            # min_similarity_map_all_9 = [torch.tensor(arr) for arr in min_similarity_map_all_9]
            # anomaly_map_all_bat_9 = [torch.tensor(arr) for arr in anomaly_map_all_bat_9]
            # min_similarity_map_all_12 = [torch.tensor(arr) for arr in min_similarity_map_all_12]
            # anomaly_map_all_bat_12 = [torch.tensor(arr) for arr in anomaly_map_all_bat_12]


            min_similarity_map_all_3 = torch.stack(min_similarity_map_all_3)
            anomaly_map_all_bat_3 = torch.stack(anomaly_map_all_bat_3)
            min_similarity_map_all_6 = torch.stack(min_similarity_map_all_6)
            anomaly_map_all_bat_6 = torch.stack(anomaly_map_all_bat_6)
            min_similarity_map_all_9 = torch.stack(min_similarity_map_all_9)
            anomaly_map_all_bat_9 = torch.stack(anomaly_map_all_bat_9)
            min_similarity_map_all_12 = torch.stack(min_similarity_map_all_12)
            anomaly_map_all_bat_12 = torch.stack(anomaly_map_all_bat_12)

            # print('min_similarity_map_all_12.shape', min_similarity_map_all_12.shape) # [8, 32, 32]
            # print('anomaly_map_all_bat_12.shape', anomaly_map_all_bat_12.shape) # [8, 32, 32, 1024]

            print("anomaly_map_all_bat_3.dtype")
            print(anomaly_map_all_bat_3.dtype)


            anomaly_maps_all = torch.cat(
                [anomaly_map_all_bat_3.unsqueeze(1),
                 anomaly_map_all_bat_6.unsqueeze(1),
                 anomaly_map_all_bat_9.unsqueeze(1),
                 anomaly_map_all_bat_12.unsqueeze(1)], 
                dim=1)  # Shape: [8, 4, 32, 32, 1024]

            # print("Shape after concatenation:", anomaly_maps_all.shape) # [8, 4, 32, 32, 1024]

            # Step 2: Permute to rearrange dimensions: [8, 1024, 4, 32, 32]
            anomaly_maps_all = anomaly_maps_all.permute(0, 4, 1, 2, 3)  # Rearrange dimensions
            # print("Shape after permutation:", anomaly_maps_all.shape) # [8, 1024, 4, 32, 32]

            # Step 3: Interpolate last two dimensions to 64x64
            print("anomaly_maps_all.dtype")
            print(anomaly_maps_all.dtype)

            anomaly_map_all_bat = F.interpolate(
                anomaly_maps_all.reshape(-1, 4, 32, 32), 
                size=(64, 64), 
                mode='bilinear', 
                align_corners=False).reshape(2, 1024, 4, 64, 64).to(device)

            print("Shape after interpolation:", anomaly_map_all_bat.shape)  # [8, 1024, 4, 64, 64]


            min_similarity_maps_all = torch.cat(
                [min_similarity_map_all_3.unsqueeze(1),
                 min_similarity_map_all_6.unsqueeze(1),
                 min_similarity_map_all_9.unsqueeze(1),
                 min_similarity_map_all_12.unsqueeze(1)], 
                dim=1)  # Shape: [8, 4, 32, 32]

            # print("Shape after concatenation:", min_similarity_maps_all.shape) # [8, 4, 32, 32]

            # Step 2: Interpolate last two dimensions to 64x64
            min_similarity_map_all_bat = F.interpolate(
                min_similarity_maps_all, 
                size=(64, 64), 
                mode='bilinear', 
                align_corners=False).to(device)
            # print("Shape after interpolation:", min_similarity_map_all_bat.shape)  # [8, 4, 64, 64]



            
            dino_features = [[] for _ in range(4)]

            # Step 2: 遍历 batch，将每一层的特征添加到对应列表中
            for batch in dino_feature_all:  # 遍历 batch_size 个元素
                for layer_idx in range(4):
                    dino_features[layer_idx].append(batch[layer_idx])

            # Step 3: 使用 torch.stack 将每一层的特征沿 batch 维度拼接
            for layer_idx in range(4):
                dino_features[layer_idx] = torch.stack(dino_features[layer_idx], dim=0).float().to(device)


            print('dino_features[0].shape', dino_features[0].shape) # [8, 1024, 768]
            print('dino_features[1].shape', dino_features[1].shape) # [8, 1024, 768]
            print('dino_features[2].shape', dino_features[2].shape) # [8, 1024, 768]
            print('dino_features[3].shape', dino_features[3].shape) # [8, 1024, 768]

            

            anomaly_mask = instance_masks.float()
            print('anomaly_map_all_bat.shape', anomaly_map_all_bat.shape) # # [2, 1024, 4, 64, 64]
            # # print('anomaly_mask.shape', anomaly_mask.shape) # [2, 1, 256, 256]
            # # anomaly_mask = anomaly_mask.repeat(4, 1, 1, 1)
            
            # # if class_name in ['transistor', 'pcb1', 'pcb4']:
            # #     anomaly_map_all_bat12 = [anomaly_map.view(-1, 256, 256) for anomaly_map in anomaly_map_all_bat12]
            # #     anomaly_map_all_bat12 = torch.stack(anomaly_map_all_bat12).float()
            # #     # print('class_name in transistor')



            if torch.isnan(anomaly_map_all_bat).any() or torch.isinf(anomaly_map_all_bat).any():
                print("Input tensor contains NaN or Inf anomaly_maps_all1")

            print("anomaly_map_all_bat.dtype")
            print(anomaly_map_all_bat.dtype)
            print(dino_features[0].dtype)
            print(min_similarity_map_all_bat.dtype)
             

            # output, cls_out = model(anomaly_map_all_bat[i_all], dino_features)
            output, _ = model(anomaly_map_all_bat, dino_features, min_similarity_map_all_bat)
            # aa = False
            # if aa:
            #     # output, _  = model(anomaly_map_all_bat[i_all], dino_features)
            #     # print('output.shape', output.shape) # [2, 2, 64, 64]
            #     output_focl = torch.softmax(output, dim=1)
            # else:
            #     output_focl = output
            output_focl = torch.softmax(output, dim=1)

            print('output_focl.shape', output_focl.shape) # (4, 2, 256, 256)
            # output_focl, _ = torch.min(output_focl, dim=2)
            output_focl = F.interpolate(output_focl, size=args.dino_resolution, mode='bilinear', align_corners=True)  # Shape: [4, 2, 256, 256]
            # anomaly_prob = output_focl[:, 1, :, :].unsqueeze(1)
            # segment_loss = loss_focal(output, anomaly_mask)
            # output_focl = output_focl
            # print('output_focl.shape', output_focl.shape) # [2, 2, 256, 256]

             

            anomaly_maps1 = output_focl[:, 1, :, :].unsqueeze(1)

            print('torch.sum(anomaly_maps1)', torch.sum(anomaly_maps1))
            
            # anomaly_maps1 = output[:, 1, :, :].unsqueeze(1)
            print('anomaly_maps1.shape', anomaly_maps1.shape) # [2, 1, 256, 256]

            print('output[: ,1 ,: ,:].unsqueeze(1).shape', output[: ,1 ,: ,:].unsqueeze(1).shape) # [2, 1, 256, 256]
            print('object_mask.shape', object_mask.shape) # [2, 1, 256, 256]
            # if class_name in ['transistor', 'pcb1', 'pcb4']:
            #     anomaly_maps12 = output12[:, 1, :, :].unsqueeze(1)

            out_mask_cv = (output_focl[: ,1 ,: ,:].unsqueeze(1) * object_mask).cpu().detach().numpy() 
            # print('out_mask_cv.shape', out_mask_cv.shape) # [2, 1, 256, 256]
            out_mask_averaged = torch.nn.functional.avg_pool2d(output_focl[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy() 
            print('out_mask_averaged.shape', out_mask_averaged.shape) # [2, 1, 256, 256]   


            out_mask_averaged = out_mask_averaged * object_mask.cpu().numpy()


            # image_score = np.max(out_mask_averaged)
            # print('image_score', image_score) #  
            # anomaly_score_prediction.append(image_score)


            print('true_mask_cv.shape', true_mask_cv.shape) # (2, 256, 256, 1)
            # flat_true_mask = true_mask_cv.flatten()
            # print('flat_true_mask.shape', flat_true_mask.shape) # 131072
            print('out_mask_cv.shape', out_mask_cv.shape) # 2, 1, 256, 256
            # flat_out_mask = out_mask_cv.flatten()
            # print('flat_out_mask.shape', flat_out_mask.shape) # (131072,)

            # print('mask_cnt * img_dim * img_dim', mask_cnt * img_dim * img_dim) # 0
            # print('(mask_cnt + 1) * img_dim * img_dim', (mask_cnt + 2) * img_dim * img_dim) # 131072



            # total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 2) * img_dim * img_dim] = flat_out_mask
            # total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 2) * img_dim * img_dim] = flat_true_mask
            # mask_cnt += 2


            # 假设 batch size 是 2，out_mask_averaged 形状是 [2, 1, 256, 256]


            for i in range(out_mask_averaged.shape[0]):
                # 对每张图片分别计算 anomaly score
                image_score = np.max(out_mask_averaged[i])  # 对当前图片取最大值
                # print(f'image_score for image {i}: {image_score}')
                anomaly_score_prediction.append(image_score)

                # 对每张图片计算平铺后的 mask
                flat_true_mask = true_mask_cv[i].flatten()  # true_mask_cv 是 [2, 256, 256, 1]，选择第 i 张图片
                print(f'flat_true_mask.shape for image {i}: {flat_true_mask.shape}')
                
                flat_out_mask = out_mask_cv[i].flatten()  # out_mask_cv 是 [2, 1, 256, 256]，选择第 i 张图片
                print(f'flat_out_mask.shape for image {i}: {flat_out_mask.shape}')
                
                # 将计算结果存储到 total_pixel_scores 和 total_gt_pixel_scores 中
                # print('mask_cnt * img_dim * img_dim', mask_cnt * img_dim * img_dim) #  
                # print('(mask_cnt + 1) * img_dim * img_dim', (mask_cnt + 1) * img_dim * img_dim) #  

                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask

                mask_cnt += 1  # 每次处理完一张图片，更新 mask_cnt


            # print('object_mask.shape', object_mask.shape) # ######################################################################################### [2, 1, 256, 256]

  


 
 
            # anomaly_maps = (anomaly_maps1*1.0 + anomaly_maps1g12*0.0) * object_mask.to(device)
            anomaly_maps = ( anomaly_maps1g12) * object_mask.to(device)
            # anomaly_maps = (anomaly_maps1*0.9 + anomaly_maps1g12*0.1) * object_mask.to(device)

            print('anomaly_maps.shape', anomaly_maps.shape) # [2, 1, 256, 256]
            # anomaly_maps = anomaly_maps1.squeeze(0) * object_mask.to(device)
            
            
            anomaly_maps_np = anomaly_maps.detach().cpu().numpy()

            if idx_ % 5 == 0:
                for bat in range(batch_size):
                    print('anomaly_maps_np.shape', anomaly_maps_np.shape)
                    single_map = anomaly_maps_np[bat]
                    print('single_map.shape0', single_map.shape)
                    single_map = single_map * 255 # (single_map - single_map.min()) / (single_map.max() - single_map.min()) * 255
                    print('single_map.shape1', single_map.shape)
                    single_map = np.squeeze(single_map).astype(np.uint8)  # Convert to uint8 for image format compatibility
                    # print('single_map.shape2', single_map.shape)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/anomaly_map_my_test_{a_}_{bat}_{model_num}.png", single_map)


                    sample = reverse_normalization(torch.tensor(instance_images[bat]))

                    sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/gt_images_test_{a_}_{bat}_{model_num}.png", sample)



                    sample = object_mask[bat].detach().cpu().numpy().squeeze() * 255
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/object_mask_test_{a_}_{bat}_{model_num}.png", sample)


                    sample = instance_masks[bat].detach().cpu().numpy().squeeze() * 255
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/anomalydino/costvolume_2_grid_test_bs16_triloss_gai_anomalydino/instance_masks_test_{a_}_{bat}_{model_num}.png", sample)





            anomaly_maps = anomaly_maps.squeeze()
            instance_label = instance_label.squeeze()
            print('anomaly_maps.shape', anomaly_maps.shape) # ######################################################################################### [2, 256, 256]
            print('anomaly_mask.shape', anomaly_mask.shape) # ######################################################################################### [2, 1, 256, 256]
            print('anomaly_mask[:, 0, :, :].shape', anomaly_mask[:, 0, :, :].shape) # [2, 256, 256]
            print('instance_label.shape', instance_label.shape) # [2]

            # print(750)
            score = torch.topk(torch.flatten(anomaly_maps, start_dim=1), 250)[0].mean(dim=1)
            # print('score.shape', score.shape)
            masks.extend([m for m in anomaly_mask[:, 0, :, :].cpu().numpy()])
            preds.extend([a for a in anomaly_maps.cpu().numpy()])
            scores.extend([s for s in score.cpu().numpy()])
            labels.extend([l for l in instance_label.cpu().numpy()])




        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        # print('anomaly_score_gt.shape', anomaly_score_gt.shape) # (110, 1)
        # print('anomaly_score_prediction.shape', anomaly_score_prediction.shape) # (110, )
        # print('anomaly_score_gt.shape', anomaly_score_gt.shape) # (110, 1)
        # print('anomaly_score_prediction.shape', anomaly_score_prediction.shape) # (110, )
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
        # print('img_dim * img_dim * mask_cnt', img_dim * img_dim * mask_cnt) # 7208960
        # print('total_gt_pixel_scores.shape', total_gt_pixel_scores.shape) # 7208960
        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        # print('total_pixel_scores.shape', total_pixel_scores.shape) # 7208960
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        # print('total_gt_pixel_scores.shape', total_gt_pixel_scores.shape) # 7208960
        # print('total_pixel_scores.shape', total_pixel_scores.shape) # 7208960
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(class_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

        log_file.write(f"Class: {class_name}\n")
        
        log_file.write(f"AUC Image: {str(auroc)}\n")
        log_file.write(f"AP Image:: {str(ap)}\n")
        log_file.write(f"AUC Pixel: { str(auroc_pixel)}\n")
        log_file.write(f"AUC Pixel: {str(ap_pixel)}\n")



        scores = normalize(np.array(scores))
        labels = np.array(labels)
        preds = np.array(preds)
        masks = np.array(masks, dtype=np.int_)

        precisions_image, recalls_image, _ = precision_recall_curve(labels, scores)
        f1_scores_image = (2 * precisions_image * recalls_image) / (precisions_image + recalls_image)
        best_f1_scores_image = np.max(f1_scores_image[np.isfinite(f1_scores_image)])
        auroc_image = roc_auc_score(labels, scores)
        AP_image = average_precision_score(labels, scores)

        precisions_pixel, recalls_pixel, _ = precision_recall_curve(masks.ravel(), preds.ravel())
        f1_scores_pixel = (2 * precisions_pixel * recalls_pixel) / (precisions_pixel + recalls_pixel)
        best_f1_scores_pixel = np.max(f1_scores_pixel[np.isfinite(f1_scores_pixel)])
        auroc_pixel = roc_auc_score(masks.ravel(), preds.ravel())
        AP_pixel = average_precision_score(masks.ravel(), preds.ravel())

        pro = compute_pro(masks, preds)

        print(f"test-------- I-AUROC/I-AP/I-F1-max/P-AUROC/P-AP/P-F1-max/PRO:{round(auroc_image, 4)}/{round(AP_image, 4)}/{round(best_f1_scores_image, 4)}/"
              f"{round(auroc_pixel, 4)}/{round(AP_pixel, 4)}/{round(best_f1_scores_pixel, 4)}/{round(pro, 4)}-----")


        # 将 class_name 和指标写入日志
        log_file.write(f"Class: {class_name}\n")
        log_file.write(f"I-AUROC: {round(auroc_image, 4)}\n")
        log_file.write(f"I-AP: {round(AP_image, 4)}\n")
        log_file.write(f"I-F1-max: {round(best_f1_scores_image, 4)}\n")
        log_file.write(f"P-AUROC: {round(auroc_pixel, 4)}\n")
        log_file.write(f"P-AP: {round(AP_pixel, 4)}\n")
        log_file.write(f"P-F1-max: {round(best_f1_scores_pixel, 4)}\n")
        log_file.write(f"PRO: {round(pro, 4)}\n\n")

        log_file.flush()  # 确保每次写入都能及时保存


        return round(auroc_image, 4) * 100, round(AP_image, 4) * 100, round(best_f1_scores_image, 4) * 100, round(
            auroc_pixel, 4) * 100, round(AP_pixel, 4) * 100, round(best_f1_scores_pixel, 4) * 100, round(pro, 4) * 100, obj_ap_pixel_list, obj_auroc_pixel_list, obj_ap_image_list, obj_auroc_image_list, a_ 




def load_vae(vae):
    print(args.instance_data_dir)
    if "VisA" in args.instance_data_dir:
        vae_path = 'model/vae/visa_diad_epoch=118-step=64498.ckpt'
    elif 'PCBBank' in args.instance_data_dir:
        vae_path = 'model/vae/pacbank_epoch=245-step=64944.ckpt'
    else:
        vae_path = None

    if vae_path:
        sd = torch.load(vae_path)["state_dict"]
        print(f"load vae in test :{vae_path}")

        keys = list(sd.keys())
        for k in keys:
            if "loss" in k:
                del sd[k]
        vae.load_state_dict(sd, map_location='cpu')
    return vae


def load_test_model(args, weight_dtype):
    dino_model = get_vit_encoder(vit_arch="vit_base", vit_model="dino", vit_patch_size=8, enc_type_feats=None).to(device, dtype=weight_dtype)
    dino_model.eval()

    val_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        torch_dtype=weight_dtype,
    )
    return dino_model, val_pipe


def main(args, class_name):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("GLAD", config=tracker_config)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder, vae, unet = model(args)

    vae.to(accelerator.device, dtype=weight_dtype)
    vae = load_vae(vae)

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae.eval()
    text_encoder.eval()

    if args.pre_compute_text_embeddings:
        pre_encoder_hidden_states = compute_text_embeddings(args.instance_prompt, tokenizer, text_encoder)
    else:
        pre_encoder_hidden_states = None

    optimizer_class = bnb.optim.AdamW8bit
    params_to_optimize = (unet.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    train_dataset = MVTecDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_name=class_name,
        tokenizer=tokenizer,
        resize=args.resolution,
        img_size=args.resolution,
        anomaly_path=args.anomaly_data_dir,
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_epochs = round(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Class name = {class_name}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  save model into {args.output_dir}")
    global_step = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # loss, global_step = train_one_epoch(accelerator,
    #                                     vae, text_encoder, unet, noise_scheduler,
    #                                     train_dataloader, pre_encoder_hidden_states,
    #                                     optimizer, lr_scheduler,
    #                                     weight_dtype,
    #                                     global_step, progress_bar,
    #                                     args)
    # logger.info(f"train--------train loss:{loss}-----")
    del vae
    del text_encoder
    del unet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    # device = torch.device("cuda")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dino_model, val_pipe = load_test_model(args, torch.float16)
    dino_frozen = copy.deepcopy(dino_model)

    args.output_dir = os.path.join('model', args.instance_data_dir.split('/')[-1] + '_' + args.output_dir + f"_{args.seed}")




    writer = SummaryWriter(log_dir='/home/ZZ/anomaly/GLAD-main/model/cost_volume/logs_test_train_2dunet_dino')
    if args.train:
        main(args, "")
    else:
        if 'MVTec-AD' in args.instance_data_dir:
            args.input_threshold = 0.45
            args.denoise_step = 650
            args.min_step = 350
            args.resolution = 256
            args.dino_resolution = 256
            checkpoint_step = 20000
            args.v = 0
            args.dino_save_path = None
            # CLSNAMES = [
            #     'transistor',
            #     "capsule",
            #     'toothbrush',
            #     'screw',

            # ]

            CLSNAMES = [
                'hazelnut',
                "capsule",
                'cable',
                'screw',
                'transistor',
                'wood',

                'grid',
                'carpet',
                'leather',
                'tile',
                'bottle',
                'metal_nut',
                'pill',
                'toothbrush',
                'zipper',
            ]
            
        elif 'MPDD' in args.instance_data_dir:
            args.input_threshold = 0.35
            args.denoise_step = 500
            args.min_step = 350
            args.resolution = 256
            args.dino_resolution = 256
            checkpoint_step = 3000
            args.v = 0
            args.dino_save_path = None
            CLSNAMES = [
                'bracket_black',
                'bracket_brown',
                'bracket_white',
                'connector',
                'metal_plate',
                'tubes',
            ]
        elif 'VisA' in args.instance_data_dir:
            args.input_threshold = 0.15
            args.denoise_step = 500
            args.min_step = 250
            args.resolution = 256
            args.dino_resolution = 256
            checkpoint_step = 20000
            args.v = 2
            args.dino_save_path = None
            CLSNAMES = [
                'candle',
                'capsules',
                'cashew',
                'chewinggum',
                'fryum',
                'macaroni1',
                'macaroni2',
                'pcb1',
                'pcb2',
                'pcb3',
                'pcb4',
                'pipe_fryum',
            ]
        elif 'PCBBank' in args.instance_data_dir:
            args.input_threshold = 0.2
            args.denoise_step = 500
            args.min_step = 250
            args.resolution = 256
            args.dino_resolution = 256
            checkpoint_step = 20000
            args.v = 1
            args.dino_save_path = 'model/pcbbank_dino_multi/PCBBank_4mlp_256_200_bs16_0.0003_15_no_grad2_lmd0.01/epoch1.pth'
            CLSNAMES = {
                'pcb1',
                'pcb2',
                'pcb3',
                'pcb4',
                'pcb5',
                'pcb6',
                'pcb7',
            }





        if args.dino_save_path:
            dino_model.load_state_dict(torch.load(args.dino_save_path))

        print(f"Test checkpoint step {checkpoint_step}", time.asctime())


        val_pipe.unet.load_state_dict(
            torch.load(f"/home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/{args.output_dir}/checkpoint-{checkpoint_step}/pytorch_model.bin", map_location='cpu')
        )

        val_pipe.unet.to(device).to(dtype=torch.float16)
        val_pipe.vae = load_vae(val_pipe.vae).to(device)

        print(args)
        performances = [[], [], [], [], [], [], []]

        test_2d = False
        train_2d = False
        train_2d_save = False
        test_2d_save = False

        train_2d_dino = False
        test_2d_dino_glad = False
        
        test_3d_dino_DRAEM_600 = False
        test_3d_dino_DRAEM_1200 = False

        test_3d_dino_DRAEM_qianghua = False

        test_3d_dino_DRAEM_qianghua_all_0 = True

        test_3d_dino_DRAEM_qianghua_all_300 = False

        a_ = 0



        pth_files = [
            "/home/ZZ/anomaly/GLAD-main/model/cost_volume/qianghua/epoch_3_6000_fscratch_3d.pth",
            # "/home/ZZ/anomaly/GLAD-main/model/cost_volume/qianghua/epoch_3_5400_fscratch_2d.pth"
        ]


        if test_3d_dino_DRAEM_qianghua:
            start_ = 1
            log_file_path = "/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_3dunet_dino_triloss_gai_3d_channel/log_dino_triloss_DRAEM_gai_3d_channel_qianghua.txt"

            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入
                # for beta in range(0,1,0.05):
                # for beta in range(0,1,3):
                for beta in range(1,2,3):
                    for model_path in pth_files:
                        print(f"Loading model weights from: {model_path}")

                        # 打开 log 文件，并将每次模型测试结果写入其中
                        

                        # 遍历所有模型路径
                        log_file.write(f"Model_qian: {model_path}\n")
                        
                        # 持续检测模型路径是否存在
                        model_found = False
                        while not os.path.exists(model_path):
                            if not model_found:
                                print(f"等待模型: {model_path} 加载...")
                                model_found = True  # 只打印一次等待信息
                            time.sleep(60)  # 每隔10秒检查一次路径是否存在

                        # 当路径存在时，加载模型
                        print(f"加载模型: {model_path}")
                        model = DiscriminativeSubNetwork_2d_att_dino_cost(in_channels=4, out_channels=2, base_channels=48).to(device)
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict']) 
                        model = model.float()
                        performances = [[], [], [], [], [], [], []]


                        performance_dream = [[], [], [], []]

     

                        obj_ap_pixel_list = []
                        obj_auroc_pixel_list = []
                        obj_ap_image_list = []
                        obj_auroc_image_list = []


                        for class_name in CLSNAMES:
                            args.instance_prompt = 'a photo of sks' + class_name
                            print(class_name, f"input_threshold is {args.input_threshold} {args.denoise_step} {args.min_step} {args.v}")

                            fix_seeds(args.seed)
                            data_directory_class = f'/home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/test_save_path_1234/{class_name}'

                            val_dataset = CustomDataset_dict_test(data_directory_class, device, args.resolution)
                           

     
                            val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=2,
                                num_workers=4,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True
                            )
              

                            results = test_2dunet_dino_DRAEM(model, 
                                dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                                class_name, checkpoint_step, log_file, 0, test_2d, len(val_dataset), a_, beta)

                            a_ = results[-1]
                            print("a_:", a_)

                            for j, result in enumerate(results[:7]):
                                performances[j].append(result)
                            

                            obj_ap_pixel_list.extend(results[-5])
                            obj_auroc_pixel_list.extend(results[-4])
                            obj_ap_image_list.extend(results[-3])
                            obj_auroc_image_list.extend(results[-2])

                            


                        performances = np.array(performances).T
                        print("mean:", np.mean(performances, axis=0))



                        print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
                        print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
                        print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
                        print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))



                        # 将结果写入 log 文件

                        log_file.write(f"AUC Image mean: {np.mean(obj_auroc_image_list)}\n")
                        log_file.write(f"AP Image mean: {np.mean(obj_ap_image_list)}\n")
                        log_file.write(f"AUC Pixel mean: {np.mean(obj_auroc_pixel_list)}\n")
                        log_file.write(f"AP Pixel mean: {np.mean(obj_ap_pixel_list)}\n")


                        log_file.write(f"Model_hou_glad: {model_path}\n")
                        log_file.write(f"Mean Performance_glad: {np.mean(performances, axis=0)}\n\n")
                        log_file.flush()  # 确保写入文件



        
        elif test_3d_dino_DRAEM_qianghua_all_0:
            start_ = 1
            log_file_path = "/home/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume/log_dino_triloss_DRAEM_gai_3d_channel_qianghua_volume_0_cost_tormin48_anomalydino.txt"

            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入
                # model_paths = [os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_1_{i}_fscratch.pth') for i in range(100,500,100)]
                for epoch_ in range(16,100):
                    # if epoch_ ==1:
                    #     start_ = 4200
                    # else:
                    #     start_ = 600
                    for idx_ in range(0, 1, 600):

                        model_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume', f'epoch_{epoch_}_{idx_}_fscratch_anomalydino_num4.pth')

                        # 打开 log 文件，并将每次模型测试结果写入其中
                        

                        # 遍历所有模型路径
                        log_file.write(f"Model_qian: {model_path}\n")
                        
                        # 持续检测模型路径是否存在
                        model_found = False
                        while not os.path.exists(model_path):
                            if not model_found:
                                print(f"等待模型: {model_path} 加载...")
                                model_found = True  # 只打印一次等待信息
                            time.sleep(60)  # 每隔10秒检查一次路径是否存在

                        # 当路径存在时，加载模型
                        print(f"加载模型: {model_path}")
                        model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=1024, out_channels=2, base_channels=48).to(device)
                        # model = DiscriminativeSubNetwork_2d_att_dino_cost(in_channels=4, out_channels=2, base_channels=64).to(device)
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict']) 
                        model = model.float()
                        performances = [[], [], [], [], [], [], []]


                        performance_dream = [[], [], [], []]

     

                        obj_ap_pixel_list = []
                        obj_auroc_pixel_list = []
                        obj_ap_image_list = []
                        obj_auroc_image_list = []


                        for class_name in CLSNAMES:
                            args.instance_prompt = 'a photo of sks' + class_name
                            print(class_name, f"input_threshold is {args.input_threshold} {args.denoise_step} {args.min_step} {args.v}")

                            

                             
                            fix_seeds(args.seed)
                            
                            val_dataset  = MVTecDataset(    # MVTecDataset1   MVTecDataset
                                instance_data_root=args.instance_data_dir,
                                instance_prompt=args.instance_prompt,
                                class_name=class_name,
                                tokenizer=None,
                                resize=args.resolution,
                                img_size=args.resolution,
                                train=False
                            )




                            val_dataloader  = DataLoader(
                                val_dataset,
                                batch_size=2,
                                shuffle=False,
                                num_workers= 0,  # args.dataloader_num_workers,              test_train_3dunet      test
                                drop_last=True,
                                worker_init_fn=worker_init_fn,
                                # pin_memory=True,
                            )   
                  


 
                            
      
 
                            results = test_2dunet_dino_DRAEM(model, 
                                dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                                class_name, checkpoint_step, log_file, 0, test_2d, len(val_dataset), a_, beta=None)

                            a_ = results[-1]
                            print("a_:", a_)

                            for j, result in enumerate(results[:7]):
                                performances[j].append(result)
                            

                            obj_ap_pixel_list.extend(results[-5])
                            obj_auroc_pixel_list.extend(results[-4])
                            obj_ap_image_list.extend(results[-3])
                            obj_auroc_image_list.extend(results[-2])

                            


                        performances = np.array(performances).T
                        print("mean:", np.mean(performances, axis=0))



                        print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
                        print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
                        print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
                        print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))



                        # 将结果写入 log 文件

                        log_file.write(f"AUC Image mean: {np.mean(obj_auroc_image_list)}\n")
                        log_file.write(f"AP Image mean: {np.mean(obj_ap_image_list)}\n")
                        log_file.write(f"AUC Pixel mean: {np.mean(obj_auroc_pixel_list)}\n")
                        log_file.write(f"AP Pixel mean: {np.mean(obj_ap_pixel_list)}\n")


                        log_file.write(f"Model_hou_glad: {model_path}\n")
                        log_file.write(f"Mean Performance_glad: {np.mean(performances, axis=0)}\n\n")
                        log_file.flush()  # 确保写入文件



        
         
        elif test_3d_dino_DRAEM_qianghua_all_300:
            start_ = 1
            log_file_path = "/home/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume/log_dino_triloss_DRAEM_gai_3d_channel_qianghua_volume_300_cost_tormin48_anomalydino.txt"

            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入
                # model_paths = [os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_1_{i}_fscratch.pth') for i in range(100,500,100)]
                for epoch_ in range(1,100):
                    for idx_ in range(300, 301, 600):

                        model_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/anomalydino/cost_volume', f'epoch_{epoch_}_{idx_}_fscratch_anomalydino.pth')

                        # 遍历所有模型路径
                        log_file.write(f"Model_qian: {model_path}\n")
                        
                        # 持续检测模型路径是否存在
                        model_found = False
                        while not os.path.exists(model_path):
                            if not model_found:
                                print(f"等待模型: {model_path} 加载...")
                                model_found = True  # 只打印一次等待信息
                            time.sleep(60)  # 每隔10秒检查一次路径是否存在

                        # 当路径存在时，加载模型
                        print(f"加载模型: {model_path}")
                        model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=1024, out_channels=2, base_channels=48).to(device)
                        # model = DiscriminativeSubNetwork_2d_att_dino_cost(in_channels=4, out_channels=2, base_channels=64).to(device)
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict']) 
                        model = model.float()
                        performances = [[], [], [], [], [], [], []]


                        performance_dream = [[], [], [], []]

     

                        obj_ap_pixel_list = []
                        obj_auroc_pixel_list = []
                        obj_ap_image_list = []
                        obj_auroc_image_list = []


                        for class_name in CLSNAMES:
                            args.instance_prompt = 'a photo of sks' + class_name
                            print(class_name, f"input_threshold is {args.input_threshold} {args.denoise_step} {args.min_step} {args.v}")

                            

                             
                            fix_seeds(args.seed)
                            
                            val_dataset  = MVTecDataset(    # MVTecDataset1   MVTecDataset
                                instance_data_root=args.instance_data_dir,
                                instance_prompt=args.instance_prompt,
                                class_name=class_name,
                                tokenizer=None,
                                resize=args.resolution,
                                img_size=args.resolution,
                                train=False
                            )




                            val_dataloader  = DataLoader(
                                val_dataset,
                                batch_size=2,
                                shuffle=False,
                                num_workers= 0,  # args.dataloader_num_workers,              test_train_3dunet      test
                                drop_last=True,
                                worker_init_fn=worker_init_fn,
                                # pin_memory=True,
                            )   
                  


 
                            
      
 
                            results = test_2dunet_dino_DRAEM(model, 
                                dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                                class_name, checkpoint_step, log_file, 0, test_2d, len(val_dataset), a_, beta=None)

                            a_ = results[-1]
                            print("a_:", a_)

                            for j, result in enumerate(results[:7]):
                                performances[j].append(result)
                            

                            obj_ap_pixel_list.extend(results[-5])
                            obj_auroc_pixel_list.extend(results[-4])
                            obj_ap_image_list.extend(results[-3])
                            obj_auroc_image_list.extend(results[-2])

                            


                        performances = np.array(performances).T
                        print("mean:", np.mean(performances, axis=0))



                        print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
                        print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
                        print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
                        print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))



                        # 将结果写入 log 文件

                        log_file.write(f"AUC Image mean: {np.mean(obj_auroc_image_list)}\n")
                        log_file.write(f"AP Image mean: {np.mean(obj_ap_image_list)}\n")
                        log_file.write(f"AUC Pixel mean: {np.mean(obj_auroc_pixel_list)}\n")
                        log_file.write(f"AP Pixel mean: {np.mean(obj_ap_pixel_list)}\n")


                        log_file.write(f"Model_hou_glad: {model_path}\n")
                        log_file.write(f"Mean Performance_glad: {np.mean(performances, axis=0)}\n\n")
                        log_file.flush()  # 确保写入文件

