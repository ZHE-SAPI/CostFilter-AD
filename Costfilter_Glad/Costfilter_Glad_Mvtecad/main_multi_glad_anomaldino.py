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
from dataset.dataset_multiclass1 import MVTecDataset1
from dataset.dataset_multiclass2 import MVTecDataset2
from dataset.dataset_multiclass3_zhengchang import MVTecDataset_anomaldino


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

from unet2d1 import DiscriminativeSubNetwork_2d
from unet3d1 import DiscriminativeSubNetwork_3d
from unet3d2 import DiscriminativeSubNetwork_3d2
from unet3d2_att import DiscriminativeSubNetwork_3d2_att
from unet2d_att import DiscriminativeSubNetwork_2d_att

from torch.cuda.amp import autocast, GradScaler



warnings.filterwarnings("ignore")
logger = get_logger(__name__)

from sklearn.decomposition import PCA
import faiss

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

def weights_init_3d(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
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


 
def normalization(original_image):
    """
    Standardizes an image to the range [-1, 1] using the specified mean and std.
    Input:
        original_image: Tensor or NumPy array with values in [0, 1] or [0, 255].
                        Shape: (C, H, W) or (B, C, H, W).
    Output:
        normalized_image: Tensor with values standardized using mean and std.
    """
    # 均值和标准差，与 reverse_normalization 保持一致
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    # 如果输入是 numpy 数组，先转换为 PyTorch 张量
    if isinstance(original_image, np.ndarray):
        original_image = torch.tensor(original_image, dtype=torch.float32)

    # 确保输入图像在 [0, 1] 范围内
    if original_image.max() > 1.0:  # 如果输入是 [0, 255] 范围，则归一化到 [0, 1]
        original_image = original_image / 255.0

    # 进行标准化
    mean = mean.to(original_image.device)
    std = std.to(original_image.device)

    normalized_image = (original_image - mean[None, :, None, None]) / std[None, :, None, None]

    return normalized_image

        
def test_train_2dunet(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step):
    # print(f"checkpoint_step:{checkpoint_step}")



    model = DiscriminativeSubNetwork_2d_att(in_channels=12, out_channels=2, base_channels=64).to(device)
    
    # optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.0001}])
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.001}], betas=(0.9, 0.98))

    checkpoint_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet/epoch_13131313.pth')

    if not os.path.exists(checkpoint_path):
        model.apply(weights_init_2d)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100 * 0.8, 100 * 0.9], gamma=0.2, last_epoch=-1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        start_epoch = 0

        # checkpoint_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_{0}.pth')
        # torch.save({
        #     'epoch': 0,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, checkpoint_path)
        # print(f'Saved checkpoint for epoch {0} at {checkpoint_path}')

    elif os.path.exists(checkpoint_path):
        print('resume', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100 * 0.8, 100 * 0.9], gamma=0.2, last_epoch=checkpoint['epoch']) # 恢复学习率调度器的状态
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    print('start_epoch', start_epoch)

    loss_focal = FocalLoss()

    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scaler = GradScaler()
    a_ = 0

    val_pipe.to(device)
    val_pipe.set_progress_bar_config(disable=True)
    val_pipe.unet.eval()
    val_pipe.vae.eval()
    val_pipe.text_encoder.eval()

    for epoch in range(start_epoch, 50):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch: " + str(epoch) + " Learning rate: " + str(lr))
        print('device', device)

        
        model.train()
        optimizer.zero_grad()  # 在每个 epoch 开始时清零梯度

        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):

            

            # model.train()
            a_ += 1
            
            image_input = batch["anomaly_images"].to(device)
            anomaly_mask = batch["anomaly_masks"].to(device)
            object_mask = batch["anomaly_masks"].to(device)
            instance_path = batch["instance_path"]

            # print('args.v', args.v)
            
            with torch.no_grad():
                
                start_time = time.time()

                image_input_512 = torch.nn.functional.interpolate(image_input, size=512, mode='bilinear', align_corners=True)
                reconstruct_images, step, result_qian, result_hou = reconstruction(val_pipe, weight_dtype, args, image_input_512, dino_frozen)

                # print('image_input.shape', image_input.shape) # [16, 3, 256, 256]
                # print('result_qian.shape', result_qian.shape) # [16, 3, 512, 512]
                # print('result_hou.shape', result_hou.shape) # [16, 3, 512, 512]
                # print('reconstruct_images.shape', reconstruct_images.shape) # [16, 3, 512, 512]

                elapsed_time = time.time()- start_time
                # print(f"代码运行时间1: {elapsed_time:.4f}秒")
                start_time = time.time()

                all_images = torch.cat([image_input_512, result_qian, result_hou, reconstruct_images], dim=0)


                all_images = torch.nn.functional.interpolate(all_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
                all_images = transform(all_images).to(dtype=weight_dtype)

                                    

                # print('all_images.shape', all_images.shape) # [64, 3, 256, 256]

                _, patch_tokens = dino_model(all_images) 

                patch_tokens_all= []
                batch_size = image_input.shape[0]  

                for i in range(len(patch_tokens)):
                    image_input_512 = patch_tokens[i][0:batch_size]
                    result_qian  = patch_tokens[i][batch_size:batch_size*2]
                    result_hou  = patch_tokens[i][batch_size*2:batch_size*3]
                    reconstruct_images  = patch_tokens[i][batch_size*3:batch_size*4]
                    patch_tokens_a = [None]*batch_size
                    # print('image_input_512.shape', image_input_512.shape)
                    # print('result_qian.shape', result_qian.shape)
                    # print('result_hou.shape', result_hou.shape)
                    # print('reconstruct_images.shape', reconstruct_images.shape)
                    for j in range(batch_size):
                        patch_tokens_a[j] = torch.cat([image_input_512[j].unsqueeze(0), result_qian[j].unsqueeze(0), result_hou[j].unsqueeze(0), reconstruct_images[j].unsqueeze(0)], dim=0)
                    # print('patch_tokens_a[j].shape', patch_tokens_a[j].shape)
                    patch_tokens_all.append(patch_tokens_a)


                # for i in range(len(patch_tokens)):
                #     patch_tokens_a = [None]*16
                #     for j in range(16):
                #         patch_tokens_a[j] = torch.cat([patch_tokens[i][0:16][j], patch_tokens[i][16:32][j], patch_tokens[i][32:48][j], patch_tokens[i][48:64][j]], dim=0)
                #         print('patch_tokens_a[j].shape', patch_tokens_a[j].shape)
                #     patch_tokens_all.append(patch_tokens_a)
                

                 

                
                








                


                # # 创建一个列表，用于存储结果
                # processed_images = []

                # # 按照指定的结构组合并添加到 processed_images 中
                # for i in range(batch_size):
                #     # 提取第 i 个 [4, 3, 256, 256] 张量，按顺序包含 image_input_512[i], result_qian[i], result_hou[i], reconstruct_images[i]
                #     group = torch.cat([
                #         all_images[i:i+1],               # 对应 image_input_512[i]，维度 [1, 3, 256, 256]
                #         all_images[batch_size + i:batch_size + i + 1],   # 对应 result_qian[i]，维度 [1, 3, 256, 256]
                #         all_images[2 * batch_size + i:2 * batch_size + i + 1],  # 对应 result_hou[i]，维度 [1, 3, 256, 256]
                #         all_images[3 * batch_size + i:3 * batch_size + i + 1]   # 对应 reconstruct_images[i]，维度 [1, 3, 256, 256]
                #     ], dim=0)  # 拼接成维度 [4, 3, 256, 256]
                    
                #     # 将结果添加到列表中
                    

                 

                #     _, patch_tokens = dino_model(all_images)
                #     print('len(patch_tokens)', len(patch_tokens)) # 4
                #     print(patch_tokens[0].shape) # [4,1025,768]
                #     print(patch_tokens[0][0].shape) # [1025,768]

                #     processed_images.append(group)



                # print('image_input.shape', image_input.shape)
                # print('result_qian.shape', result_qian.shape)
                # print('result_hou.shape', result_hou.shape)
                # print('reconstruct_images.shape', reconstruct_images.shape)

                
                
                
                # patch_tokens_i = patch_tokens[0]

                # patch_tokens_q = patch_tokens[1]
                # patch_tokens_h = patch_tokens[2]
                # patch_tokens_r = patch_tokens[3]

                # _, patch_tokens_i = dino_model(image_input.to(dtype=weight_dtype))
                # _, patch_tokens_r = dino_model(reconstruct_images.to(dtype=weight_dtype))

            model.train()

            # print('instance_path', instance_path)


            # patch_tokens_r_all = [ [tensor.squeeze(0).to(device) for tensor in latent] for latent in batch["patch_tokens_r_all"] ]

            # patch_tokens_i = [ tensor.squeeze(0).to(device) for tensor in batch["patch_tokens_i"] ]   


            # # save_path = instance_path[0].replace(".png", ".pt")
            # save_path = instance_path[0].replace("MVTec-AD", "MVTec-AD-pt").replace(".png", ".pt")

            # start_time = time.time()

            # loaded_data = torch.load(save_path)
            # patch_tokens_r_all = loaded_data["patch_tokens_r_all"]
            # patch_tokens_i = loaded_data["patch_tokens_i"]

            elapsed_time = time.time()- start_time
            # print(f"代码运行时间2: {elapsed_time:.4f}秒")
            start_time = time.time()

            # patch_tokens_r = patch_tokens_r_all[-1]


            # print('patch_tokens_i[0].shape', patch_tokens_i[0].shape)
            # print(len(patch_tokens_r_all))
            # print(len(patch_tokens_r))
            # print('patch_tokens_r[0].shape', patch_tokens_r[0].shape)

            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = 1, 1024, 768
            h = int(n ** 0.5)
            

            sigma_t = 6
            kernel_size_t = 2 * int(4 * sigma_t + 0.5) + 1
            

            # anomaly_maps1 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
            # anomaly_maps2 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)

            # print(patch_tokens_r_all[0][0]==patch_tokens_r_all[-1][0])


            # print('len(patch_tokens)', len(patch_tokens)) # 4
            # print(patch_tokens[0].shape) # [4,1025,768]
            # print(patch_tokens[0][0].shape) # [1025,768]

            # print(len(patch_tokens_all)) # 4
            # print(len(patch_tokens_all[0])) # 16
            # print(patch_tokens_all[0][0].shape) # [4, 1025, 768]
            anomaly_map_all_bat = []
            for bat in range(batch_size):
                anomaly_map_all = []
                anomaly_maps1 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device) # b=1
                for idx in range(len(patch_tokens_all)): # 4
                    anomaly_map_list = []
                    # print('idx', idx)
                    

                    for i in range(patch_tokens_all[0][0].shape[0]-1): # 0 1 2
                        # print('i', i)
                        # print('patch_tokens_all[idx][bat].shape', patch_tokens_all[idx][bat].shape) # [4, 1025, 768]
                        # print('patch_tokens_all[idx][bat][0].unsqueeze(0).shape', patch_tokens_all[idx][bat][0].unsqueeze(0).shape) # [1, 1025, 768]
                        # print('patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :].shape', patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :].shape) # [1, 1024, 768]

                        pi = patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :]
                        pr = patch_tokens_all[idx][bat][i+1].unsqueeze(0)[:, 1:, :]


            
                        pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
                        pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

                        cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
                        # print('torch.max(cos0)', torch.max(cos0))
                        # print('torch.min(cos0)', torch.min(cos0))

                        anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                        # print('torch.max(anomaly_map1)', torch.max(anomaly_map1))
                        # print('torch.min(anomaly_map1)', torch.min(anomaly_map1))

                        # anomaly_map1 = anomaly_map1/2.
                        # print('torch.max(anomaly_map1)', torch.max(anomaly_map1))
                        # print('torch.min(anomaly_map1)', torch.min(anomaly_map1))

                        # anomaly_map1, _ = torch.max(cos0, dim=-1)
                        # print('anomaly_map1.shape0 ', anomaly_map1.shape) # [1, 1024]
                        # max_pooled = F.max_pool1d(cos0, kernel_size=1024, stride=1024)
                        # max_pooled = max_pooled.view(cos0.shape[0],1024)
                        # # print('max_pooled.shape ', max_pooled.shape) #  
                        # equal = torch.allclose(anomaly_map1, max_pooled)
                        # # print("Are the results equal?", equal)

                        anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # print('anomaly_map1.shape1', anomaly_map1.shape) # [1, 1, 256, 256]
                        




                        # anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                        # # print('anomaly_map1000.shape',anomaly_map1.shape) #[1, 1024]
                        # anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # # print('anomaly_map1111.shape',anomaly_map1.shape) # [1, 1, 256, 256]
                        if i ==2:
                            anomaly_maps1 += anomaly_map1
                        # print('anomaly_map1.shape', anomaly_map1.shape) # [1, 1, 256, 256]


                        # print('anomaly_map1.shape', anomaly_map1.shape)
                        # anomaly_map1 = gaussian_blur2d(anomaly_map1, kernel_size=(kernel_size_t, kernel_size_t), sigma=(sigma_t, sigma_t))[:, 0]  # after interpolate cai shiyong gaussian_blur2d
                        # print('anomaly_map1.shape', anomaly_map1.shape)
                        anomaly_map1 = (anomaly_map1 - anomaly_map1.mean()) / (anomaly_map1.std() + 1e-6)
                        anomaly_map_list.append(anomaly_map1)
                      

                    anomaly_map_combined = torch.cat(anomaly_map_list, dim=1) 
                    # print('anomaly_map_combined.shape', anomaly_map_combined.shape) # [1, 3, 256, 256]
                    anomaly_map_all.append(anomaly_map_combined)


                anomaly_maps_all1 = torch.stack(anomaly_map_all).squeeze(1)
                # print('anomaly_maps_all1.shape0', anomaly_maps_all1.shape) # [4, 3, 256, 256]


                

                anomaly_maps1 = gaussian_blur2d(anomaly_maps1/4.0, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
                anomaly_maps = anomaly_maps1 #  * object_mask.to(device)
                anomaly_maps_np = anomaly_maps.detach().cpu().numpy()
                

                if idx_ % 10 == 0:
                    # print('instance_path', instance_path)
                    # print('pt_path', batch["pt_path"])
                    # print('a_', a_)
                    single_map = anomaly_maps_np
                    # print('single_map.shape0', single_map.shape) # (256, 256)
                    single_map = single_map * 255 # (single_map - single_map.min()) / (single_map.max() - single_map.min()) * 255
                    # print('single_map.shape1', single_map.shape) # (256, 256)
                    single_map = np.squeeze(single_map).astype(np.uint8)  # Convert to uint8 for image format compatibility
                    # print('single_map.shape2', single_map.shape) # (256, 256)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_train_bs16_fscratch/anomaly_maps_glad_{a_}_{bat}_{epoch}.png", single_map)
                    # print('anomaly_mask[bat].shape', anomaly_mask[bat].shape) # [1, 1, 256, 256]
                    # print('anomaly_mask[bat].detach().cpu().numpy()[0].shape', anomaly_mask[0].detach().cpu().numpy()[0].shape) # [256, 256]

                    sample = anomaly_mask[bat].detach().cpu().numpy()[0] * 255
                    # print(sample)
                    # print(anomaly_mask[bat].detach().cpu().numpy()[1] * 255)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_train_bs16_fscratch/in_masks_train_{a_}_{bat}_{epoch}.png", sample)


                if torch.isnan(anomaly_maps_all1).any() or torch.isinf(anomaly_maps_all1).any():
                    print("Input tensor contains NaN or Inf anomaly_maps_all1")

                

                

                

                # print('torch.max(anomaly_maps_all1)', torch.max(anomaly_maps_all1))
                # print('torch.min(anomaly_maps_all1)', torch.min(anomaly_maps_all1))


                anomaly_map_all_bat.append(anomaly_maps_all1)
            # print(len(anomaly_map_all_bat))
            # print('anomaly_map_all_bat[0].shape', anomaly_map_all_bat[0].shape) # [4, 3, 256, 256]

            anomaly_map_all_bat = [anomaly_map.view(-1, 256, 256) for anomaly_map in anomaly_map_all_bat]
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat)


            elapsed_time = time.time()- start_time
            # print(f"代码运行时间3: {elapsed_time:.4f}秒")
            start_time = time.time()
            # print('anomaly_map_all_bat.shape', anomaly_map_all_bat.shape) # [16, 12, 256, 256] 
            # print('anomaly_mask.shape', anomaly_mask.shape) # [16, 1, 256, 256] 
            # anomaly_mask = anomaly_mask.repeat(4, 1, 1, 1)
            optimizer.zero_grad()
            with autocast():
                output = model(anomaly_map_all_bat)
                # print('output.shape', output.shape) # (4, 2, 256, 256)
                output = torch.softmax(output, dim=1)
                # print('output.shape', output.shape) # (4, 2, 256, 256)
                segment_loss = loss_focal(output, anomaly_mask)
                # print('segment_loss', segment_loss)

            scaler.scale(segment_loss).backward()  # segment_loss.backward()
            # scaler.step(optimizer) # optimizer.step()
            # scaler.update()

            scaler.step(optimizer)  # 执行优化步骤
            scaler.update()  # 更新 scaler 的状态
            # optimizer.zero_grad()  # 清空梯度，准备下一次梯度累积

            # # 如果我们已经累积了足够的步数，就更新一次权重
            # if (idx_ + 1) % 2 == 0:  # 当累积步骤到达设定值
            #     scaler.step(optimizer)  # 执行优化步骤
            #     scaler.update()  # 更新 scaler 的状态
            #     optimizer.zero_grad()  # 清空梯度，准备下一次梯度累积


            anomaly_maps2 = output[:, 1, :, :].unsqueeze(1).detach()
            # anomaly_maps1 = output[:, 1, :, :].unsqueeze(1)

            if args.v != 0:
                distance_map = torch.mean(torch.abs(image_input - reconstruct_images), dim=1, keepdim=True)
                anomaly_maps2 = anomaly_maps2 + args.v * torch.max(anomaly_maps2) / torch.max(distance_map) * distance_map

            # anomaly_maps2 = gaussian_blur2d(anomaly_maps2, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
            anomaly_maps = anomaly_maps2 # * object_mask.to(device)
            anomaly_maps_np = anomaly_maps.detach().cpu().numpy()
            
            # print('idx_', idx_)
            # print('idx_ % 10', idx_ % 10)
            
            if idx_ % 10 == 0:
                for bat in range(batch_size):
                    
                    # print('j', j)
                    single_map = anomaly_maps_np[bat]
                    # print('single_map.shape0', single_map.shape)
                    single_map = single_map * 255 # (single_map - single_map.min()) / (single_map.max() - single_map.min()) * 255
                    # print('single_map.shape1', single_map.shape)
                    single_map = np.squeeze(single_map).astype(np.uint8)  # Convert to uint8 for image format compatibility
                    # print('single_map.shape2', single_map.shape)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_train_bs16_fscratch/anomaly_maps_my_{a_}_{bat}_{epoch}.png", single_map)
                    # print(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_train_bs16/anomaly_maps_my_{a_}_{bat}_{epoch}.png")

            # elapsed_time = time.time()- start_time
            # print(f"代码运行时间2: {elapsed_time:.4f}秒")     
            # start_time = time.time()

            current_losses = {'segment_loss': segment_loss,  }
            log_losses_tensorboard(writer, current_losses, a_)

            elapsed_time = time.time()- start_time
            # print(f"代码运行时间4: {elapsed_time:.4f}秒")
            start_time = time.time()

            if idx_ % 100 == 0:
                # Save model and optimizer states
                checkpoint_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_{epoch + 1}_{idx_}_fscratch.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                }, checkpoint_path)
                print(f'Saved checkpoint for epoch {epoch + 1} of {idx_} at {checkpoint_path}')

        # scheduler.step()
        scheduler.step(segment_loss)

        # Save model and optimizer states
        checkpoint_path = os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_{epoch + 1}_fscratch.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
        }, checkpoint_path)
        print(f'Saved checkpoint for epoch {epoch + 1} at {checkpoint_path}')
            




def test_2dunet(model, dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step, log_file, model_num, iftest):
    print(f"test:{class_name}")
 
    

    
    a_ = 0
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


        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
            model.eval()
            a_ += 1
            image_input = batch["instance_images"].to(device)
            anomaly_mask = batch["instance_masks"].to(device)
            object_mask = batch["object_mask"].to(device)
            instance_path = batch["instance_path"]


            # print('args.v', args.v)
              
            image_input_512 = torch.nn.functional.interpolate(image_input, size=512, mode='bilinear', align_corners=True)
            reconstruct_images, step, result_qian, result_hou = reconstruction(val_pipe, weight_dtype, args, image_input_512, dino_frozen, iftest)
            all_images = torch.cat([image_input_512, result_qian, result_hou, reconstruct_images], dim=0)
            all_images = torch.nn.functional.interpolate(all_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
            all_images = transform(all_images).to(dtype=weight_dtype)

 

            _, patch_tokens = dino_model(all_images) 
            patch_tokens_all= []
            batch_size = image_input.shape[0]  

            for i in range(len(patch_tokens)):
                image_input_512 = patch_tokens[i][0:batch_size]
                result_qian  = patch_tokens[i][batch_size:batch_size*2]
                result_hou  = patch_tokens[i][batch_size*2:batch_size*3]
                reconstruct_images  = patch_tokens[i][batch_size*3:batch_size*4]
                patch_tokens_a = [None]*batch_size
                # print('image_input_512.shape', image_input_512.shape)
                # print('result_qian.shape', result_qian.shape)
                # print('result_hou.shape', result_hou.shape)
                # print('reconstruct_images.shape', reconstruct_images.shape)
                for j in range(batch_size):
                    patch_tokens_a[j] = torch.cat([image_input_512[j].unsqueeze(0), result_qian[j].unsqueeze(0), result_hou[j].unsqueeze(0), reconstruct_images[j].unsqueeze(0)], dim=0)
                # print('patch_tokens_a[j].shape', patch_tokens_a[j].shape)
                patch_tokens_all.append(patch_tokens_a)


            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = 1, 1024, 768
            h = int(n ** 0.5)
            
            sigma_t = 6
            kernel_size_t = 2 * int(4 * sigma_t + 0.5) + 1

            anomaly_map_all_bat = []
            for bat in range(batch_size):
                anomaly_map_all = []
                anomaly_maps1 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
                for idx in range(len(patch_tokens_all)): # 4
                    anomaly_map_list = []
                    # print('idx', idx)
                    

                    for i in range(patch_tokens_all[0][0].shape[0]-1): # 0 1 2
                        # print('i', i)
                        # print('patch_tokens_all[idx][bat].shape', patch_tokens_all[idx][bat].shape) # [4, 1025, 768]
                        # print('patch_tokens_all[idx][bat][0].unsqueeze(0).shape', patch_tokens_all[idx][bat][0].unsqueeze(0).shape) # [1, 1025, 768]
                        # print('patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :].shape', patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :].shape) # [1, 1024, 768]

                        pi = patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :]
                        pr = patch_tokens_all[idx][bat][i+1].unsqueeze(0)[:, 1:, :]


            
                        pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
                        pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

                        cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
                        # print('torch.max(cos0)', torch.max(cos0))
                        # print('torch.min(cos0)', torch.min(cos0))

                        anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                        # print('torch.max(anomaly_map1)', torch.max(anomaly_map1))
                        # print('torch.min(anomaly_map1)', torch.min(anomaly_map1))

                        # anomaly_map1 = anomaly_map1/2.
                        # print('torch.max(anomaly_map1)', torch.max(anomaly_map1))
                        # print('torch.min(anomaly_map1)', torch.min(anomaly_map1))

                        # anomaly_map1, _ = torch.max(cos0, dim=-1)
                        # print('anomaly_map1.shape0 ', anomaly_map1.shape) # [1, 1024]
                        # max_pooled = F.max_pool1d(cos0, kernel_size=1024, stride=1024)
                        # max_pooled = max_pooled.view(cos0.shape[0],1024)
                        # # print('max_pooled.shape ', max_pooled.shape) #  
                        # equal = torch.allclose(anomaly_map1, max_pooled)
                        # # print("Are the results equal?", equal)

                        anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # print('anomaly_map1.shape1', anomaly_map1.shape) # [1, 1, 256, 256]
                        # anomaly_map_list.append(anomaly_map1)




                        # anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                        # # print('anomaly_map1000.shape',anomaly_map1.shape) #[1, 1024]
                        # anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # # print('anomaly_map1111.shape',anomaly_map1.shape) # [1, 1, 256, 256]
                        if i ==2:
                            anomaly_maps1 += anomaly_map1
                        # print('anomaly_map1.shape', anomaly_map1.shape) # [1, 1, 256, 256]

                        
 


                        # anomaly_map1 = gaussian_blur2d(anomaly_map1, kernel_size=(kernel_size_t, kernel_size_t), sigma=(sigma_t, sigma_t))[:, 0]  # after interpolate cai shiyong gaussian_blur2d
                        # anomaly_map1 = (anomaly_map1 - anomaly_map1.mean()) / (anomaly_map1.std() + 1e-6)
                        anomaly_map_list.append(anomaly_map1)


                    anomaly_map_combined = torch.cat(anomaly_map_list, dim=1) 
                    # print('anomaly_map_combined.shape', anomaly_map_combined.shape) # [1, 3, 256, 256]
                    anomaly_map_all.append(anomaly_map_combined)


                anomaly_maps_all1 = torch.stack(anomaly_map_all).squeeze(1)
                # print('anomaly_maps_all1.shape0', anomaly_maps_all1.shape) # [4, 3, 256, 256]


                

                anomaly_maps1 = gaussian_blur2d(anomaly_maps1/4.0, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
                anomaly_maps = anomaly_maps1 #  * object_mask.to(device)
                anomaly_maps_np = anomaly_maps.detach().cpu().numpy()
                

                if idx_ % 5 == 0:
                    # print('instance_path', instance_path)
                    # print('pt_path', batch["pt_path"])
                    # print('a_', a_)
                    
                     
                    single_map = anomaly_maps_np
                    # print('single_map.shape0', single_map.shape) # (256, 256)
                    single_map = single_map * 255 # (single_map - single_map.min()) / (single_map.max() - single_map.min()) * 255
                    # print('single_map.shape1', single_map.shape) # (256, 256)
                    single_map = np.squeeze(single_map).astype(np.uint8)  # Convert to uint8 for image format compatibility
                    # print('single_map.shape2', single_map.shape) # (256, 256)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_test_bs16/anomaly_maps_glad_{a_}_{bat}_test_bs16_{model_num}.png", single_map)
                    # print('anomaly_mask[bat].shape', anomaly_mask[bat].shape) # [1, 1, 256, 256]
                    # print('anomaly_mask[bat].detach().cpu().numpy()[0].shape', anomaly_mask[0].detach().cpu().numpy()[0].shape) # [256, 256]

                    sample = anomaly_mask[bat].detach().cpu().numpy()[0] * 255
                    # print(sample)
                    # print(anomaly_mask[bat].detach().cpu().numpy()[1] * 255)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_test_bs16/in_masks_test_{a_}_{bat}_test_bs16_{model_num}.png", sample)


                if torch.isnan(anomaly_maps_all1).any() or torch.isinf(anomaly_maps_all1).any():
                    print("Input tensor contains NaN or Inf anomaly_maps_all1")

                

                

                

                # print('torch.max(anomaly_maps_all1)', torch.max(anomaly_maps_all1))
                # print('torch.min(anomaly_maps_all1)', torch.min(anomaly_maps_all1))


                anomaly_map_all_bat.append(anomaly_maps_all1)
            # print(len(anomaly_map_all_bat))
            # print('anomaly_map_all_bat[0].shape', anomaly_map_all_bat[0].shape) # [4, 3, 256, 256]

            anomaly_map_all_bat = [anomaly_map.view(-1, 256, 256) for anomaly_map in anomaly_map_all_bat]
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat)


            
            # print('anomaly_map_all_bat.shape', anomaly_map_all_bat.shape) # [16, 12, 256, 256] 
            # print('anomaly_mask.shape', anomaly_mask.shape) # [16, 1, 256, 256] 
            # anomaly_mask = anomaly_mask.repeat(4, 1, 1, 1)
            

            if torch.isnan(anomaly_maps_all1).any() or torch.isinf(anomaly_maps_all1).any():
                print("Input tensor contains NaN or Inf anomaly_maps_all1")

            

            with autocast():
                output = model(anomaly_map_all_bat)
                # print('output.shape', output.shape) # (4, 2, 256, 256)
                output = torch.softmax(output, dim=1)
                # print('output.shape', output.shape) # (4, 2, 256, 256)
                # segment_loss = loss_focal(output, anomaly_mask)


            anomaly_maps1 = output[:, 1, :, :].unsqueeze(1)
            # anomaly_maps1 = output[:, 1, :, :].unsqueeze(1)
            # print('anomaly_maps1.shape', anomaly_maps1.shape)



            

             

            anomaly_maps1 = gaussian_blur2d(anomaly_maps1, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
            anomaly_maps = anomaly_maps1 * object_mask.to(device)

            # print('anomaly_maps1.shape', anomaly_maps1.shape)
            # anomaly_maps = anomaly_maps1.squeeze(0) * object_mask.to(device)
            # print('anomaly_maps.shape', anomaly_maps.shape)

            anomaly_maps_np = anomaly_maps.detach().cpu().numpy()
            
            if idx_ % 5 == 0:
                for bat in range(batch_size):
                     
                    single_map = anomaly_maps_np[bat]
                    # print('single_map.shape0', single_map.shape)
                    single_map = single_map * 255 # (single_map - single_map.min()) / (single_map.max() - single_map.min()) * 255
                    # print('single_map.shape1', single_map.shape)
                    single_map = np.squeeze(single_map).astype(np.uint8)  # Convert to uint8 for image format compatibility
                    # print('single_map.shape2', single_map.shape)
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_test_bs16/anomaly_map_my_test_{a_}_{bat}_bs16_{model_num}.png", single_map)


                    sample = reverse_normalization(torch.tensor(image_input[bat]))

                    sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_test_bs16/gt_images_test_{a_}_{bat}_bs16_{model_num}.png", sample)



                    sample = object_mask[bat].detach().cpu().numpy() * 255
                    cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/model/costvolume_2_grid_test_bs16/object_mask_test_{a_}_{bat}_bs16_{model_num}.png", sample)


            
            score = torch.topk(torch.flatten(anomaly_maps, start_dim=1), 250)[0].mean(dim=1)
            print('score.shape', score.shape)
            masks.extend([m for m in anomaly_mask[:, 0, :, :].cpu().numpy()])
            preds.extend([a for a in anomaly_maps.cpu().numpy()])
            scores.extend([s for s in score.cpu().numpy()])
            labels.extend([l for l in batch["instance_label"].cpu().numpy()])



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
            auroc_pixel, 4) * 100, round(AP_pixel, 4) * 100, round(best_f1_scores_pixel, 4) * 100, round(pro, 4) * 100



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
            print('torch.sum(rotated_img_tensor==img_ref)', torch.sum(rotated_img_tensor==img_ref))
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


def get_class_labels(instance_path, device):
    """
    根据路径从 CLSNAMES 中匹配类别，并返回类别索引的 Tensor。
    
    :param instance_path: list 或 iterable，包含图像路径
    :param CLSNAMES: list，包含类别名称的列表
    :param clsname_to_index: dict，类别名称到索引的映射
    :param device: 设备（'cpu' 或 'cuda'），用来指定输出张量所在的设备
    :return: Tensor，包含每个路径对应的类别索引
    """
    # 初始化分类标签
    cls_labels = []
    # print('clsname_to_index', clsname_to_index)

    for path in instance_path:  # 遍历所有路径
        matched = False  # 标记是否找到匹配类别
        # print('path', path)
        for cls_name in CLSNAMES:
            # print('cls_name', cls_name)

            if cls_name in path:  # 判断类别是否出现在路径中

                cls_labels.append(clsname_to_index[cls_name])
                matched = True
                break
        
        if not matched:
            print(f"Warning: No matching class found for path {path}")
            cls_labels.append(-1)  # 如果没有匹配类别，用 -1 表示未知类别

    # 将类别标签转换为 Tensor 并转移到指定设备
    cls_labels_tensor = torch.tensor(cls_labels, dtype=torch.long).to(device)
    
    return cls_labels_tensor




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




def compute_background_mask(img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        if isinstance(img_features, torch.Tensor):
            img_features = img_features.cpu().detach().numpy()
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))

        print('first_pc values:', first_pc[:10])  # 查看前 10 个主成分值

        print('first_pc.shape', first_pc.shape) # (1024, 1)

        print('Threshold:', threshold)
        print('First PC max:', first_pc.max())
        print('First PC min:', first_pc.min())

        if masking_type == True:
            mask = first_pc > threshold
            # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            print('m.shape', m.shape)
            print('m sum:', m.sum())
            print('m.size * 0.35', m.size * 0.35)

            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)

        # mask = torch.tensor(mask.squeeze(), dtype=torch.bool)
        return mask.squeeze()



def convert_path(instance_path, save_root, file_extension=".pt"):
    """
    将输入路径转换为指定的输出路径，并修改文件扩展名。

    Args:
        instance_path (str): 输入的文件路径。
        save_root (str): 目标根目录。
        file_extension (str): 输出文件的扩展名 (默认 .pt)。

    Returns:
        str: 转换后的输出路径。
    """
    # 获取相对路径（去掉原始根目录部分）
    relative_path = os.path.relpath(instance_path, start="/home/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets")

    # 更改根目录并替换文件扩展名
    out_path = os.path.join(save_root, os.path.splitext(relative_path)[0] + file_extension)

    instance_dir = os.path.dirname(out_path)
    # 如果目录不存在，就创建它
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir, exist_ok=True)

    print('instance_path', instance_path)
    print('out_path', out_path)
    print('instance_dir', instance_dir)

    return out_path




def train_save_pt_anomaldino(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step, epoch_num, seed):
 
    
    save_root = "/home/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_1234"
    with torch.no_grad():
       
 
        transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

  

        print("Length of val_dataloader:", len(val_dataloader))
        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
            
            data_dict = {
                "instance_images": None,  # 初始化一个包含25个None的列表，用于存储不同iter的latents
                
                "instance_masks": None,  # 预留位置，稍后可以填充具体数据
                "instance_path": None,
                "instance_label": None,

                # "img_augmented": None,


                "features_ref_3": None,
                "features_ref_6": None,
                "features_ref_9": None,
                "features_ref_12": None,

                # 你可以添加其他需要的键
               
            }
            
            instance_images = batch["instance_images"].to(device)
            instance_masks = batch["instance_masks"].to(device)
            instance_path = batch["instance_path"]
            instance_label = batch["instance_label"].to(device)
            print(instance_path)
            print(instance_label)


            # print('instance_path', instance_path)
            
            batch_size = instance_images.shape[0]  
            class_labels = get_class_labels(instance_path, device)

            out_dir = convert_path(instance_path[0], save_root)
            # out_dirs = [convert_path(path, save_root) for path in instance_paths]


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
            

            object_name = get_object_name_from_path(instance_path, masking_default)
            print('object_name', object_name)

            mask_ref_images = False
            # masking = masking_default[object_name] 
            print('masking_default[object_name[0]]', masking_default[object_name[0]])
            


            with torch.no_grad():
                features_ref_3 = []
                features_ref_6 = []
                features_ref_9 = []
                features_ref_12 = []


                start_time = time.time()

                print('instance_images.shape', torch.tensor(instance_images.shape)) # [  1,   3, 256, 256]
                print('torch.tensor(instance_images[0]).shape', torch.tensor(instance_images[0]).shape) #  3 256 256
                print('reverse_normalization(torch.tensor(instance_images[0])).squeeze().shape', reverse_normalization(torch.tensor(instance_images[0])).squeeze().shape) #  3 256 256
                img_augmented = augment_image(reverse_normalization(torch.tensor(instance_images[0])).squeeze())   

                print('len(img_augmented)', len(img_augmented)) # 8
                print(torch.stack(img_augmented).shape) # [8, 3, 256, 256]


                img_augmented_stack = torch.stack(img_augmented)
                # img_augmented_before_aug = torch.stack(img_augmented)

                img_augmented_stack = normalization(img_augmented_stack) # 等价于 transformer_compose


                # print(torch.max(instance_images[0]))
                # print(torch.min(instance_images[0]))

                # print(torch.max(img_augmented_stack[0]))
                # print(torch.min(img_augmented_stack[0]))


                # tolerance = 1e-4  # 设置容忍误差
                # comparison = torch.isclose(instance_images[0], img_augmented_stack[0], atol=tolerance)

                # # 统计相等元素的数量
                # num_equal = torch.sum(comparison)

                # print('num_equal', num_equal)







                img_augmented = torch.nn.functional.interpolate(img_augmented_stack, size=args.dino_resolution, mode='bilinear', align_corners=True)

                img_augmented = transform(img_augmented)
                print('img_augmented.shape', img_augmented.shape) # [8, 3, 256, 256]

                

                with torch.no_grad():
                    for i_aug in range(len(img_augmented)):
                        
                        print('img_augmented[i_aug].shape', img_augmented[i_aug].shape) # [3, 256, 256]
                        # print('dino_model', dino_model)

                        _, patch_tokens_i_tem_aug = dino_model(img_augmented[i_aug].unsqueeze(0).to(dtype=weight_dtype))
                        patch_tokens_i_tem_aug  = [feature[:, 1:, :].squeeze(0).float().cpu().numpy() for feature in patch_tokens_i_tem_aug]  
                        print('patch_tokens_i_tem_aug[-1].shape', patch_tokens_i_tem_aug[-1].shape) # [1024, 768]

                        mask_ref = compute_background_mask(patch_tokens_i_tem_aug[-1].squeeze(), (32,32), threshold=3, masking_type=(mask_ref_images and masking_default[object_name[0]]))
                        print('mask_ref.shape', mask_ref.shape) # (1024,)
                        print('torch.sum(torch.tensor(mask_ref).float())', torch.sum(torch.tensor(mask_ref).float()))
                        features_ref_3.append(patch_tokens_i_tem_aug[0][mask_ref])
                        features_ref_6.append(patch_tokens_i_tem_aug[1][mask_ref])
                        features_ref_9.append(patch_tokens_i_tem_aug[2][mask_ref])
                        features_ref_12.append(patch_tokens_i_tem_aug[3][mask_ref])

                    features_ref_3 = np.concatenate(features_ref_3, axis=0)
                    features_ref_6 = np.concatenate(features_ref_6, axis=0)
                    features_ref_9 = np.concatenate(features_ref_9, axis=0)
                    features_ref_12 = np.concatenate(features_ref_12, axis=0)
                    print('features_ref_12.shape', features_ref_12.shape) # [8*1024, 768]

                # image_input_512 = torch.nn.functional.interpolate(anomaly_images, size=512, mode='bilinear', align_corners=True)
                # reconstruct_latent, step, latents_all, denoise_s_hou = reconstruction(val_pipe, weight_dtype, args, image_input_512, dino_frozen)

            data_dict = {
                "instance_images": torch.tensor(instance_images).to(dtype=torch.float16),  # 初始化一个包含25个None的列表，用于存储不同iter的latents
                
                "instance_masks": torch.tensor(instance_masks).to(dtype=torch.float16),  # 预留位置，稍后可以填充具体数据
                "instance_path": instance_path,
                "instance_label": torch.tensor(instance_label).to(dtype=torch.float16),

                # "img_augmented": img_augmented,


                "features_ref_3": torch.tensor(features_ref_3).to(dtype=torch.float16),
                "features_ref_6": torch.tensor(features_ref_6).to(dtype=torch.float16),
                "features_ref_9": torch.tensor(features_ref_9).to(dtype=torch.float16),
                "features_ref_12": torch.tensor(features_ref_12).to(dtype=torch.float16),

                # 你可以添加其他需要的键
               
            }
            if idx_ % 20 == 0:
                print(instance_images.shape) # [1, 3, 256, 256]
                print(instance_masks.shape) # [1, 1, 256, 256]
                print(img_augmented_stack[0].unsqueeze(0).shape) # [1, 3, 256, 256]

                sample = reverse_normalization(torch.tensor(instance_images))
                sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                cv2.imwrite(f"/home/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_{seed}/train_instance_images_{seed}_{idx_}.png", sample)


                sample = instance_masks.squeeze(0).detach().cpu().numpy()[0] * 255
                cv2.imwrite(f"/home/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_{seed}/train_instance_masks_{seed}_{epoch_num}_{idx_}.png", sample)



                sample = torch.tensor(mask_ref).float().squeeze().view(32,32).detach().cpu().numpy()* 255
                cv2.imwrite(f"/home/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_{seed}/train_mask_ref_{seed}_{epoch_num}_{idx_}.png", sample)

                for i in range(img_augmented_stack.shape[0]):
                    sample = reverse_normalization(torch.tensor(img_augmented_stack[i].unsqueeze(0)))
                    sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(f"/home/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_{seed}/train_img_augmented_{seed}_{idx_}_{i}.png", sample)



            print(type(out_dir)) # str
            torch.save(data_dict, out_dir)
            print('save', out_dir)
            


def test_save_pt(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step, epoch_num, seed):
 
   
 
    with torch.no_grad():
        val_pipe.to(device)
        val_pipe.set_progress_bar_config(disable=True)
        val_pipe.unet.eval()
        val_pipe.vae.eval()
        val_pipe.text_encoder.eval()
 
        transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        preds = []
        masks = []
        scores = []
        labels = []

        print("Length of val_dataloader:", len(val_dataloader))
        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
            
            data_dict = {
                "latents_all": [None] * 25,  # 初始化一个包含25个None的列表，用于存储不同iter的latents
                "batch": {
                    "image_input": None,  # 预留位置，稍后可以填充具体数据
                    "anomaly_mask": None,
                    "object_mask": None,
                    "instance_label": None,
                    "instance_path": None,
                    "idx_hou": None,
                    "reconstruct_latent": None,
                    # 你可以添加其他需要的键
                }
            }
            
            image_input = batch["instance_images"].to(device)
            anomaly_mask = batch["instance_masks"].to(device)
            instance_path = batch["instance_path"]
            object_mask = batch["object_mask"].to(device)
            instance_label = batch["instance_label"].to(device)

            # print('instance_path', instance_path)
            
            with torch.no_grad():
                
                start_time = time.time()

                image_input_512 = torch.nn.functional.interpolate(image_input, size=512, mode='bilinear', align_corners=True)
                reconstruct_latent, step, latents_all, denoise_s_hou = reconstruction(val_pipe, weight_dtype, args, image_input_512, dino_frozen)

            data_dict = {
                "latents_all": latents_all,  # 初始化一个包含25个None的列表，用于存储不同iter的latents
                "batch": {
                    "image_input": image_input,  # 预留位置，稍后可以填充具体数据
                    "anomaly_mask": anomaly_mask,
                    "object_mask": object_mask,
                    "instance_label": instance_label,
                    "instance_path": instance_path,
                    "idx_hou": denoise_s_hou,
                    "reconstruct_latent": reconstruct_latent,
                    # 你可以添加其他需要的键
                }
                }
            if idx_ % 10 == 0:
                # print(image_input.shape)
                # print(anomaly_mask.shape)
                sample = reverse_normalization(torch.tensor(image_input))

                sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/test_save_path_{seed}/test_images_epoch_denoise_{class_name}_{epoch_num}_{idx_}.png", sample)


                sample = anomaly_mask.squeeze(0).detach().cpu().numpy()[0] * 255
                cv2.imwrite(f"/home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/test_save_path_{seed}/test_masks_epoch_denoise_{class_name}_{epoch_num}_{idx_}.png", sample)

            file_name = f'/home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/test_save_path_{seed}/test_epoch_denoise_{class_name}_{epoch_num}_{idx_}.pt'
            torch.save(data_dict, file_name)
            
        


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




    writer = SummaryWriter(log_dir='/home/ZZ/anomaly/GLAD-main/model/cost_volume/logs_test_train_2dunet')
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
            CLSNAMES = [
                'hazelnut',
                "capsule",
                'grid',
                'carpet',
                'leather',
                'tile',
                'wood',
                'bottle',
                'cable',
                'metal_nut',
                'pill',
                'screw',
                'toothbrush',
                'transistor',
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
            torch.load(f"/home/customer/Desktop/ZZ/anomaly/GLAD-main/{args.output_dir}/checkpoint-{checkpoint_step}/pytorch_model.bin", map_location='cpu')
        )

        val_pipe.unet.to(device).to(dtype=torch.float16)
        val_pipe.vae = load_vae(val_pipe.vae).to(device)

        print(args)
        performances = [[], [], [], [], [], [], []]

        test_2d = False
        train_2d = False
        train_2d_save = True
        test_2d_save = False

        if test_2d:
            start_ = 1
            # model_paths = [os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_1_{i}_fscratch.pth') for i in range(100,500,100)]
            model_paths = [os.path.join('/home/ZZ/anomaly/GLAD-main/model/cost_volume/test_train_2dunet', f'epoch_9.pth')]

            log_file_path = "/home/ZZ/anomaly/GLAD-main/model/cost_volume/log_9.txt"

            # 打开 log 文件，并将每次模型测试结果写入其中
            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入

                # 遍历所有模型路径
                for i, model_path in enumerate(model_paths):
                    log_file.write(f"Model: {model_path}\n")
                    
                    i = i+start_
                    # 持续检测模型路径是否存在
                    model_found = False
                    while not os.path.exists(model_path):
                        if not model_found:
                            print(f"等待模型: {model_path} 加载...")
                            model_found = True  # 只打印一次等待信息
                        time.sleep(10)  # 每隔10秒检查一次路径是否存在

                    # 当路径存在时，加载模型
                    print(f"加载模型: {model_path}")
                    model = DiscriminativeSubNetwork_2d_att(in_channels=12, out_channels=2, base_channels=64).to(device)
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])

                    performances = [[], [], [], [], [], [], []]



 

 


                    for class_name in CLSNAMES:
                        args.instance_prompt = 'a photo of sks' + class_name
                        print(class_name, f"input_threshold is {args.input_threshold} {args.denoise_step} {args.min_step} {args.v}")

                        fix_seeds(args.seed)
                        val_dataset = MVTecDataset2(
                            instance_data_root=args.instance_data_dir,
                            instance_prompt=args.instance_prompt,
                            class_name=class_name,
                            tokenizer=None,
                            resize=args.resolution,
                            img_size=args.resolution,
                            train=False
                        )
                        val_dataloader = DataLoader(
                            val_dataset,
                            batch_size=1,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=True,
                        )
         





                        
                        results = test_2dunet(model, 
                            dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                            class_name,
                            checkpoint_step, log_file, i, test_2d
                        )


                        for j, result in enumerate(results):
                            performances[j].append(result)
                    performances = np.array(performances).T
                    print("mean:", np.mean(performances, axis=0))

                    # 将结果写入 log 文件
                    log_file.write(f"Model: {model_path}\n")
                    log_file.write(f"Mean Performance: {np.mean(performances, axis=0)}\n\n")
                    log_file.flush()  # 确保写入文件



        

        elif train_2d:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=False,
            )
            fix_seeds(args.seed)
            
            train_dataset = MVTecDataset2(    # MVTecDataset1   MVTecDataset
                instance_data_root=args.instance_data_dir,
                instance_prompt=args.instance_prompt,
                class_name="",
                tokenizer=tokenizer,
                resize=args.resolution,
                img_size=args.resolution,
                anomaly_path=args.anomaly_data_dir,
                train=True
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=8,
                shuffle=True,
                num_workers= 4,  # args.dataloader_num_workers,              test_train_3dunet      test
                drop_last=True
            )   

 

            print('train_2dunet')

            test_train_2dunet(
                    dino_model, dino_frozen, val_pipe, torch.float16, train_dataloader, args, device,
                    None,
                    checkpoint_step
                )
        elif train_2d_save:

            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=False,
            )


            CLSNAMES = [
                'hazelnut',
                "capsule",
                'grid',
                'carpet',
                'leather',
                'tile',
                'wood',
                'bottle',
                'cable',
                'metal_nut',
                'pill',
                'screw',
                'toothbrush',
                'transistor',
                'zipper',
            ]
            clsname_to_index = {name: idx for idx, name in enumerate(CLSNAMES)}
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=False,
            )

            fix_seeds(args.seed)
            
            train_dataset = MVTecDataset_anomaldino(    # MVTecDataset1   MVTecDataset
                instance_data_root=args.instance_data_dir,
                instance_prompt=args.instance_prompt,
                class_name="",
                tokenizer=tokenizer,
                resize=args.resolution,
                img_size=args.resolution,
                anomaly_path=args.anomaly_data_dir,
                train=True
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers= 4,  # args.dataloader_num_workers,              test_train_3dunet      test
                drop_last=True
            )   

 

            print('train_save_pt')
            for epoch_num in range(1):
                train_save_pt_anomaldino(
                        dino_model, dino_frozen, val_pipe, torch.float16, train_dataloader, args, device,
                        None,
                        checkpoint_step, epoch_num, args.seed
                    )


        elif test_2d_save:
            for class_name in CLSNAMES:
                args.instance_prompt = 'a photo of sks' + class_name
                print(class_name, f"input_threshold is {args.input_threshold} {args.denoise_step} {args.min_step} {args.v}")

                fix_seeds(args.seed)
                val_dataset = MVTecDataset(
                    instance_data_root=args.instance_data_dir,
                    instance_prompt=args.instance_prompt,
                    class_name=class_name,
                    tokenizer=None,
                    resize=args.resolution,
                    img_size=args.resolution,
                    train=False
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    num_workers=4,
                    shuffle=False,
                    pin_memory=True,
                )
                print('test_save_pt')
                results = test_save_pt(
                    dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                    class_name,
                    checkpoint_step, 0, args.seed
                )
               