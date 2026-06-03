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


from creat_model import model
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from utilize.utilize import normalize, fix_seeds, compute_pro, reconstruction
from tensorboardX import SummaryWriter

from creat_model import get_vit_encoder
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import cv2
import random
import warnings
from loss import FocalLoss, SSIM, SoftIoULoss, FocalLoss_gamma

from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel

from torch.cuda.amp import autocast, GradScaler
from PIL.ImageOps import exif_transpose
from PIL import Image
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--train", type=bool)

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="./CompVis/stable-diffusion-v1-4",
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


class CustomDataset(Dataset):
    def __init__(self, directory, device):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.pt')]
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载.pt文件
        file_path = os.path.join(self.directory, self.files[idx])
        data_dict = torch.load(file_path, map_location='cpu')

        # 提取需要的数据
        # latents_all = data_dict["latents_all"].to(self.device)

        anomaly_images = data_dict["batch"]["anomaly_images"].to(self.device)
        anomaly_masks = data_dict["batch"]["anomaly_masks"].to(self.device)
        instance_label = data_dict["batch"]["instance_label"].to(self.device)
        instance_path = data_dict["batch"]["instance_path"]
        idx_hou = data_dict["batch"]["idx_hou"]
        reconstruct_latent = data_dict["batch"]["reconstruct_latent"].to(self.device)

        instance_image = Image.open(instance_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.transformer_compose(instance_image).to(self.device)

        latents_all = [torch.tensor(latent).to(self.device) for latent in data_dict["latents_all"]]
        return latents_all, instance_image, anomaly_images, anomaly_masks, instance_label, instance_path, idx_hou, reconstruct_latent


class CustomDataset_dict(Dataset):
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

        anomaly_images = data_dict["batch"]["anomaly_images"]
        anomaly_masks = data_dict["batch"]["anomaly_masks"]
        instance_label = data_dict["batch"]["instance_label"]
        instance_path = data_dict["batch"]["instance_path"]
        idx_hou = self.pad_tensor(torch.tensor(data_dict["batch"]["idx_hou"]).squeeze()).unsqueeze(0)
        reconstruct_latent = data_dict["batch"]["reconstruct_latent"]

        latents_all = [torch.tensor(latent) for latent in data_dict["latents_all"]]
        instance_image = anomaly_images.clone()
        # instance_image = Image.open(data_dict["batch"]["instance_path"][0])
        # instance_image = exif_transpose(instance_image)
        # if not instance_image.mode == "RGB":
        #     instance_image = instance_image.convert("RGB")
        # instance_image = self.transformer_compose(instance_image)
        # print('instance_path', instance_path)
        return latents_all, instance_image, anomaly_images, anomaly_masks, instance_label, instance_path, idx_hou, reconstruct_latent

    def pad_tensor(self, tensor, target_length=25):
        # 如果张量的长度小于目标长度
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

    for path in instance_path[0]:  # 遍历所有路径
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


def compute_dynamic_gamma(cls_out, target, dino_resolution, num_classes=15):
    """
    Compute gamma dynamically for each image based on classification output (cls_out).
    The idea is to increase gamma (focus on hard examples) when classification confidence is low,
    and decrease gamma (focus on easy examples) when confidence is high.

    Additionally, if the classification is wrong, increase gamma to focus more on hard examples.
    """
    # cls_out has shape [batch_size, num_classes]

    # Define difficult classes that should always have high gamma (set to 4)
    difficult_classes = ['candle',
                'capsules',
                'cashew',
                'macaroni1',
                'macaroni2',
                'pcb1',
                'pcb2',
                'pcb3',
                'pcb4',
                ]

    # Create a mapping of class index to class name (reverse the clsname_to_index mapping)
    clsname_to_index = {name: idx for idx, name in enumerate([
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


def train_2dunet_dino_triloss_volume(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step):
    # print(f"checkpoint_step:{checkpoint_step}")

    model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=1024, out_channels=2, base_channels=48).to(device)

    # model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=1024, out_channels=2, base_channels=48).to(device)


    # model = DiscriminativeSubNetwork_2d_att_dino_cost(in_channels=4, out_channels=2, base_channels=64).to(device)
    model = model.float()
    # optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.0001}])
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.001}], betas=(0.9, 0.98))

    checkpoint_path = os.path.join('./checkpoints_path/epoch_5_4500_fsc4545454165841+++++ratch_channel48min.pth')

    if not os.path.exists(checkpoint_path):
        model.apply(weights_init_2d)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100 * 0.8, 100 * 0.9], gamma=0.2, last_epoch=-1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        start_epoch = 0

        # checkpoint_path = os.path.join('./checkpoints_path', f'epoch_{0}.pth')
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


    loss_focal = FocalLoss_gamma() # FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scaler = GradScaler()
    a_ = 0
    a__tensornoaed = 0

    val_pipe.to(device)
    val_pipe.set_progress_bar_config(disable=True)
    val_pipe.unet.eval()
    val_pipe.vae.eval()
    val_pipe.text_encoder.eval()

    for epoch in range(start_epoch, 12):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch: " + str(epoch) + " Learning rate: " + str(lr))
        print('device', device)

        
        model.train()
        optimizer.zero_grad()  # 在每个 epoch 开始时清零梯度

        index_mapping = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):

        # for idx_, batch in enumerate(val_dataloader):

            start_time = time.time()
            latents_all, instance_image, image_input, anomaly_mask, instance_label, instance_path, denoise_s_hou1, reconstruct_latent = batch


            # instance_image = instance_image.squeeze().to(device)
            image_input = image_input.squeeze().to(device)
            anomaly_mask = anomaly_mask.to(device)
            instance_label = instance_label.to(device)
            denoise_s_hou1 = denoise_s_hou1.to(device)
            reconstruct_latent = reconstruct_latent.squeeze().to(device)


            batch_size = image_input.shape[0]  
            denoise_s_hou  = [None] * batch_size

            # print('instance_path', instance_path)
            class_labels = get_class_labels(instance_path, device)
            # print('class_labels', class_labels)
            # print('denoise_s_hou1.shape', denoise_s_hou1.shape)
            # print('denoise_s_hou1', denoise_s_hou1)
            for j in range(batch_size):
                denoise_s_hou[j] = denoise_s_hou1[j][denoise_s_hou1[j] != 0]
            # print('denoise_s_hou', denoise_s_hou)
            # model.train()
            a_ += 1
            
            result  = [None] * batch_size
            idx_hou1 = [None] * batch_size
            idx_hou2 = [None] * batch_size
            for j in range(batch_size): 
                
                # idx_hou1[j], idx_hou2[j] = (denoise_s_hou[j][-1], denoise_s_hou[j][-2]) if len(denoise_s_hou[j]) > 2 else (denoise_s_hou[j][-1], denoise_s_hou[j][-1])
                # print('idx_hou1[j]', idx_hou1[j])
                # print('idx_hou2[j]', idx_hou2[j])
                # print(j)
                
                if len(denoise_s_hou[j]) > 2:
                    denoise_s_hou[j] = denoise_s_hou[j][:-1]
                # print('denoise_s_hou[j]', denoise_s_hou[j])
                # idx_hou1[j], idx_hou2[j] = random.sample(denoise_s_hou[j], 2) if len(denoise_s_hou[j]) > 1 else (denoise_s_hou[j][-1], denoise_s_hou[j][-1])

                if len(denoise_s_hou[j]) > 2:
                    # 随机选择 2 个元素的索引
                    indices = torch.randperm(len(denoise_s_hou[j]), device=denoise_s_hou[j].device)[:2]
                    idx_hou1[j], idx_hou2[j] = denoise_s_hou[j][indices[0]], denoise_s_hou[j][indices[1]]
                else:
                    # 如果元素少于 2 个，选择最后一个元素重复作为两个索引
                    idx_hou1[j], idx_hou2[j] = denoise_s_hou[j][-1], denoise_s_hou[j][-1]

                # print('idx_hou1[j]1', idx_hou1[j])
                # print('idx_hou2[j]1', idx_hou2[j])
                # print(j)


                result[j] = [latents_all[idx_hou1[j]][j, :, :, :].squeeze(), latents_all[idx_hou2[j]][j, :, :, :].squeeze()]
            # print('args.v', args.v)
            result = torch.stack([torch.stack(sublist) for sublist in result])
            with torch.no_grad():
                # print('reconstruct_latent.shape', reconstruct_latent.shape)
                # print('result[:,0,:,:,:].shape', result[:,0,:,:,:].shape)
                # print('result[:,1,:,:,:].shape', result[:,1,:,:,:].shape)
                reconstruct_images = val_pipe.vae.decode(reconstruct_latent.to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]
                result_qian = val_pipe.vae.decode(result[:,0,:,:,:].to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]
                result_hou = val_pipe.vae.decode(result[:,1,:,:,:].to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]



                image_input_512 = torch.nn.functional.interpolate(image_input, size=512, mode='bilinear', align_corners=True)
                # reconstruct_images, _, result_qian, result_hou = reconstruction(val_pipe, weight_dtype, args, image_input_512, dino_frozen)
                reconstruct_images__ = torch.nn.functional.interpolate(reconstruct_images, size=256, mode='bilinear', align_corners=True)
                # print('image_input.shape', image_input.shape) # [16, 3, 256, 256]
                # print('result_qian.shape', result_qian.shape) # [16, 3, 512, 512]
                # print('result_hou.shape', result_hou.shape) # [16, 3, 512, 512]
                # print('reconstruct_images.shape', reconstruct_images.shape) # [16, 3, 512, 512]

                # elapsed_time = time.time()- start_time
                # print(f"代码运行时间1: {elapsed_time:.4f}秒")
                # start_time = time.time()

                all_images = torch.cat([image_input_512, result_qian, result_hou, reconstruct_images], dim=0)


                all_images = torch.nn.functional.interpolate(all_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
                all_images = transform(all_images)

                                    

                # print('all_images.shape', all_images.shape) # [64, 3, 256, 256]

                _, patch_tokens = dino_model(all_images.to(dtype=weight_dtype)) 
                # print('instance_image.shape', instance_image.shape) # [64, 3, 256, 256]
                # print('transform(image_input).shape', transform(image_input).shape)
                # print('reconstruct_images.shape', reconstruct_images.shape)
                # _, dino_features = dino_model(image_input_512.to(dtype=weight_dtype))
                _, dino_features = dino_model(transform(image_input).to(dtype=weight_dtype)) 
                patch_tokens_all= []
                
                # print('dino_features[0].shape', dino_features[0].shape)
                # print('patch_tokens[0].shape', patch_tokens[0].shape)

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

            # elapsed_time = time.time()- start_time
            # print(f"代码运行时间2: {elapsed_time:.4f}秒")
            # start_time = time.time()

            # patch_tokens_r = patch_tokens_r_all[-1]


            # print('patch_tokens_i[0].shape', patch_tokens_i[0].shape)
            # print(len(patch_tokens_r_all))
            # print(len(patch_tokens_r))
            # print('patch_tokens_r[0].shape', patch_tokens_r[0].shape)

            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = 1, 1024, 768
            h = int(n ** 0.5)
            

            # sigma_t = 6
            # kernel_size_t = 2 * int(4 * sigma_t + 0.5) + 1
            

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
            min_similarity_map_all_bat = []

            for bat in range(batch_size):
                min_similarity_map_all = []
                anomaly_map_all = []
                anomaly_maps111 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device) # b=1
                min_similarity_map_all_1 = torch.zeros(3, 4, 32, 32).to(device) 
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
                        # print('anomaly_map1.shape', anomaly_map1.shape) # [1, 1024]
                        # max_pooled = F.max_pool1d(cos0, kernel_size=1024, stride=1024)
                        # max_pooled = max_pooled.view(cos0.shape[0],1024)
                        # # print('max_pooled.shape ', max_pooled.shape) #  
                        # equal = torch.allclose(anomaly_map1, max_pooled)
                        # # print("Are the results equal?", equal)
                        # print('anomaly_map1.view(1, 32, 32).shape ', anomaly_map1.view(1, 32, 32).shape)
                        min_similarity_map_all.append(anomaly_map1.view(1, 32, 32))
                        anomaly_map11 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # print('anomaly_map1.shape1', anomaly_map1.shape) # [1, 1, 256, 256]
                        
                        


                        # # print('1 - cos0.shape', (1 - cos0).shape) # [1, 1024, 1024]
                        pre_min_dim = 1024 # 48
                        one_minus_cos0 = 1 - cos0





                        # print('one_minus_cos0.shape', one_minus_cos0.shape) # [1, 1024, 1024]
                        _, indices = torch.topk(one_minus_cos0, pre_min_dim, dim=-1, largest=False, sorted=False)
                        # indices= torch.sort(indices, dim=1)[0]
                        # print('indices.shape', indices.shape) # [1, 1024, 512]
                        # print('indices', indices)
                        # print('torch.sort(indices, dim=-1)[0]', torch.sort(indices, dim=-1)[0])
                        selected_values = torch.gather(one_minus_cos0, dim=-1, index=torch.sort(indices, dim=-1)[0])
                        # print('selected_values.shape', selected_values.shape) # [1, 1024, 512]
                        anomal_map_3d = selected_values.view(1, 32, 32, pre_min_dim).unsqueeze(1)
                        # print('anomal_map_3d.shape', anomal_map_3d.shape) # [1, 1, 32, 32, 1024]


                        # anomal_map_3d = one_minus_cos0.view(1, 32, 32, pre_min_dim).unsqueeze(1)

                        # print('anomal_map_3d.shape', anomal_map_3d.shape) # [1, 1, 32, 32, 1024]




                        # anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                        # # print('anomaly_map1000.shape',anomaly_map1.shape) #[1, 1024]
                        # anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        # # print('anomaly_map1111.shape',anomaly_map1.shape) # [1, 1, 256, 256]
                        if i ==2:
                            anomaly_maps111 += anomaly_map11
                        # print('anomaly_map1.shape', anomaly_map1.shape) # [1, 1, 256, 256]


                        # print('anomaly_map1.shape', anomaly_map1.shape)
                        # anomaly_map1 = gaussian_blur2d(anomaly_map1, kernel_size=(kernel_size_t, kernel_size_t), sigma=(sigma_t, sigma_t))[:, 0]  # after interpolate cai shiyong gaussian_blur2d
                        # print('anomaly_map1.shape', anomaly_map1.shape)
                        # anomaly_map1 = (anomaly_map1 - anomaly_map1.mean()) / (anomaly_map1.std() + 1e-6)
                        anomaly_map_list.append(anomal_map_3d)
                      

                    anomaly_map_combined = torch.cat(anomaly_map_list, dim=1) 
                    # print('anomaly_map_combined.shape', anomaly_map_combined.shape) # [1, 3, 32, 32, 1024]
                    anomaly_map_all.append(anomaly_map_combined)


                # print('torch.stack(anomaly_map_all).shape', torch.stack(anomaly_map_all).shape) # [4, 1, 3, 32, 32, 1024]
                anomaly_maps_all1 = torch.stack(anomaly_map_all).squeeze().view(12, 32, 32, pre_min_dim) # view之前 [4, 1, 3, 32, 32, 1024]
                # print('anomaly_maps_all1.shape', anomaly_maps_all1.shape) # [12, 32, 32, 1024]

                # anomaly_maps_all1 = anomaly_maps_all1.permute(0, 3, 1, 2)
                anomaly_maps_all1 = F.interpolate(anomaly_maps_all1.permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=True)
                # print('anomaly_maps_all1.shape', anomaly_maps_all1.shape) # [12, 1024, 64, 64]
                anomaly_maps_all1 = anomaly_maps_all1.view(4, 3, pre_min_dim, 64, 64).permute(1, 2, 0, 3, 4)
                # print('anomaly_maps_all1.shape', anomaly_maps_all1.shape) # [3, 1024, 4, 64, 64]


                min_similarity_map_all = torch.stack(min_similarity_map_all).squeeze()
                # print('min_similarity_map_all.shape', min_similarity_map_all.shape) # [12, 32, 32]

                for i_ in range(3):
                    for j_ in range(4):
                        min_similarity_map_all_1[i_, j_] = min_similarity_map_all[index_mapping[i_][j_]]


                anomaly_maps111 = gaussian_blur2d(anomaly_maps111/4.0, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
                anomaly_maps_np = anomaly_maps111.detach().cpu().numpy()
                


                if torch.isnan(anomaly_maps_all1).any() or torch.isinf(anomaly_maps_all1).any():
                    print("Input tensor contains NaN or Inf anomaly_maps_all1")

                

                
                # 验证值是否正确
                # a = min_similarity_map_all_1
                # assert torch.equal(a[0, 0], min_similarity_map_all[0])  # True
                # assert torch.equal(a[0, 1], min_similarity_map_all[3])  # True
                # assert torch.equal(a[0, 2], min_similarity_map_all[6])  # True
                # assert torch.equal(a[0, 3], min_similarity_map_all[9])  # True
                # assert torch.equal(a[1, 0], min_similarity_map_all[1])  # True
                # assert torch.equal(a[1, 1], min_similarity_map_all[4])  # True
                # assert torch.equal(a[1, 2], min_similarity_map_all[7])  # True
                # assert torch.equal(a[1, 3], min_similarity_map_all[10])  # True
                # assert torch.equal(a[2, 0], min_similarity_map_all[2])  # True
                # assert torch.equal(a[2, 1], min_similarity_map_all[5])  # True
                # assert torch.equal(a[2, 2], min_similarity_map_all[8])  # True
                # assert torch.equal(a[2, 3], min_similarity_map_all[11])  # True
                # print("All assertions passed!")

                # print('torch.max(anomaly_maps_all1)', torch.max(anomaly_maps_all1))
                # print('torch.min(anomaly_maps_all1)', torch.min(anomaly_maps_all1))

                min_similarity_map_all_bat.append(min_similarity_map_all_1)
                anomaly_map_all_bat.append(anomaly_maps_all1)
            # print(len(anomaly_map_all_bat))
            # print('anomaly_map_all_bat[0].shape', anomaly_map_all_bat[0].shape) # [12, 1024, 64, 64]

            # anomaly_map_all_bat = [anomaly_map.view(-1, 256, 256) for anomaly_map in anomaly_map_all_bat]
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat).float().permute(1, 0, 2, 3, 4, 5).contiguous()
            # print('anomaly_map_all_bat.shape', anomaly_map_all_bat.shape) # [3, 8, 512, 4, 64, 64]



            min_similarity_map_all_bat = torch.stack(min_similarity_map_all_bat).float()
            # print('min_similarity_map_all_bat.shape', min_similarity_map_all_bat.shape) # [8, 3, 4, 32, 32]

            min_similarity_map_all_bat = F.interpolate(min_similarity_map_all_bat.view(8, -1, 32, 32), size=64, mode='bilinear', align_corners=True).view(8, 3, 4, 64, 64) 

            # print('min_similarity_map_all_bat.shape', min_similarity_map_all_bat.shape) # [8, 3, 4, 64, 64]
            # elapsed_time = time.time()- start_time
            # print(f"代码运行时间3: {elapsed_time:.4f}秒")
            # start_time = time.time()
            
            anomaly_mask = anomaly_mask.squeeze(1).float() # .repeat(4, 1, 1, 1)
            # print('anomaly_mask.shape', anomaly_mask.shape) # [16, 1, 256, 256] 


            dino_features = [feature[:, 1:, :].float() for feature in dino_features]
            # dino_features = [feature.unsqueeze(1).expand(-1, pre_min_dim, -1, -1) for feature in dino_features]

            # if torch.isnan(dino_features[0]).any() or torch.isinf(dino_features[0]).any():
            #     print("Input tensor contains NaN or Inf dino_features[0]")
            # if torch.isnan(dino_features[1]).any() or torch.isinf(dino_features[1]).any():
            #     print("Input tensor contains NaN or Inf dino_features[1]")
            # if torch.isnan(dino_features[2]).any() or torch.isinf(dino_features[2]).any():
            #     print("Input tensor contains NaN or Inf dino_features[2]")
            # if torch.isnan(dino_features[3]).any() or torch.isinf(dino_features[3]).any():
            #     print("Input tensor contains NaN or Inf dino_features[3]")

            # print('dino_features[0].shape', dino_features[0].shape)
            # print('dino_features[1].shape', dino_features[1].shape)
            # print('dino_features[2].shape', dino_features[2].shape)
            # print('dino_features[3].shape', dino_features[3].shape) # [8, 1024, 768]

            # optimizer.zero_grad()
            # with autocast():
            #     output = model(anomaly_map_all_bat, dino_features)
            #     # print('output.shape', output.shape) # (4, 2, 256, 256)
            #     output_focl = torch.softmax(output, dim=1)
            #     # print('output_focl.shape', output_focl.shape) # (4, 2, 256, 256)
            #     segment_loss = loss_focal(output_focl, anomaly_mask)
            #     # ce_loss = nn.CrossEntropyLoss()(output, anomaly_mask.squeeze().long())
            #     # print('segment_loss', segment_loss)
            #     loss = segment_loss # + 0.5 * ce_loss

            # if torch.isnan(loss) or torch.isinf(loss):
            #     print(f"Skipping batch {idx_} due to invalid loss: {loss}")
            #     print('instance_path', instance_path)
            #     continue  # 跳过当前 batch

            # scaler.scale(loss).backward()  # loss.backward()
            

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm()}")

            # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step(optimizer)  # 执行优化步骤
            # scaler.update()  # 更新 scaler 的状态
            

            

            # print('anomaly_map_all_bat[0].shape', anomaly_map_all_bat[0].shape) # [8, 1024, 4, 64, 64]
            # print('dino_features[3].shape', dino_features[3].shape) # [8, 1024, 768]*4
            # print('min_similarity_map_all_bat[:, 1].shape', min_similarity_map_all_bat[:, 1].shape) # [8, 4, 64, 64] 
            
            
            for i_all in range(anomaly_map_all_bat.shape[0]):

                optimizer.zero_grad()
                # output, cls_out = model(anomaly_map_all_bat[i_all], dino_features)

                output, cls_out = model(anomaly_map_all_bat[i_all], dino_features, min_similarity_map_all_bat[:, i_all])
     
                # print('cls_out.shape', cls_out.shape)
                # print('output.shape', output.shape)

                # print('class_labels.shape', class_labels.shape) # [8]
                classification_loss = criterion(cls_out, class_labels.long())



                output_focl = torch.softmax(output, dim=1)
                # print('output_focl.shape', output_focl.shape) # [4, 2, 64, 64]
                # output_focl, _ = torch.min(output_focl, dim=2)
                # print('output_focl.shape', output_focl.shape) # [4, 2, 64, 64]

                output_focl = F.interpolate(output_focl, size=args.dino_resolution, mode='bilinear', align_corners=True)  # Shape: [4, 2, 256, 256]
                # print('output_focl.shape', output_focl.shape) # [4, 2, 64, 64]

                anomaly_prob = output_focl[:, 1, :, :].unsqueeze(1)


                # print('anomaly_prob.shape', anomaly_prob.shape)  # [8, 1, 256, 256]
                # print('anomaly_mask.shape', anomaly_mask.shape) # [8, 1, 256, 256]
                # print('output_focl.shape', output_focl.shape)  # [8, 2, 256, 256]
                # print('anomaly_mask.shape', anomaly_mask.shape)  # [8, 1, 256, 256]

                # Calculate dynamic gamma based on classification output
                gamma = compute_dynamic_gamma(cls_out, class_labels.float(), args.dino_resolution, num_classes=15).to(device)
                # print('compute_dynamic_gamma', gamma)
                # Focal_loss = loss_focal(output_focl.float(), anomaly_mask.float(), gamma)
                
                # start_time = time.time()

                # Focal_loss1 = loss_focal(output_focl.float(), anomaly_mask.float())

                # elapsed_time = time.time()- start_time
                # print(f"代码运行时间1: {elapsed_time:.4f}秒")
                # start_time = time.time()


                loss_images = []
                for i in range(batch_size):
                    loss_i = loss_focal(output_focl.float()[i:i+1], anomaly_mask.float()[i:i+1], gamma[i])  # For each image, slice the batch
                    loss_images.append(loss_i)

                # Average of losses for each image
                Focal_loss = sum(loss_images) / len(loss_images)
                # elapsed_time = time.time()- start_time
                # print(f"代码运行时间2: {elapsed_time:.4f}秒")

                # print(Focal_loss==Focal_loss1)
                # print(Focal_loss)
                # print(Focal_loss1)



                ssim_loss = loss_ssim(anomaly_prob.float(), anomaly_mask.float())


                # SL1_loss = F.smooth_l1_loss(anomaly_prob.float(), anomaly_mask.float())

                l2_loss = loss_l2(anomaly_prob.float(), anomaly_mask.float())


                SIOU_loss = SoftIoULoss(anomaly_prob.float(), anomaly_mask.float())

                alpha, beta, gamma, yita = 1.0, 0.2, 1.0, 0.1
                segmentation_loss = alpha * Focal_loss + beta * ssim_loss + gamma * l2_loss + yita * SIOU_loss
                # ce_loss = nn.CrossEntropyLoss()(output, anomaly_mask.squeeze().long())
                # print('segmentation_loss', segmentation_loss)
                # loss = Focal_loss # + 0.5 * ce_loss

                

                loss = segmentation_loss + 0.5 * classification_loss 

                

                

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {idx_} due to invalid loss: {loss}")
                    print('instance_path', instance_path)
                    continue  # 跳过当前 batch

                loss.backward() 
            

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient for {name}: {param.grad.norm()}")

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()  # 执行优化步骤

                a__tensornoaed += 1


                # 准确率计算
                predicted_classes = torch.argmax(cls_out, dim=1)
                correct_predictions = (predicted_classes == class_labels).sum().item()
                total_samples = class_labels.size(0)
                train_accuracy = correct_predictions / total_samples
                # print('train_accuracy', train_accuracy)


                current_losses = {'Focal_loss': alpha * Focal_loss, 'sum_anomaly_prob':torch.sum(anomaly_prob), 'ssim_loss': beta * ssim_loss, 'l2_loss': gamma * l2_loss, 'SIOU_loss': yita * SIOU_loss, 'cls_loss': 0.5 * classification_loss, 'seg_loss': segmentation_loss, 'train_accuracy': train_accuracy, } # 0.5 * ce_loss
                # print(current_losses)
                log_losses_tensorboard(writer, current_losses, a__tensornoaed)





           

            anomaly_maps2 = output_focl[:, 1, :, :].unsqueeze(1).detach()
            # anomaly_maps1 = output_focl[:, 1, :, :].unsqueeze(1)

            # if args.v != 0:
            #     # print('transform(image_input).shape', transform(image_input).shape)
            #     # print('reconstruct_images__.shape', reconstruct_images__.shape)
            #     distance_map = torch.mean(torch.abs(transform(image_input) - reconstruct_images__), dim=1, keepdim=True)
            #     anomaly_maps2 = anomaly_maps2 + args.v * torch.max(anomaly_maps2) / torch.max(distance_map) * distance_map

            # anomaly_maps2 = gaussian_blur2d(anomaly_maps2, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
            anomaly_maps_np = anomaly_maps2.detach().cpu().numpy()
            
            # print('idx_', idx_)
            # print('idx_ % 10', idx_ % 10)
            
           
            if idx_ % 300 == 0:
                # Save model and optimizer states
                checkpoint_path = os.path.join('./checkpoints_path', f'epoch_{epoch + 1}_{idx_}_fscratch_channel48min.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                }, checkpoint_path)
                print(f'Saved checkpoint for epoch {epoch + 1} of {idx_} at {checkpoint_path}')

            if idx_ % 2000 == 0:
                scheduler.step(loss)

        # # scheduler.step()
        # scheduler.step(Focal_loss)

        # Save model and optimizer states
        checkpoint_path = os.path.join('./checkpoints_path', f'epoch_{epoch + 1}_fscratch_channel48min.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
        }, checkpoint_path)
        print(f'Saved checkpoint for epoch {epoch + 1} at {checkpoint_path}')
            




def load_vae(vae):
    print(args.instance_data_dir)
    if "VisA" in args.instance_data_dir:
        vae_path = './model_Glad/Multi-class_VisA/VisA_VAE/multi-class_visa_epoch=118-step=64498.ckpt'
    elif 'PCBBank' in args.instance_data_dir:
        vae_path = 'model/vae/pacbank_epoch=245-step=64944.ckpt'
    else:
        vae_path = None

    if vae_path:
        sd = torch.load(vae_path, map_location='cpu')["state_dict"]
        print(f"load vae in test :{vae_path}")

        keys = list(sd.keys())
        for k in keys:
            if "loss" in k:
                del sd[k]
        vae.load_state_dict(sd) #  vae.load_state_dict(sd, map_location='cpu')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dino_model, val_pipe = load_test_model(args, torch.float16)
    dino_frozen = copy.deepcopy(dino_model)

    args.output_dir = os.path.join('model', args.instance_data_dir.split('/')[-1] + '_' + args.output_dir + f"_{args.seed}")




    writer = SummaryWriter(log_dir='./checkpoints_path')
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
                torch.load("./model_Glad/Multi-class_VisA/VisA_20000step_bs32_eps_anomaly2_multiclass_0_1234/checkpoint-20000/pytorch_model.bin", map_location="cpu"))


        val_pipe.unet.to(device).to(dtype=torch.float16)
        val_pipe.vae = load_vae(val_pipe.vae).to(device)

        print(args)
        performances = [[], [], [], [], [], [], []]

        test_2d = False
        train_2d = False
        train_2d_save = False
        test_2d_save = False

        train_2d_dino  = True
        test_2d_dino  = False

        if train_2d_dino:
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
            clsname_to_index = {name: idx for idx, name in enumerate(CLSNAMES)}

            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_fast=False,
            )
            fix_seeds(args.seed)

            data_directory = '/home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_Glad/Costfilter_Glad_VisA/train_save_path_1234_VISA'
            train_cost_dataset = CustomDataset_dict(data_directory, device, args.resolution)
            print('data_directory', data_directory)
            train_cost_dataloader = DataLoader(
                train_cost_dataset,
                batch_size=8,
                shuffle=True,  # 如果需要随机打乱数据
                num_workers=4,  # 根据你的机器调整
                pin_memory=True,
                drop_last=True
                )


            train_2dunet_dino_triloss_volume(dino_model, dino_frozen, val_pipe, torch.float16, train_cost_dataloader, args, device, None, checkpoint_step)

