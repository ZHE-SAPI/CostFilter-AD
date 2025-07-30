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

from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel

from torch.cuda.amp import autocast, GradScaler
from PIL.ImageOps import exif_transpose
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


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


class CustomDataset_dict_test(Dataset):
    # Here, since Glad training and inference time is too long, 
    # we first save the features reconstructed by GLAD and the features of the input image offline to a notebook in pt format 
    # (We save the diversity of abnormal synthetic images for training data (for costfilter) over 10 Glad epochs, for testing data (for costfilter) over 1 Glad epochs)
    # When generating training/inference data, glad's input is the training data in mvtec with synthetic anomalies/ teating data in mvtec, and its output is the corresponding reconstructed image.
    
    # Of course, you can also choose not to save it as a pt format file and directly output it from glad, but it will take relative long time.

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
       

        image_input = data_dict["batch"]["image_input"]
        anomaly_mask = data_dict["batch"]["anomaly_mask"]
        object_mask = data_dict["batch"]["object_mask"]
        instance_path = data_dict["batch"]["instance_path"]
        instance_label = data_dict["batch"]["instance_label"]

        idx_hou = self.pad_tensor(torch.tensor(data_dict["batch"]["idx_hou"]).squeeze()).unsqueeze(0)
        reconstruct_latent = data_dict["batch"]["reconstruct_latent"]
        
        latents_all = [torch.tensor(latent) for latent in data_dict["latents_all"]]
       
        instance_image = image_input
        

        return latents_all, instance_image, image_input, anomaly_mask, instance_label, instance_path, idx_hou, reconstruct_latent, object_mask

    def pad_tensor(self, tensor, target_length=25):
        # 如果张量的长度小于目标长度
        if tensor.size(0) < target_length:
            padding = target_length - tensor.size(0)  # 计算需要填充的长度
            tensor = F.pad(tensor, (padding, 0))  # 在前面填充，后面不填充
        return tensor

    def transformer_compose(self, image, mask=None):
        transforms_resize = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.BILINEAR)
        
        transforms_to_tensor = transforms.ToTensor()
        
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transforms_normalize = transforms.Normalize(mean, std)

        image = transforms_resize(image)
        
        image = transforms_to_tensor(image)
        image = transforms_normalize(image)

        if mask:
            mask = transforms_resize(mask)
            mask = transforms_to_tensor(mask)
            
            return image, mask
        return image

 

            
def test_2dunet_dino_DRAEM(model, dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step, log_file, len_dataset, a_, beta):
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


        total_pixel_scores = np.zeros((img_dim * img_dim * len_dataset))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len_dataset))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        index_mapping = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

        for idx_, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
            model.eval()
            a_ += 1
            latents_all, _, image_input, anomaly_mask, instance_label, instance_path, denoise_s_hou1, reconstruct_latent, object_mask = batch

            image_input = image_input.squeeze().to(device)
            anomaly_mask = anomaly_mask.to(device)
            instance_label = instance_label.to(device)
            denoise_s_hou1 = denoise_s_hou1.to(device)
            reconstruct_latent = reconstruct_latent.squeeze().to(device)
            object_mask = object_mask.to(device)


            batch_size = image_input.shape[0]  
            denoise_s_hou  = [None] * batch_size



            is_normal = instance_label.cpu().numpy()
            for i in range(is_normal.shape[0]):
                anomaly_score_gt.append(is_normal[i])
            true_mask = anomaly_mask.squeeze(1)
            true_mask_cv = true_mask.detach().cpu().numpy()[:, :, :, :].transpose((0,2,3,1)) # .transpose((1, 2, 0))
            
            for j in range(batch_size):
                denoise_s_hou[j] = denoise_s_hou1[j][denoise_s_hou1[j] != 0]
            
             
            result  = [None] * batch_size
            idx_hou1 = [None] * batch_size
            idx_hou2 = [None] * batch_size
            for j in range(batch_size): 
                
                idx_hou1[j], idx_hou2[j] = (denoise_s_hou[j][-1], denoise_s_hou[j][-2]) if len(denoise_s_hou[j]) > 2 else (denoise_s_hou[j][-1], denoise_s_hou[j][-1])
                
                result[j] = [latents_all[idx_hou1[j]][j, :, :, :].squeeze(), latents_all[idx_hou2[j]][j, :, :, :].squeeze()]
            result = torch.stack([torch.stack(sublist) for sublist in result])
            with torch.no_grad():
               
                reconstruct_images = val_pipe.vae.decode(reconstruct_latent.to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]
                result_qian = val_pipe.vae.decode(result[:,0,:,:,:].to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]
                result_hou = val_pipe.vae.decode(result[:,1,:,:,:].to(device) / val_pipe.vae.config.scaling_factor, return_dict=False)[0]


            image_input_512 = torch.nn.functional.interpolate(image_input, size=512, mode='bilinear', align_corners=True)
            all_images = torch.cat([image_input_512, result_qian, result_hou, reconstruct_images], dim=0)
            all_images = torch.nn.functional.interpolate(all_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
            all_images = transform(all_images)

            _, patch_tokens = dino_model(all_images.to(dtype=weight_dtype)) 
            _, dino_features = dino_model(transform(image_input).to(dtype=weight_dtype)) 
            patch_tokens_all= []

            for i in range(len(patch_tokens)):
                image_input_512 = patch_tokens[i][0:batch_size]
                result_qian  = patch_tokens[i][batch_size:batch_size*2]
                result_hou  = patch_tokens[i][batch_size*2:batch_size*3]
                reconstruct_images  = patch_tokens[i][batch_size*3:batch_size*4]
                patch_tokens_a = [None]*batch_size
                
                for j in range(batch_size):
                    patch_tokens_a[j] = torch.cat([image_input_512[j].unsqueeze(0), result_qian[j].unsqueeze(0), result_hou[j].unsqueeze(0), reconstruct_images[j].unsqueeze(0)], dim=0)
                patch_tokens_all.append(patch_tokens_a)


            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = 1, 1024, 768
            h = int(n ** 0.5)
            
             

            anomaly_map_all_bat = []
            anomaly_map_all_bat12 = []
            min_similarity_map_all_bat = []
            anomaly_maps1g12 = torch.zeros((b*batch_size, 1, args.dino_resolution, args.dino_resolution)).to(device)
            anomaly_maps2g12 = torch.zeros((b*batch_size, 1, args.dino_resolution, args.dino_resolution)).to(device)
            for bat in range(batch_size):
                min_similarity_map_all = []
                anomaly_map_all = []
                anomaly_map_all12 = []
                anomaly_maps1g = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
                anomaly_maps2g = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
                min_similarity_map_all_1 = torch.zeros(3, 4, 32, 32).to(device) 
                for idx in range(len(patch_tokens_all)): # 4
                    anomaly_map_list = []
                    anomaly_map_list12 = []
                    

                    for i in range(patch_tokens_all[0][0].shape[0]-1): # 0 1 2
                        
                        pi = patch_tokens_all[idx][bat][0].unsqueeze(0)[:, 1:, :]
                        pr = patch_tokens_all[idx][bat][i+1].unsqueeze(0)[:, 1:, :]

                        pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
                        pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

                        cos0 = torch.bmm(pi, pr.permute(0, 2, 1))
                        
                        anomaly_map1, _ = torch.min(1 - cos0, dim=-1)

                        min_similarity_map_all.append(anomaly_map1.view(1, 32, 32))
                        
                        anomaly_map11 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 
                        

                        pre_min_dim = 1024 # 48
                        one_minus_cos0 = 1 - cos0
                        _, indices = torch.topk(one_minus_cos0, pre_min_dim, dim=-1, largest=False, sorted=False)
                        
                        selected_values = torch.gather(one_minus_cos0, dim=-1, index=torch.sort(indices, dim=-1)[0])

                        anomal_map_3d = selected_values.view(1, 32, 32, pre_min_dim).unsqueeze(1)

                        if class_name in ['transistor', 'pcb1', 'pcb4']:
                            anomaly_map12, _ = torch.min(1 - cos0, dim=-2)
                            anomaly_map12 = F.interpolate(anomaly_map12.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True) 

                        if i ==2:
                            anomaly_maps1g += anomaly_map11
                            
                            if class_name in ['transistor', 'pcb1', 'pcb4']:
                                anomaly_maps2g += anomaly_map12

                        anomaly_map_list.append(anomal_map_3d)

                    anomaly_map_combined = torch.cat(anomaly_map_list, dim=1)
                    anomaly_map_all.append(anomaly_map_combined)

                anomaly_maps_all1 = torch.stack(anomaly_map_all).squeeze().view(12, 32, 32, pre_min_dim) # view之前 [4, 1, 3, 32, 32, 1024]

                anomaly_maps_all1 = F.interpolate(anomaly_maps_all1.permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=True)
                anomaly_maps_all1 = anomaly_maps_all1.view(4, 3, pre_min_dim, 64, 64).permute(1, 2, 0, 3, 4)

                min_similarity_map_all = torch.stack(min_similarity_map_all).squeeze()
                
                for i_ in range(3):
                    for j_ in range(4):
                        min_similarity_map_all_1[i_, j_] = min_similarity_map_all[index_mapping[i_][j_]]


                anomaly_maps1g = anomaly_maps1g/len(patch_tokens_all)
                
                if class_name in ["transistor", "pcb1", "pcb4"]:
                    anomaly_maps2g = anomaly_maps2g/len(patch_tokens_all)
                    anomaly_maps1g = (anomaly_maps1g + anomaly_maps2g) / 2
                else:
                    anomaly_maps1g = anomaly_maps1g

                anomaly_maps1g = gaussian_blur2d(anomaly_maps1g, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
                anomaly_maps1g12[bat] = anomaly_maps1g.unsqueeze(0)

                min_similarity_map_all_bat.append(min_similarity_map_all_1)
                anomaly_map_all_bat.append(anomaly_maps_all1)
               
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat).float().permute(1, 0, 2, 3, 4, 5).contiguous()
            
            min_similarity_map_all_bat = torch.stack(min_similarity_map_all_bat).float()
            min_similarity_map_all_bat = F.interpolate(min_similarity_map_all_bat.view(2, -1, 32, 32), size=64, mode='bilinear', align_corners=True).view(2, 3, 4, 64, 64) 
           
            anomaly_mask = anomaly_mask.squeeze(1).float()
           
            if torch.isnan(anomaly_maps_all1).any() or torch.isinf(anomaly_maps_all1).any():
                print("Input tensor contains NaN or Inf anomaly_maps_all1")

            
            dino_features = [feature[:, 1:, :].float() for feature in dino_features]
            
            output_focl_mean = []

            for i_all in range(anomaly_map_all_bat.shape[0]):

                output, _ = model(anomaly_map_all_bat[i_all], dino_features, min_similarity_map_all_bat[:, i_all])
                
                output_focl = torch.softmax(output, dim=1)

                output_focl = F.interpolate(output_focl, size=args.dino_resolution, mode='bilinear', align_corners=True)  # Shape: [4, 2, 256, 256]
               
                output_focl_mean.append(output_focl)

            output_focl_mean = torch.stack(output_focl_mean)  # Shape: [N, 2, 2, 256, 256], where N is anomaly_map_all_bat.shape[0]

            # 对第 0 维取平均
            output_focl_mean_avg = output_focl_mean.mean(dim=0)  # Shape: [2, 2, 256, 256]


            output_focl = output_focl_mean_avg.squeeze()

            anomaly_maps1 = output_focl[:, 1, :, :].unsqueeze(1)

            out_mask_cv = (output_focl[: ,1 ,: ,:].unsqueeze(1) * object_mask).detach().cpu().numpy()
            out_mask_averaged = torch.nn.functional.avg_pool2d(output_focl[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy() 

            out_mask_averaged = out_mask_averaged * object_mask.cpu().numpy()

            anomaly_maps = (anomaly_maps1*0.5 + anomaly_maps1g12*0.5) * object_mask.to(device) # you can adjust the 0.5 to others for better results

            anomaly_maps_np = anomaly_maps.detach().cpu().numpy()

            anomaly_maps = anomaly_maps.squeeze()
            instance_label = instance_label.squeeze()
           
            score = torch.topk(torch.flatten(anomaly_maps, start_dim=1), 250)[0].mean(dim=1)
            masks.extend([m for m in anomaly_mask[:, 0, :, :].cpu().numpy()])
            preds.extend([a for a in anomaly_maps.cpu().numpy()])
            scores.extend([s for s in score.cpu().numpy()])
            labels.extend([l for l in instance_label.cpu().numpy()])


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
        vae_path = './model_Glad/Multi-class_VisA/VisA_VAE/multi-class_visa_epoch=118-step=64498.ckpt'
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




    writer = SummaryWriter(log_dir='./logs')
    if args.train:
        main(args, "")
    else:
        if 'Mvtecad' in args.instance_data_dir:
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
            torch.load("./model/MVTec_AD_dino_multi/MVTec-AD_20000step_bs32_eps_anomaly2_multiclass_0_1234/checkpoint-20000/pytorch_model.bin", map_location='cpu')
        )

        val_pipe.unet.to(device).to(dtype=torch.float16)
        val_pipe.vae = load_vae(val_pipe.vae).to(device)

        print(args)
        performances = [[], [], [], [], [], [], []]


        test_3d_dino_DRAEM_qianghua = True

        test_3d_dino_DRAEM_qianghua_all = False


        a_ = 0

        pth_files = [
            "./checkpoints_path/epoch_4_900_fscratch_channel48min.pth",
        ]


        if test_3d_dino_DRAEM_qianghua:
            start_ = 1
            log_file_path = "./test_costfilterad_glad.txt"
            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入

                for beta in range(1):
                    for model_path in pth_files:
                        print(f"Loading model weights from: {model_path}")

                        log_file.write(f"Model_qian: {model_path}\n")
                        
                        model_found = False
                        while not os.path.exists(model_path):
                            if not model_found:
                                print(f"等待模型: {model_path} 加载...")
                                model_found = True  # 只打印一次等待信息
                            time.sleep(60)  # 每隔10秒检查一次路径是否存在

                        # 当路径存在时，加载模型
                        print(f"加载模型: {model_path}")
                        model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=1024, out_channels=2, base_channels=48).to(device)
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
                            data_directory_class = f'./test_save_path_Glad_mvtecad/{class_name}'

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
                                class_name, checkpoint_step, log_file, len(val_dataset), a_, beta)

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




                        log_file.write(f"Model_hou_glad: {model_path}\n")
                        log_file.write(f"Mean Performance_glad: {np.mean(performances, axis=0)}\n\n")
                        log_file.flush()  # 确保写入文件



        
        elif test_3d_dino_DRAEM_qianghua_all:
            start_ = 1
            log_file_path = "/checkpoints_path/test_costfilterad_glad_all.txt"

            with open(log_file_path, "a") as log_file:  # 使用 "a" 模式以追加方式写入
                for epoch_ in range(1,10):
                    
                    for idx_ in range(300, 6905, 300):

                        model_path = os.path.join('/checkpoints_path', f'epoch_{epoch_}_{idx_}_fscratch_channel48min.pth')

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
                            data_directory_class = f'./test_save_path_Glad_mvtecad/{class_name}'

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
                                class_name, checkpoint_step, log_file, len(val_dataset), a_, beta=None)

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




                        log_file.write(f"Model_hou_glad: {model_path}\n")
                        log_file.write(f"Mean Performance_glad: {np.mean(performances, axis=0)}\n\n")
                        log_file.flush()  # 确保写入文件
