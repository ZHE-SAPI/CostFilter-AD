import argparse
import logging
import os
import pprint
import shutil
import time
import pandas as pd

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
# from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
# from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.eval_helper1 import dump, log_metrics, merge_together, performances

from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    load_state_visa,
    load_state_visa_0,
    save_checkpoint,
    set_random_seed,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single
# from models.HVQ_TR_switch import HVQ_TR_switch
from models.HVQ_TR_switch_OT import HVQ_TR_switch_OT
import os
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms")

import heapq

from kornia.filters import gaussian_blur2d
import numpy as np
import torch.nn.functional as F
import cv2
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from loss import FocalLoss, SSIM, SoftIoULoss, FocalLoss_gamma
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description="UniAD Framework")

parser.add_argument("--config", default="./config_mpdd_withOT.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument('--train_only_four_decoder',default=False,type=bool)

def main():
    

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))



    rank = 0
    world_size = 1
    
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    print('config.save_path', config.save_path)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = HVQ_TR_switch_OT(channel=272, embed_dim=256)
    # C

    model.to(device)

    
    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))



    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=196, out_channels=2, base_channels=48).to(device)

    
    model_unet = model_unet.float()
    optimizer = torch.optim.Adam([{"params": model_unet.parameters(), "lr": 0.001}], betas=(0.9, 0.98))
    model_unet.apply(weights_init_2d)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    resume_model_HVQ_TR_switch = config.saver.get("resume_model", None)
    print('resume_model_HVQ_TR_switch', resume_model_HVQ_TR_switch)

    print('HVQ_TR_switch')

    if resume_model_HVQ_TR_switch:
        _, _ = load_state_visa_0(resume_model_HVQ_TR_switch, model, optimizer=optimizer)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=False)


    loss_focal = FocalLoss_gamma()  # FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(
        log_dir='./log')


    if args.evaluate:
        loaded_state_dict = torch.load(resume_model_HVQ_TR_switch)["state_dict"]



        # 定义保存输出的文件
        output_file = "./model_comparison.txt"

        # 将输出重定向到文件
        with open(output_file, "w") as f:
            sys.stdout = f  # 重定向标准输出

            # 打印模型结构
            print("Model structure:")
            for name, param in model.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")

            # 打印参数文件中的键
            if loaded_state_dict:
                print("\nModel parameters in file:")
                for key in loaded_state_dict.keys():
                    print(f"Param name in file: {key}")

            # 比较模型结构和参数文件
            if loaded_state_dict:
                print("\nComparison of model layers and parameter file keys:")
                model_keys = set([name for name, _ in model.named_parameters()])
                param_file_keys = set(loaded_state_dict.keys())

                missing_in_model = param_file_keys - model_keys
                missing_in_file = model_keys - param_file_keys
                matching_keys = model_keys & param_file_keys

                print(f"Matching keys ({len(matching_keys)}): {sorted(list(matching_keys))}")
                print(f"Keys missing in model ({len(missing_in_model)}): {sorted(list(missing_in_model))}")
                print(f"Keys missing in parameter file ({len(missing_in_file)}): {sorted(list(missing_in_file))}")

            # 恢复标准输出
            sys.stdout = sys.__stdout__

        print(f"Output saved to {output_file}")




        # model.load_state_dict(loaded_state_dict)
        # validate(train_loader, model, model_unet, device)
        validate(val_loader, model, model_unet, device)
        return

    # criterion = build_criterion(config.criterion)
    a_ = 0
    for epoch in range(last_epoch, config.trainer.max_epoch): # 36.pth is the best
    
        last_iter = epoch * len(train_loader)
        a_, scheduler = train_one_epoch(
            train_loader,
            model,
            model_unet,
            optimizer,
            scheduler,
            epoch,
            last_iter,
            tb_logger,
            frozen_layers,
            device,
            a_,
            loss_focal,
            loss_l2,
            loss_ssim,
            criterion,
            writer,

        )

        checkpoint_path = os.path.join('./checkpoint_paths', f'epoch_{epoch}_hvq_my_mpdd.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
        }, checkpoint_path)
        print(f'Saved checkpoint for epoch {epoch} at {checkpoint_path}')


def train_one_epoch(
    train_loader,
    model,
    model_unet,
    optimizer,
    scheduler,
    epoch,
    start_iter,
    tb_logger,
    frozen_layers,
    device,
    a_,
    loss_focal,
    loss_l2,
    loss_ssim,
    criterion,
    writer,

    ):
    model.training = True


    model.eval()
    model_unet.train()

    world_size = 1
    rank = 0



    for idx_, input in enumerate(tqdm(train_loader)):
        a_ += 1
        lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            anomaly_map_all_bat, min_similarity_map_all_bat, dino_features, _, _, _ = model(input, device, out_pred_ =False)

        
        # output, cls_out = model(anomaly_map_all_bat[i_all], dino_features)
        model_unet.train()
        optimizer.zero_grad()
        output, cls_out = model_unet(anomaly_map_all_bat.float().to(device), dino_features,
                                min_similarity_map_all_bat.float().to(device))
        
        class_labels = input["clslabel"].to(device)
        anomaly_mask = input["mask"].to(device)
        classification_loss = criterion(cls_out, class_labels.long())
        output_focl = torch.softmax(output, dim=1)

        output_focl = F.interpolate(output_focl, size=224, mode='bilinear',
                                    align_corners=True)  # Shape: [4, 2, 256, 256]

        anomaly_prob = output_focl[:, 1, :, :].unsqueeze(1)

        # Calculate dynamic gamma based on classification output
        gamma = compute_dynamic_gamma(cls_out, input["clsname"], class_labels.float(), 224, num_classes=15).to(device)


        loss_images = []
        for i in range(output_focl.shape[0]):
            loss_i = loss_focal(output_focl.float()[i:i + 1], anomaly_mask.float()[i:i + 1],
                                gamma[i])  # For each image, slice the batch
            loss_images.append(loss_i)

        # Average of losses for each image
        Focal_loss = sum(loss_images) / len(loss_images)

        ssim_loss = loss_ssim(anomaly_prob.float(), anomaly_mask.float())

        l2_loss = loss_l2(anomaly_prob.float(), anomaly_mask.float())

        SIOU_loss = SoftIoULoss(anomaly_prob.float(), anomaly_mask.float())

        alpha, beta, gamma, yita = 1.0, 0.1, 1.0, 0.1
        segmentation_loss = alpha * Focal_loss + beta * ssim_loss + gamma * l2_loss + yita * SIOU_loss

        loss = segmentation_loss + 0.5 * classification_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Skipping batch {idx_} due to invalid loss: {loss}")
            print('filename', input["filename"])
            continue  # 跳过当前 batch

        loss.backward()

    
        optimizer.step()  # 执行优化步骤

        # 准确率计算
        predicted_classes = torch.argmax(cls_out, dim=1)
        correct_predictions = (predicted_classes == class_labels).sum().item()
        total_samples = class_labels.size(0)
        train_accuracy = correct_predictions / total_samples

        current_losses = {'Focal_loss': alpha * Focal_loss, 'sum_anomaly_prob': torch.sum(anomaly_prob), 'sum_anomaly_mask': torch.sum(anomaly_mask),
                          'ssim_loss': beta * ssim_loss, 'l2_loss': gamma * l2_loss, 'SIOU_loss': yita * SIOU_loss,
                          'cls_loss': 0.5 * classification_loss, 'seg_loss': segmentation_loss,
                          'train_accuracy': train_accuracy, }  # 0.5 * ce_loss
        log_losses_tensorboard(writer, current_losses, a_)

        # backward
       
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model_unet.parameters(), 2.0)

    scheduler.step(loss)    

        

        
    return a_, scheduler

def validate(val_loader, model, model_unet, device):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    model.training = False # never delete, useful


    log_file_path = "./checkpoint_paths/log_costfilter_hvq_my_mpdd.txt"

    with open(log_file_path, "a") as log_file:
        
        for epoch_ in range(1): 
            # model_path = os.path.join('./checkpoint_paths', f'epoch_{epoch_}_hvq_my_mpdd.pth')
            model_path = os.path.join('./checkpoint_paths/costfilter_hvqtrans_mpdd.pth')
            # 遍历所有模型路径
            log_file.write(f"Model: {model_path}\n")
            
            # 持续检测模型路径是否存在
            model_found = False
            while not os.path.exists(model_path):
                if not model_found:
                    print(f"等待模型: {model_path} 加载...")
                    model_found = True  # 只打印一次等待信息
                time.sleep(60)  # 每隔10秒检查一次路径是否存在

            # 当路径存在时，加载模型
            print(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model_unet.load_state_dict(checkpoint['model_state_dict']) 
            model_unet = model_unet.float()
            model_unet.eval()  


            rank = 0
            world_size = 1

            # rank = dist.get_rank()
            logger = logging.getLogger("global_logger")
            # criterion = build_criterion(config.criterion)
            end = time.time()

            if rank == 0:
                config.evaluator.eval_dir_test = f"{config.evaluator.eval_dir}-epoch{epoch_}"
                os.makedirs(config.evaluator.eval_dir_test, exist_ok=True)
            
            pred_max_heap = []  # 保存 pred_max 的前 50 大值（使用最小堆）
            pred_min_heap = []  # 保存 pred_min 的前 50 小值（使用最大堆，取负值存入）
            with torch.no_grad():
                i_tem = 0
                for idx_, input in enumerate(tqdm(val_loader, desc="Validation Progress")):
                    
                    with torch.no_grad():
                        anomaly_map_all_bat, min_similarity_map_all_bat, dino_features, pred, outputs, pred_ceshi = model(input, device, out_pred_ =True)
                        output, _ = model_unet(anomaly_map_all_bat.float().to(device), dino_features,
                                        min_similarity_map_all_bat.float().to(device))
                        output_focl = torch.softmax(output, dim=1)
                        output_focl = F.interpolate(output_focl, size=224, mode='bilinear', align_corners=True)
                        anomaly_maps1 = output_focl[:, 1, :, :].unsqueeze(1)
                        
                        anomaly_maps = (anomaly_maps1 + pred*0.00) * input["object_mask"].unsqueeze(0).to(device) # don't need to integrate with the baseline response for best results
                        
                        outputs["pred"] = anomaly_maps


                    dump(config.evaluator.eval_dir_test, outputs)
                   
                    pred_max = pred_ceshi.max().item()
                    pred_min = pred_ceshi.min().item()

                    i_tem += 1
                    sigma = 6
                    kernel_size = 2 * int(4 * sigma + 0.5) + 1


                    batch_time.update(time.time() - end)
                    end = time.time()

                    if (idx_ + 1) % 100 == 0 and rank == 0:
                        logger.info(
                            "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                                idx_ + 1, len(val_loader), batch_time=batch_time
                            )
                        )


            ret_metrics = {}  # only ret_metrics on rank0 is not empty
            if rank == 0:
                logger.info("Gathering final results ...")
                # total loss
                # logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num))
                fileinfos, preds, masks, pred_imgs = merge_together(config.evaluator.eval_dir_test)
                shutil.rmtree(config.evaluator.eval_dir_test)
                # evaluate, log & vis
                ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
                log_metrics(ret_metrics, config.evaluator.metrics)




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




def compute_dynamic_gamma(cls_out, clsname, target, dino_resolution, num_classes=15):
    """
    Compute gamma dynamically for each image based on classification output (cls_out).
    The idea is to increase gamma (focus on hard examples) when classification confidence is low,
    and decrease gamma (focus on easy examples) when confidence is high.

    Additionally, if the classification is wrong, increase gamma to focus more on hard examples.
    """
    

    difficult_classes = ['tubes','bracket_brown']


    batch_size = cls_out.shape[0]

    predicted_classes = cls_out.max(dim=1)[1]  # shape: [batch_size]

    # Check if the prediction is correct
    correct_prediction = (predicted_classes == target)  # shape: [batch_size]
    
    gamma = 3.0 - cls_out.max(dim=1).values  # Max prob from cls_out to determine focus level
    gamma = torch.clamp(gamma, min=1.5)

    for i in range(batch_size):
        
        if clsname in difficult_classes:
            gamma[i] = 3.5  # Set all gamma values for this sample to 4 (you can customize it per class if needed)

    # If the prediction is incorrect, increase gamma for that image
    # For simplicity, let's set gamma to a higher value if the prediction is wrong
    gamma[~correct_prediction] = 3.5  # Increase gamma for misclassified images (e.g., 4.0)

    return gamma  # Shape [batch_size]




def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()



def reverse_normalization(normalized_image):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to(normalized_image.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to(normalized_image.device)

    # 如果 normalized_image 是 numpy 数组，先转换为 PyTorch 张量
    if isinstance(normalized_image, np.ndarray):
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32)

    # 反向归一化
    original_image = normalized_image * std[None, :, None, None] + mean[None, :, None, None]

    return original_image


if __name__ == "__main__":
    rank = 0
    world_size = 1

    main()
