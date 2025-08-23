import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset, MVTecDataset_train
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler, cal_anomaly_maps_28
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")
import heapq

from kornia.filters import gaussian_blur2d
import numpy as np
import torch.nn.functional as F
import cv2
from unet3d_att_dino_channel_test_48_min import DiscriminativeSubNetwork_3d_att_dino_channel
from loss import FocalLoss, SSIM, SoftIoULoss, FocalLoss_gamma
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list, model_paths):
    setup_seed(1)

    # total_iters = 10000
    batch_size = 12
    image_size = 448
    crop_size = 392

    # image_size = 448
    # crop_size = 448
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)


    train_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        train_data = MVTecDataset_train(root=train_path, transform=data_transform, gt_transform=gt_transform, phase="train", clsname = item, clslabel = i)
        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                               drop_last=True)



    # encoder_name = 'dinov2reg_vit_small_14'
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    # encoder_name = 'dinov2_vit_base_14'
    # encoder_name = 'dino_vit_base_16'
    # encoder_name = 'ibot_vit_base_16'
    # encoder_name = 'mae_vit_base_16'
    # encoder_name = 'beitv2_vit_base_16'
    # encoder_name = 'beit_vit_base_16'
    # encoder_name = 'digpt_vit_base_16'
    # encoder_name = 'deit_vit_base_16'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    # bottleneck.append(nn.Sequential(FeatureJitter(scale=40),
    #                                 bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)))

    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        # blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    state_dict = torch.load(model_paths[0], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # trainable = nn.ModuleList([bottleneck, decoder])

    # for m in trainable.modules():
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # optimizer = StableAdamW([{'params': trainable.parameters()}],
    #                         lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    # lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       # warmup_iters=100)


    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=768, out_channels=2, base_channels=48).to(device)

    
    model_unet = model_unet.float()
    optimizer = torch.optim.Adam([{"params": model_unet.parameters(), "lr": 0.0001}], betas=(0.9, 0.98))
    model_unet.apply(weights_init_2d)
    model_unet.train()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    loss_focal = FocalLoss_gamma()  # FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(
        log_dir='./checkpoint_paths/visaad0.0001/logs')


    print_fn('train image number:{}'.format(len(train_data_list)))


    # it = 0
    best_auroc = 0
    a_ = 0
    for epoch in range(0, 6):
        model.eval()
        model_unet.train()

        loss_list = []
        # for img, anomaly_mask, isnormal, clsname, class_labels in train_dataloader:
        for img, anomaly_mask, isnormal, clsname, class_labels in tqdm(train_dataloader):

            a_ += 1
            img = img.to(device)
            class_labels = class_labels.to(device)
            anomaly_mask = anomaly_mask.to(device)

            with torch.no_grad():
                en, de = model(img)
                # loss = global_cosine(en, de)
                # for i, (e, d) in enumerate(zip(en, de)):
                #     print(f"[Layer {i}] en[{i}].shape: {e.shape}, de[{i}].shape: {d.shape}")
                # [Layer 0] en[0].shape: torch.Size([12, 768, 28, 28]), de[0].shape: torch.Size([12, 768, 28, 28])
                # [Layer 1] en[1].shape: torch.Size([12, 768, 28, 28]), de[1].shape: torch.Size([12, 768, 28, 28])
                batch_size = en[0].shape[0]
                min_anomaly_map = []  # 保存每一层的 anomaly_map1
                anomaly_map_all_bat = []
                epsilon = 1e-8

                for rec_feat, org_feat in zip(en, de):
                    # 计算每层特征的 cos0

                    H, W = org_feat.shape[2:]
                    assert H * W == 28*28, "Feature map size must be 28*28"

                    pi = org_feat.reshape(batch_size, -1, H * W).permute(0, 2, 1)  # [batch_size, 196, C]
                    pr = rec_feat.reshape(batch_size, -1, H * W).permute(0, 2, 1)

                    # 归一化
                    pi = pi / (torch.norm(pi, p=2, dim=-1, keepdim=True) + epsilon)
                    pr = pr / (torch.norm(pr, p=2, dim=-1, keepdim=True) + epsilon)


                    # 计算余弦相似度矩阵
                    cos0 = torch.bmm(pi, pr.permute(0, 2, 1))  # [batch_size, H*W, H*W]

                    # 计算每层的 anomaly_map1
                    anomaly_map, _ = torch.min(1 - cos0, dim=-1)  # [batch_size, H*W]
                    anomaly_map = nn.UpsamplingBilinear2d(scale_factor=64/28)(
                        anomaly_map.reshape(batch_size, 1, H, W)
                    )
                    min_anomaly_map.append(anomaly_map)



                    pre_min_dim = 768 # 48
                    one_minus_cos0 = 1 - cos0 # [batch_size, H*W, H*W]





                    _, indices = torch.topk(one_minus_cos0, pre_min_dim, dim=-1, largest=False, sorted=False)
                
                    selected_values = torch.gather(one_minus_cos0, dim=-1, index=torch.sort(indices, dim=-1)[0])
                    anomal_map_3d = selected_values.view(batch_size, 28, 28, pre_min_dim).unsqueeze(1)
                    anomaly_map_all_bat.append(anomal_map_3d)


                # 综合四层的 anomaly_map1
                min_anomaly_map = torch.stack(min_anomaly_map, dim=0).permute(1, 0, 2, 3, 4).squeeze(0) 
                anomaly_map_all_bat = torch.stack(anomaly_map_all_bat, dim=0).squeeze(2) # [4, batch_size, 28, 28, 196]
                anomaly_map_all_bat = nn.UpsamplingBilinear2d(scale_factor=64/28)(anomaly_map_all_bat.permute(0, 1, 4, 2, 3).view(-1, pre_min_dim, H, W))
                anomaly_map_all_bat = anomaly_map_all_bat.view(2, batch_size, pre_min_dim, 64, 64).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)


                pred1, _ = cal_anomaly_maps_28(en, de, img.shape[-1])

                for i_ in range(img.shape[0]):
                    sample = nn.UpsamplingBilinear2d(size=(392, 392))(pred1)[i_].detach().cpu().numpy().squeeze() * 255
                    cv2.imwrite(
                        f"./checkpoint_paths/visaad0.0001/logs/train/pred_dinomaly_{i_}_{a_}.png",
                        sample)


                    sample = reverse_normalization(img[i_])

                    sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(f"./checkpoint_paths/visaad0.0001/logs/train/images_dinomaly_{i_}_{a_}.png", sample)


                
                dino_features = [feat.view(feat.shape[0], feat.shape[1], 784).permute(0,2,1) for feat in en] # [batch_size, 196, C]  [24, 32, 56, 160]

                min_similarity_map_all_bat = torch.zeros_like(min_anomaly_map.squeeze())
                min_similarity_map_all_bat[:, 0, :, :] = nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1)
                min_similarity_map_all_bat[:, 1, :, :] = nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1)




            model_unet.train()
            optimizer.zero_grad()
            output, cls_out = model_unet(anomaly_map_all_bat.float().to(device), dino_features,
                                    min_similarity_map_all_bat.float().to(device))

            classification_loss = criterion(cls_out, class_labels.long())
           
            output_focl = torch.softmax(output, dim=1)
            output_focl = F.interpolate(output_focl, size=392, mode='bilinear',
                                        align_corners=True)  # Shape: [4, 2, 256, 256]

            anomaly_prob = output_focl[:, 1, :, :].unsqueeze(1)

            # Calculate dynamic gamma based on classification output
            gamma = compute_dynamic_gamma(cls_out, clsname, class_labels.float(), 392, num_classes=15).to(
                device)


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
                print(f"Skipping batch {a_} due to invalid loss: {loss}")
                continue  

            loss.backward()

        
            optimizer.step()

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

            torch.nn.utils.clip_grad_norm_(model_unet.parameters(), 2.0)


        scheduler.step(loss)

        checkpoint_path = os.path.join('./checkpoint_paths/visaad0.0001', f'epoch_{epoch}_costfilter_dinaomaly_my_visaad.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_unet.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
        }, checkpoint_path)
        print(f'Saved checkpoint for epoch {epoch} at {checkpoint_path}')


def test_only(item_list, model_paths):
    setup_seed(1)

    batch_size = 12
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)


    test_data_list = []
    for item in item_list:
        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        test_data_list.append(test_data)

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12  # for base

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    # bottleneck.append(nn.Sequential(FeatureJitter(scale=40),
    #                                 bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)))

    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        # blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)


    # trainable = nn.ModuleList([bottleneck, decoder])

    # for m in trainable.modules():
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # optimizer = StableAdamW([{'params': trainable.parameters()}],
    #                         lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    # lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
    #                                    warmup_iters=100)

    # print_fn('train image number:{}'.format(len(train_data)))
    best_auroc = 0

    model_unet = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=768, out_channels=2, base_channels=48).to(device)

    
    model_unet = model_unet.float()
    model_unet.eval()

    log_file_path = "./checkpoint_paths/visaad0.0001/log_dinomaly_my_visa.txt"

    with open(log_file_path, "a") as log_file:
        state_dict = torch.load(model_paths[0], map_location=device)
        model.load_state_dict(state_dict)

        for epoch_ in range(1): 
            model_path_unet = os.path.join('./checkpoint_paths/visaad0.0001/costfilter_dinaomaly_visa.pth')
            import time


            log_file.write(f"Model: {model_path_unet}\n")
            
            model_found = False
            while not os.path.exists(model_path_unet):
                if not model_found:
                    print(f"等待模型: {model_path_unet} 加载...")
                    model_found = True  # 只打印一次等待信息
                time.sleep(60)  # 每隔10秒检查一次路径是否存在

            print(f"加载模型: {model_path_unet}")
            checkpoint = torch.load(model_path_unet, map_location=device)
            model_unet.load_state_dict(checkpoint['model_state_dict']) 
            model_unet.eval()  


            print_fn(f"Evaluating model: {model_path_unet}")
            
            model.eval()

            auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
            auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
            start_time = time.time()

            for item, test_data in zip(item_list, test_data_list):
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                              num_workers=4)
                results = evaluation_batch(model, model_unet, test_dataloader, device, max_ratio=0.01, resize_mask=392, lamda = 0.1)
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                auroc_sp_list.append(auroc_sp)
                ap_sp_list.append(ap_sp)
                f1_sp_list.append(f1_sp)
                auroc_px_list.append(auroc_px)
                ap_px_list.append(ap_px)
                f1_px_list.append(f1_px)
                aupro_px_list.append(aupro_px)

                print_fn(
                    '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                log_file.write('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}\n'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px) + '\n')
                log_file.flush()

            total_train_time = time.time() - start_time
            print(f"Total testing time: {total_train_time:.2f} seconds")
            log_file.write(f"Total testing time: {total_train_time:.2f} seconds\n")

            print_fn(
                'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                    np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

            log_file.write('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}\n'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)) + '\n')
            log_file.flush()
            


                
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

    difficult_classes = ['macaroni1', 'macaroni2', 'candle', 'capsules', 'cashew', 'fryum']


    batch_size = cls_out.shape[0]
    # Find the predicted class for each sample
    predicted_classes = cls_out.max(dim=1)[1]  # shape: [batch_size]

    # Check if the prediction is correct
    correct_prediction = (predicted_classes == target)  # shape: [batch_size]
    # Calculate gamma based on classification confidence
    # Decrease gamma as classification confidence increases

    # cls_out = torch.softmax(cls_out, dim=1)

    gamma = 3.0 - cls_out.max(dim=1).values  # Max prob from cls_out to determine focus level
    # gamma = gamma.unsqueeze(1).expand(-1, num_classes)  # Expand gamma to the shape [batch_size, num_classes]
    gamma = torch.clamp(gamma, min=1.5)

    # Set gamma to 4 for the difficult classes
    for i in range(batch_size):
        # predicted_class_idx = predicted_classes[i].item()  # Get the predicted class index for the i-th sample
        # predicted_class_name = list(clsname_to_index.keys())[predicted_class_idx]  # Get the class name

        # If the predicted class is one of the difficult ones, set gamma to 4
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

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)

    # 如果 normalized_image 是 numpy 数组，先转换为 PyTorch 张量
    if isinstance(normalized_image, np.ndarray):
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32)

    # 反向归一化
    original_image = normalized_image * std[None, :, None, None] + mean[None, :, None, None]
    
    return original_image
    



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/sysmanager/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets/VisA_pytorch') # replace to the path of your VisA_pytorch dataset 
    parser.add_argument('--save_dir', type=str, default='./saved_results_visa')
    parser.add_argument('--save_name', type=str,
                        default='vitill_visa_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1')
    parser.add_argument('--train', action='store_true', help='train setting')
    args = parser.parse_args()
    #

    
    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
                             
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    model_paths = [
    os.path.join(args.save_dir, args.save_name, "model_9000.pth"),

    ]

    if args.train:
        train(item_list, model_paths=model_paths)
    else:
        

        test_only(item_list, model_paths=model_paths)


# cd /CostFilterAD/Costfilter_Dinomaly

# nohup python costfilter_dinomaly_visa_uni_my_train_test.py --train > costfilter_dinomaly_visa_uni_my_train.log 2>&1 &


# nohup python costfilter_dinomaly_visa_uni_my_train_test.py > costfilter_dinomaly_visa_uni_my_test.log 2>&1 &
