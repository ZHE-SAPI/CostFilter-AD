# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

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
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
# from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")


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


def train(item_list):
    setup_seed(1)

    total_iters = 10000
    batch_size = 16
    image_size = 448
    crop_size = 392

    # image_size = 448
    # crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    # test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #                         for test_data in test_data_list]

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
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)
            # loss = global_cosine(en, de)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            # loss = global_cosine(en, de)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f"model_{it+1}.pth"))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
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

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                

                best_auroc = 0

                # inside evaluation block
                if np.mean(auroc_px_list) > best_auroc:
                    best_auroc = np.mean(auroc_px_list)
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'best_model.pth'))
                    print_fn(f"[INFO] Best model saved with I-AUROC {best_auroc:.4f}")

                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return



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
    for model_path in model_paths:
        print_fn(f"Evaluating model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

        for item, test_data in zip(item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                          num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
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

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

        

        # inside evaluation block
        if np.mean(auroc_px_list) > best_auroc:
            best_auroc = np.mean(auroc_px_list)
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'best_model.pth'))
            print_fn(f"[INFO] Best model saved with P-AUROC {best_auroc:.4f}")

        

            

    
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets/BTAD1')
    parser.add_argument('--save_dir', type=str, default='./saved_results_btad')
    parser.add_argument('--save_name', type=str,
                        default='vitill_BTAD_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1')
    args = parser.parse_args()
    #
    item_list = ['01', '02', '03' ]
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    # train(item_list)
    model_paths = [
    os.path.join(args.save_dir, args.save_name, "model_9000.pth"),
    # os.path.join(args.save_dir, args.save_name, "model_2000.pth"),
    # os.path.join(args.save_dir, args.save_name, "model_2000.pth")
    ]

    test_only(item_list, model_paths=model_paths)


# cd /home/customer/Desktop/ZZ/anomaly/Dinomaly-master
# nohup python dinomaly_btad_uni_test.py > dinomaly_btad_uni_test.log 2>&1 &