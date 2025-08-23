import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math

import pickle
from tqdm import tqdm
import sys
import torch.nn as nn
import heapq

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    # loss = loss / len(a)
    return loss


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss



def normalize_max_min(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min()+ 1e-8)
    else:
        return (pred - min_value) / (max_value - min_value + 1e-8)

def apply_ad_scoremap(image, scoremap01, alpha=0.3, tau=0.06, blur_ks=5):
    img = np.asarray(image, dtype=np.uint8)          # RGB uint8
    s   = np.asarray(scoremap01, dtype=np.float32)   # float [0,1]

    if blur_ks and blur_ks > 0:
        s = cv2.GaussianBlur(s, (blur_ks, blur_ks), 0)

    if tau > 0:
        s = np.where(s >= tau, s, 0.0)

    s_u8 = (np.clip(s, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(s_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    fused = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    return fused.astype(np.uint8), heat.astype(np.uint8)

    
# def apply_ad_scoremap(image, scoremap, alpha=0.7):
#     np_image = np.asarray(image, dtype=np.float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8), scoremap.astype(np.uint8)


def global_cosine_hm_percent_pred1(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()  # 指定 dim=1 是通道维
    loss_maps = []  # 用于保存每一层的 [B, H, W] loss map

    for item in range(len(a)):
        a_ = a[item].detach()  # [B, C, H, W]
        b_ = b[item]           # [B, C, H, W]

        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)  # [B, 1, H, W]

        # # 计算 top-p 的阈值
        # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        # # 注册 hook（低于阈值的位置抑制梯度）
        # partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        # b_.register_hook(partial_func)

        loss_maps.append(point_dist)  # shape [B, H, W]

    mean_loss_map = torch.stack(loss_maps, dim=0).mean(dim=0)
    return mean_loss_map  # shape [B, 1, H, W]




# def global_cosine_hm_percent_pred1(a, b, p=0.9, factor=0.):
#     cos_loss = torch.nn.CosineSimilarity()
#     loss = 0
#     for item in range(len(a)):
#         a_ = a[item].detach()
#         b_ = b[item]
#         with torch.no_grad():
#             point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
#         # mean_dist = point_dist.mean()
#         # std_dist = point_dist.reshape(-1).std()
#         thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

#         loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
#                                         b_.reshape(b_.shape[0], -1)))

#         partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
#         b_.register_hook(partial_func)

#     loss = loss / len(a)
#     return loss


def regional_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += point_dist.mean()

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        focal_factor = focal_factor ** alpha
        focal_factor = torch.clip(focal_factor, min=min_grad)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)

    return loss


def regional_cosine_focal(a, b, p=0.9, alpha=2.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh
        focal_factor = focal_factor ** alpha

        loss += (point_dist * focal_factor.detach()).mean()

    return loss


def regional_cosine_hm(a, b, p=0.9):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        L = point_dist[point_dist >= thresh]
        loss += L.mean()

    return loss


def region_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list



def cal_anomaly_maps_28(fs_list, ft_list, out_size=28):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        # a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    start_list = []
    end_list = []
    with torch.no_grad():
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)

    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, norm_factor=None, feature_used='all',
               max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            if feature_used == 'trained':
                anomaly_map, _ = cal_anomaly_map(en[3:], de[3:], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            elif feature_used == 'freezed':
                anomaly_map, _ = cal_anomaly_map(en[:3], de[:3], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            else:
                anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
                sp_score = sp_score.mean()
            pr_list_sp.append(sp_score)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_batch(model, model_unet, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, lamda = 0.5):

    a__ = 0
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # pred_max_heap = []  # 保存 pred_max 的前 50 大值（使用最小堆）
    # pred_min_heap = []  # 保存 pred_min 的前 50 小值（使用最大堆，取负值存入）
    with torch.no_grad():
        # for img, gt, label, img_path in dataloader:
        for img, gt, label, img_path in tqdm(dataloader):
            a__ += 1
            img = img.to(device)
            # starter.record()
            print('img.shape', img.shape)
            print('gt.shape', gt.shape)

            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de = output[0], output[1]
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
                # print("anomaly_map.shape", anomaly_map.shape) # [16, 1, 64, 64]
                min_anomaly_map.append(anomaly_map)



                pre_min_dim = 768 # 48
                one_minus_cos0 = 1 - cos0 # [batch_size, H*W, H*W]





                print('one_minus_cos0.shape', one_minus_cos0.shape) # [1, 1024, 1024]
                _, indices = torch.topk(one_minus_cos0, pre_min_dim, dim=-1, largest=False, sorted=False)
                print('indices.shape', indices.shape) # [batch_size, H*W, pre_min_dim]
                # print('indices', indices)
                # print('torch.sort(indices, dim=-1)[0]', torch.sort(indices, dim=-1)[0]) # [batch_size, H*W, pre_min_dim]
                selected_values = torch.gather(one_minus_cos0, dim=-1, index=torch.sort(indices, dim=-1)[0])
                # print('selected_values.shape', selected_values.shape) # [batch_size, 196, 196]
                anomal_map_3d = selected_values.view(batch_size, 28, 28, pre_min_dim).unsqueeze(1)
                anomaly_map_all_bat.append(anomal_map_3d)


            # 综合四层的 anomaly_map1
            min_anomaly_map = torch.stack(min_anomaly_map, dim=0).permute(1, 0, 2, 3, 4).squeeze(0) 
            print("min_anomaly_map.shape", min_anomaly_map.shape) # [4, batch_size, 32, 32]     batch_size,4,1,64,64
            anomaly_map_all_bat = torch.stack(anomaly_map_all_bat, dim=0).squeeze(2) # [4, batch_size, 28, 28, 196]
            anomaly_map_all_bat = nn.UpsamplingBilinear2d(scale_factor=64/28)(anomaly_map_all_bat.permute(0, 1, 4, 2, 3).view(-1, pre_min_dim, H, W))
            anomaly_map_all_bat = anomaly_map_all_bat.view(2, batch_size, pre_min_dim, 64, 64).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)
            print("anomaly_map_all_bat.shape", anomaly_map_all_bat.shape) # [batch_size, 768, 2, 64, 64]


            # p = 0.9
                
            pred1, _ = cal_anomaly_maps(en, de, img.shape[-1])

            print("pred1.shape", pred1.shape)  # 12, 1, 392, 392
            
            dino_features = [feat.view(feat.shape[0], feat.shape[1], 784).permute(0,2,1) for feat in en] # [batch_size, 196, C]  [24, 32, 56, 160]

            print("torch.mean(min_anomaly_map, dim=1).shape", torch.mean(min_anomaly_map, dim=1).shape) # 16, 1, 64, 64]
            print("nn.UpsamplingBilinear2d(scale_factor=64/28)(pred1).shape", nn.UpsamplingBilinear2d(size=(64, 64))(pred1).shape) # 16,1,64,64

            print("nn.UpsamplingBilinear2d.shape", (torch.mean(min_anomaly_map, dim=1) + 0.1 * nn.UpsamplingBilinear2d(size=(64, 64))(pred1)).squeeze(1).shape) # [16, 64, 64]
            

            print("min_anomaly_map.shape", min_anomaly_map.shape)
            # print("min_anomaly_map.dim()", min_anomaly_map.dim())
            min_similarity_map_all_bat = torch.zeros_like(min_anomaly_map.squeeze())

            if min_anomaly_map.dim() == 4:
                min_similarity_map_all_bat = min_similarity_map_all_bat.unsqueeze(0)  # 加上 batch 维
           

            print("nn.UpsamplingBilinear2d(size=(64, 64))(pred1).shape", nn.UpsamplingBilinear2d(size=(64, 64))(pred1).shape)
            print("nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1).shape", nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1).shape)
            print("min_similarity_map_all_bat.shape", min_similarity_map_all_bat.shape)
            print("min_similarity_map_all_bat[:, 0, :, :].shape", min_similarity_map_all_bat[:, 0, :, :].shape)

            min_similarity_map_all_bat[:, 0, :, :] = nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1)
            min_similarity_map_all_bat[:, 1, :, :] = nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1)

            # out_pred_= True
            # if out_pred_==True:
            #     min_similarity_map_all_bat = min_anomaly_map.squeeze() # batch_size,4,64,64
            #     print("min_similarity_map_all_bat[:, 0, :, :].shape", min_similarity_map_all_bat[:, 0, :, :].shape) # [16, 64, 64]
            #     min_similarity_map_all_bat[:, 0, :, :] = ((torch.mean(min_anomaly_map, dim=1) + 0.1 * nn.UpsamplingBilinear2d(size=(64, 64))(pred1))/70).squeeze(1)
            # else:
            #     min_similarity_map_all_bat = min_anomaly_map.permute(1,0,2,3) # [1, 4, 32, 32]
            #     # print("min_similarity_map_all_bat[:, 0, :, :].shape", min_similarity_map_all_bat[:, 0, :, :].shape) # [16, 64, 64]
            #     # print("torch.mean(min_anomaly_map, dim=0).unsqueeze(1).shape", torch.mean(min_anomaly_map, dim=0).shape) # [1, 64, 64]
            #     min_similarity_map_all_bat[:, 0, :, :] = (torch.mean(min_anomaly_map, dim=0) + 0.1 * nn.UpsamplingBilinear2d(size=(64, 64))(pred1).squeeze(1))/40
            # print("Max value min_similarity_map_all_bat[:, 0, :, :]:", min_similarity_map_all_bat[:, 0, :, :].max().item())  
            # print("Min value: min_similarity_map_all_bat[:, 0, :, :]", min_similarity_map_all_bat[:, 0, :, :].min().item())  
            pred_ori = pred1 # [batch_size, 1, 224, 224]
                        
                        
            pred_ceshi = pred1.squeeze(1)


            output, _ = model_unet(anomaly_map_all_bat.float().to(device), dino_features,
                                        min_similarity_map_all_bat.float().to(device))
            output_focl = torch.softmax(output, dim=1)
            output_focl = F.interpolate(output_focl, size=392, mode='bilinear', align_corners=True)
            anomaly_maps1 = output_focl[:, 1, :, :].unsqueeze(1)
            print('anomaly_maps1.shape', anomaly_maps1.shape) # [12, 1, 392, 392]
            anomaly_maps = (anomaly_maps1*lamda + pred_ori*(1-lamda))
            print('anomaly_maps.shape', anomaly_maps.shape) #  [1, 1, 392, 392]
            


            
            # print('pred_ori.max().item()', pred_ori.max().item()) 
            # print('pred_ori.min().item()', pred_ori.min().item()) 

            # print('anomaly_maps1.max().item()', anomaly_maps1.max().item()) 
            # print('anomaly_maps1.min().item()', anomaly_maps1.min().item()) 



            pred_max = pred_ceshi.max().item()
            pred_min = pred_ceshi.min().item()


            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1

            # # 更新 pred_max_heap
            # if len(pred_max_heap) < 150:
            #     heapq.heappush(pred_max_heap, pred_max)  # 如果堆中少于 50 个值，直接加入
            # else:
            #     heapq.heappushpop(pred_max_heap, pred_max)  # 加入新值并移除堆中最小值

            # # 更新 pred_min_heap
            # if len(pred_min_heap) < 150:
            #     heapq.heappush(pred_min_heap, -pred_min)  # 存入负值，模拟最大堆
            # else:
            #     heapq.heappushpop(pred_min_heap, -pred_min)  # 加入新值并移除堆中最大值










            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

            if resize_mask is not None:
                anomaly_maps = F.interpolate(anomaly_maps, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            print('resize_mask', resize_mask)

            # print('anomaly_maps1.max().item()', anomaly_maps1.max().item()) 
            # print('anomaly_maps1.min().item()', anomaly_maps1.min().item()) 




            # for i_ in range(1): # anomaly_maps.shape[0]
                
            #     print('anomaly_maps[i_].shape', anomaly_maps[i_].shape)

            #     sample_pred = np.squeeze(anomaly_maps[i_].detach().cpu().numpy())
            #     # sample_pred = np.squeeze(sample_pred * 255)
            #     print('sample_pred.shape0', sample_pred.shape)
            #     cv2.imwrite(f"./test_mvtec/pred_{i_}_{a__}.png",(normalize_max_min(sample_pred)*255).astype(np.uint8))


            #     sample = gt[i_].detach().cpu().numpy().squeeze() * 255
            #     print('sample.shape', sample.shape)
            #     cv2.imwrite(f"./test_mvtec/mask_{i_}_{a__}.png",sample)


            #     sample = reverse_normalization(img[i_])
            #     sample = sample.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
            #     print('sample.shape1', sample.shape)
            #     cv2.imwrite(f"./test_mvtec/gt_images_test_{i_}_{a__}.png", sample)


            #     heatmap_fused, heatmap = apply_ad_scoremap(sample, normalize_max_min(sample_pred))
            #     cv2.imwrite(f"./test_mvtec/heatmap_test_{i_}_{a__}.png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            #     cv2.imwrite(f"./test_mvtec/heatmap_fused_test_{i_}_{a__}.png", cv2.cvtColor(heatmap_fused, cv2.COLOR_RGB2BGR))
            for i_ in range(1):  # anomaly_maps.shape[0]
                print('anomaly_maps[i_].shape', anomaly_maps[i_].shape)

                # 1) anomaly map：先在 float 上归一化（传给apply用），另存图时再量化
                score = np.squeeze(anomaly_maps[i_].detach().cpu().numpy()).astype(np.float32)  # float
                score01 = normalize_max_min(score)  # float in [0,1]

                # 保存pred（可视用）
                pred_u8 = (score01 * 255.0).astype(np.uint8)
                cv2.imwrite(f"./test_mvtec/pred_{i_}_{a__}.png", pred_u8)

                # 2) gt mask：保存时转uint8
                mask_u8 = (gt[i_].detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
                print('mask_u8.shape', mask_u8.shape)
                cv2.imwrite(f"./test_mvtec/mask_{i_}_{a__}.png", mask_u8)

                # 3) 原图：clip并转uint8（避免融合时类型/范围隐性问题）
                sample = reverse_normalization(img[i_]).cpu().numpy().transpose(0, 2, 3, 1)[0] * 255.0
                sample = np.clip(sample, 0, 255).astype(np.uint8)
                print('sample.shape1', sample.shape)
                cv2.imwrite(f"./test_mvtec/gt_images_test_{i_}_{a__}.png", sample)

                # 4) 融合：传float[0,1]的score
                heatmap_fused, heatmap = apply_ad_scoremap(sample, score01)
                cv2.imwrite(f"./test_mvtec/heatmap_test_{i_}_{a__}.png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"./test_mvtec/heatmap_fused_test_{i_}_{a__}.png", cv2.cvtColor(heatmap_fused, cv2.COLOR_RGB2BGR))



            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_maps)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_maps.flatten(1), dim=1)[0]
            else:
                anomaly_maps = anomaly_maps.flatten(1)
                sp_score = torch.sort(anomaly_maps, dim=1, descending=True)[0][:, :max(int(anomaly_maps.shape[1] * max_ratio), 250)] # int(anomaly_maps.shape[1] * max_ratio)
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)


        # pred_max_top50 = sorted(pred_max_heap, reverse=True)  # 最大堆中保存的就是 Top 50 最大值
        # pred_min_top50 = sorted([-val for val in pred_min_heap])  # 还原负值，得到 Top 50 最小值

        # # 打印最终结果
        # print("Top 50 pred_max:", pred_max_top50)
        # print("Top 50 pred_min:", pred_min_top50)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_loco(model, dataloader, device, _class_=None, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)
            defect_type_list.extend(defect_type)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

        defect_type_list = np.array(defect_type_list)
        auroc_logic = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_both = (auroc_logic + auroc_struct) / 2

    return auroc_sp, auroc_logic, auroc_struct, auroc_both


def evaluation_uniad(model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            if reg_calib:
                en, de, reg = model({'image': img})
            else:
                en, de = model({'image': img})

            anomaly_map = torch.mean(F.mse_loss(de, en, reduction='none'), dim=1, keepdim=True)
            anomaly_map = F.interpolate(anomaly_map, size=(img.shape[-1], img.shape[-1]), mode='bilinear',
                                        align_corners=False)

            if reg_calib:
                if reg.shape[1] == 2:
                    reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                    reg_max = reg[:, 1].view(-1, 1, 1, 1)
                    anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                    # anomaly_map = anomaly_map - reg_max

                else:
                    reg = F.interpolate(reg, size=img.shape[-1], mode='bilinear', align_corners=True)
                    anomaly_map = anomaly_map - reg

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp, [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

    return


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            for i in range(3):
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

def reverse_normalization(normalized_image):

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(normalized_image.device)

    # 如果 normalized_image 是 numpy 数组，先转换为 PyTorch 张量
    if isinstance(normalized_image, np.ndarray):
        normalized_image = torch.tensor(normalized_image, dtype=torch.float32)

    # 反向归一化
    original_image = normalized_image * std[None, :, None, None] + mean[None, :, None, None]
    
    return original_image