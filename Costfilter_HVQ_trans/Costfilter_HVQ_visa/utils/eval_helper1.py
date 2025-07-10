import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from skimage import measure
import pandas as pd
from statistics import mean
from sklearn.metrics import auc
import pickle

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    pred_imgs = outputs['pred_imgs'].cpu().numpy()
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            pred_imgs=pred_imgs[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )

def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    pred_imgs = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])
        pred_imgs.append(npz['pred_imgs'])
    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    pred_imgs = np.asarray(pred_imgs)                  # N x 3 x H x W
    return fileinfos, preds, masks, pred_imgs


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def compute_pro(masks, amaps, num_th=200):

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

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
        fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc



class EvalMetric:
    def __init__(self, preds, labels, masks=None):
        self.preds = preds  # 模型预测值
        self.labels = labels  # 图像级标签
        self.masks = masks  # 像素级标签 (可选)
        self.scores = torch.topk(torch.tensor(self.preds[:, None, ...]).cuda().view(self.preds.shape[0], -1), 250, dim=1)[0].mean(dim=1)  # 获取每张图像前 250 个最高值的平均值
        self.scores = normalize(np.array((self.scores).cpu()))

        assert not np.any(np.isnan(self.masks)), "masks contains NaN!"
        assert not np.any(np.isnan(self.preds)), "preds contains NaN!"
        assert not np.any(np.isinf(self.masks)), "masks contains Inf!"
        assert not np.any(np.isinf(self.preds)), "preds contains Inf!"



    def eval_image_metrics(self):
        # 计算 Image-AP 和 Image-F1-max
        precisions_image, recalls_image, _ = precision_recall_curve(self.labels, self.scores)
        f1_scores_image = (2 * precisions_image * recalls_image) / (precisions_image + recalls_image)
        best_f1_scores_image = np.max(f1_scores_image[np.isfinite(f1_scores_image)])
        AP_image = average_precision_score(self.labels, self.scores)
        auroc_image = roc_auc_score(self.labels, self.scores)

        return AP_image, best_f1_scores_image, auroc_image

    def eval_pixel_metrics(self):
        # 计算 Pixel-AP 和 Pixel-F1-max
        precisions_pixel, recalls_pixel, _ = precision_recall_curve(self.masks.ravel(), self.preds.ravel())

        f1_scores_pixel = (2 * precisions_pixel * recalls_pixel) / (precisions_pixel + recalls_pixel)
        best_f1_scores_pixel = np.max(f1_scores_pixel[np.isfinite(f1_scores_pixel)])
        AP_pixel = average_precision_score(self.masks.ravel(), self.preds.ravel())
        auroc_pixel = roc_auc_score(self.masks.ravel(), self.preds.ravel())

        return AP_pixel, best_f1_scores_pixel, auroc_pixel

    def eval_pixel_pro(self):  
        pro = compute_pro(self.masks, self.preds)
        return pro




class EvalDataMeta:
    def __init__(self, preds, masks, labels_cls):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W
        self.label01cls = labels_cls


class EvalImageAP(EvalMetric):
    def eval_(self):
        ap_score, _, _ = self.eval_image_metrics()
        return ap_score


class EvalImageF1Max(EvalMetric):
    def eval_(self):
        _, best_f1, _ = self.eval_image_metrics()
        return best_f1

class EvalImageAuroc_image(EvalMetric):
    def eval_(self):
        _, _, auroc_image = self.eval_image_metrics()
        return auroc_image


class EvalPixelAP(EvalMetric):
    def eval_(self):
        ap_score, _, _ = self.eval_pixel_metrics()
        return ap_score


class EvalPixelF1Max(EvalMetric):
    def eval_(self):
        _, best_f1, _ = self.eval_pixel_metrics()
        return best_f1

class EvalPixelAuroc_pixel(EvalMetric):
    def eval_(self):
        _, _, auroc_pixel = self.eval_pixel_metrics()
        return auroc_pixel

class EvalPixelPRO(EvalMetric):
    def eval_(self):
        return self.eval_pixel_pro()





class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )

class EvalImageTop250(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape  # 获取批量大小和图像维度
        preds = torch.tensor(preds[:, None, ...]).cuda()  # 转换为 PyTorch 张量，形状 N x 1 x H x W
        preds_flat = preds.view(N, -1)  # 展平为 N x (H*W)
        score = torch.topk(preds_flat, 250, dim=1)[0].mean(dim=1)  # 获取每张图像前 250 个最高值的平均值
        return score.cpu().numpy()  # 转回 NumPy 格式




class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
    "top250": EvalImageTop250,  # 新增的指标
    "IImageap": EvalImageAP,
    "IImagef1max": EvalImageF1Max,
    "IImageauroc": EvalImageAuroc_image,
    "PPixelap": EvalPixelAP,
    "PPixelf1max": EvalPixelF1Max,
    "PPixelpro": EvalPixelPRO,
    "PPixelauroc": EvalPixelAuroc_pixel,
}



def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        labels_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
                labels_cls.append(((mask.reshape(mask.shape[0]*mask.shape[1]).sum(axis=0)) != 0).astype(int))

        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        labels_cls = np.array(labels_cls)  # N

        N, _, _ = preds_cls.shape  # 获取批量大小和图像维度
        preds_flat_ = torch.tensor(preds_cls[:, None, ...]).cuda().view(N, -1)  # 展平为 N x (H*W)
        score_ = torch.topk(preds_flat_, 250, dim=1)[0].mean(dim=1)  # 获取每张图像前 250 个最高值的平均值

        # # 保存到本地
        # with open(f"/home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_visa/H+O_VISA_curve/filtered_data-HO-{clsname}.pkl", "wb") as f:
        #     pickle.dump({"scores": normalize(np.array((score_).cpu())) , "labels": labels_cls, "preds": preds_cls, "masks": masks_cls}, f)

        # print("Filtered data saved successfully!")


        data_meta = EvalDataMeta(preds_cls, masks_cls, labels_cls)
        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})

                if evalname.startswith("IImage") or evalname.startswith("PPixel"):
                    eval_method = eval_lookup_table[evalname](data_meta.preds, data_meta.label01cls, data_meta.masks)
                    auc = eval_method.eval_()
                else:
                    eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                    auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics

def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
