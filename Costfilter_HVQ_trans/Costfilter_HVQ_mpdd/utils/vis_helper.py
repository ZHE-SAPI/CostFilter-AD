import os

import cv2
import numpy as np
from datasets.image_reader import build_image_reader

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


# def visualize_compound(fileinfos, preds, masks, pred_imgs, cfg_vis, cfg_reader):
#     vis_dir = cfg_vis.save_dir
#     max_score = cfg_vis.get("max_score", None)
#     min_score = cfg_vis.get("min_score", None)
#     max_score = preds.max() if not max_score else max_score
#     min_score = preds.min() if not min_score else min_score

#     image_reader = build_image_reader(cfg_reader)

#     for i, fileinfo in enumerate(fileinfos):
#         clsname = fileinfo["clsname"]
#         filename = fileinfo["filename"]
#         filedir, filename = os.path.split(filename)
#         _, defename = os.path.split(filedir)
#         save_dir = os.path.join(vis_dir, clsname, defename)
#         os.makedirs(save_dir, exist_ok=True)

#         # read image
#         h, w = int(fileinfo["height"]), int(fileinfo["width"])
#         image = image_reader(fileinfo["filename"])
#         pred = preds[i][:, :, None].repeat(3, 2)
#         pred_gray = cv2.resize(pred, (w, h))
#         pred = pred_gray
#         # pred imgs
#         pred_img = np.transpose(pred_imgs[i],(1,2,0))
#         pred_img = np.clip((pred_img * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.uint8)
#         pred_img = cv2.resize(pred_img, (w,h))
#         # self normalize just for analysis
#         scoremap_self = apply_ad_scoremap(image, normalize(pred))
#         # global normalize
#         pred = np.clip(pred, min_score, max_score)
#         pred = normalize(pred, max_score, min_score)
#         scoremap_global = apply_ad_scoremap(image, pred)

#         if masks is not None:
#             mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
#             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             save_path = os.path.join(save_dir, filename)
#             if mask.sum() == 0:
#                 scoremap = np.vstack([image, pred_img, scoremap_global, scoremap_self])
#             else:
#                 scoremap = np.vstack([image, pred_img, mask, scoremap_global, scoremap_self])
#         else:
#             scoremap = np.vstack([image, scoremap_global, scoremap_self])

#         scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(save_path, scoremap)

def visualize_compound(fileinfos, preds, masks, pred_imgs, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred_gray = cv2.resize(pred, (w, h))  # Resize pred_gray to match the image
        pred = pred_gray  # Update pred after resizing

        # Define save paths for each component
        base_name = filename.split('.')[0]
        save_path_image = os.path.join(save_dir, f"{base_name}_image_HVQ.JPG")
        save_path_pred_img = os.path.join(save_dir, f"{base_name}_pred_img_HVQ.JPG")
        save_path_scoremap_global = os.path.join(save_dir, f"{base_name}_scoremap_global_HVQ.JPG")
        save_path_scoremap_self = os.path.join(save_dir, f"{base_name}_scoremap_self_HVQ.JPG")
        save_path_mask = os.path.join(save_dir, f"{base_name}_mask_HVQ.JPG")
        save_path_pred_gray = os.path.join(save_dir, f"{base_name}_pred_gray_HVQ.JPG")  # New save path for pred_gray

        # pred imgs
        pred_img = np.transpose(pred_imgs[i], (1, 2, 0))
        pred_img = np.clip((pred_img * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.uint8)
        pred_img = cv2.resize(pred_img, (w, h))
        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Save each component individually
        cv2.imwrite(save_path_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path_pred_img, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path_scoremap_global, cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path_scoremap_self, cv2.cvtColor(scoremap_self, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_path_pred_gray, pred_gray)  # Save pred_gray

        if masks is not None:
            cv2.imwrite(save_path_mask, mask)

        # # Debug prints for saved paths
        # print(f"Saved image to: {save_path_image}")
        # print(f"Saved pred_img to: {save_path_pred_img}")
        # print(f"Saved scoremap_global to: {save_path_scoremap_global}")
        # print(f"Saved scoremap_self to: {save_path_scoremap_self}")
        # print(f"Saved pred_gray to: {save_path_pred_gray}")
        # if masks is not None:
        #     print(f"Saved mask to: {save_path_mask}")



def visualize_single(fileinfos, preds, cfg_vis, cfg_reader):
    vis_dir = cfg_vis.save_dir
    max_score = cfg_vis.get("max_score", None)
    min_score = cfg_vis.get("min_score", None)
    max_score = preds.max() if not max_score else max_score
    min_score = preds.min() if not min_score else min_score

    image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        pred = cv2.resize(pred, (w, h))

        # write global normalize image
        pred = np.clip(pred, min_score, max_score)
        pred = normalize(pred, max_score, min_score)
        scoremap_global = apply_ad_scoremap(image, pred)

        save_path = os.path.join(save_dir, filename)
        scoremap_global = cv2.cvtColor(scoremap_global, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap_global)
