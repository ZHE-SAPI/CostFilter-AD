from __future__ import division

import json
import logging

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
import imgaug.augmenters as iaa
from einops import rearrange
from PIL.ImageOps import exif_transpose
from datasets.perlin_noise import rand_perlin_2d_np
import torch
logger = logging.getLogger("global_logger")
import cv2
import glob
import scipy.ndimage as ndimage
import numpy as np
cls_list = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
            'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

# cls_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2',
#             'pcb1','pcb2','pcb3','pcb4','pipe_fryum']
            
def build_custom_dataloader(cfg, training, distributed=True):

    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
        input_size = cfg["input_size"],
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    if training:
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
        input_size=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        self.input_size = input_size

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.structure_grid_size = 8
        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]

        
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        # print('meta.get("clsname", None)', meta.get("clsname", None))
        # print('filename', filename)
        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
            input["clslabel"] = cls_list.index(meta["clsname"])
            # print('input["clsname"]', input["clsname"])
            # print('input["clslabel"]', input["clslabel"])
        else:
            input["clsname"] = filename.split("/")[-4]
            input["clslabel"] = cls_list.index(input["clsname"])
            # print('input["clsname"]', input["clsname"])
            # print('input["clslabel"]', input["clslabel"])

        self.clsname = input["clsname"]
        # print('self.clsname', self.clsname)

        # image_before_normal = exif_transpose(image)
        image_before_normal = Image.fromarray(image, "RGB")

        image = Image.fromarray(image, "RGB")


        # read / generate mask
        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)


        if self.training == True:
            # print('self.input_size', self.input_size)
            transform_resize_img = transforms.Resize((self.input_size), interpolation=transforms.InterpolationMode.BILINEAR)
            transform_resize_lbl = transforms.Resize((self.input_size), interpolation=transforms.InterpolationMode.NEAREST)

            instance_image_resize = transform_resize_img(image_before_normal)
            anomaly_image, anomaly_mask, anomaly_label, beta = self.generate_anomaly(np.array(instance_image_resize))
            anomaly_image = transform_resize_img(anomaly_image)
            anomaly_image = transforms.ToTensor()(anomaly_image)
            image = self.normalize_fn(anomaly_image)

            mask = transform_resize_lbl(anomaly_mask)
            mask = transforms.ToTensor()(mask)
            # example["anomaly_masks"] = example["anomaly_masks"] * beta
            input["label"] = anomaly_label



            input.update({"image": image, "mask": mask})

        else:
            transform_resize_img = transforms.Resize((self.input_size), interpolation=transforms.InterpolationMode.BILINEAR)
            instance_image_resize = transform_resize_img(image_before_normal)

            if self.clsname in ["screw", "bottle", "capsule", "zipper", "bracket_black", "bracket_brown",
                                "metal_plate"]:
                mode = 1
            elif self.clsname in ["hazelnut", "pill", "metal_nut", "toothbrush", "candle",  "cashew", "chewinggum",
                                        "fryum", "macaroni1", "macaroni2", "pipe_fryum", "bracket_white"]:
                mode = 2
            elif self.clsname in ["tile", "grid", "cable", "carpet", "leather", "wood", "transistor", "capsules",
                                       "pcb1", "pcb2", "pcb3", "pcb4", "connector", "tubes"]:
                mode = 3
            else:
                mode = 3
            foreground_mask = self.generate_target_foreground_mask(np.array(instance_image_resize), mode=mode).astype(np.float32)

            sigma = 6.0

            object_mask = ndimage.gaussian_filter(foreground_mask, sigma=sigma)
            object_mask = np.where(object_mask > 0, 1.0, 0.0)
            object_mask = ndimage.gaussian_filter(object_mask, sigma=sigma)
            input.update({"object_mask": object_mask})
            input.update({"image": image, "mask": mask})
        return input




    def generate_anomaly(self, image):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )

        anomaly_source_paths = sorted(glob.glob("/home/sysmanager/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets/dtd/images" + "/*/*.jpg"))
        anomaly_source_image = self.anomaly_source(image, anomaly_source_paths, aug)

        perlin_scale = 6
        min_perlin_scale = 0
        threshold = 0.3 # 之前是0.3
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2).astype(np.float32)

        if self.clsname in ["screw", "bottle", "capsule", "zipper", "bracket_black", "bracket_brown",
                                "metal_plate"]:
            mode = 1
        elif self.clsname in ["hazelnut", "pill", "metal_nut", "toothbrush", "candle", "cashew", "chewinggum",
                                    "fryum", "macaroni1", "macaroni2", "pipe_fryum", "bracket_white"]:
            mode = 2
        elif self.clsname in ["tile", "grid", "cable", "carpet", "leather", "wood", "transistor", "capsules",
                                    "pcb1", "pcb2", "pcb3", "pcb4", "connector", "tubes"]:
            mode = 3
        else:
            mode = 3

        foreground_mask = self.generate_target_foreground_mask(image, mode=mode).astype(np.float32)
        perlin_thr = np.expand_dims(foreground_mask, axis=2) * perlin_thr

        # cv2.imshow("perlin_thr", perlin_thr)
        # cv2.imshow("foreground_mask", foreground_mask)
        # cv2.waitKey()

        anomaly_source_thr = anomaly_source_image * perlin_thr
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * anomaly_source_thr + beta * image * perlin_thr

        # cv2.imwrite("anomaly_source_image.png", anomaly_source_image[:, :, ::-1])
        # cv2.imwrite("perlin_thr.png", perlin_thr[:, :, ::-1] * 255)
        # cv2.imwrite("augmented_image.png",augmented_image[:, :, ::-1])
        # print(111111111111111111111111111111111111111)
        # exit()

        anomaly = torch.rand(1).numpy()[0]
        if anomaly > 0.5:
            augmented_image = augmented_image.astype(np.uint8)
            msk = (perlin_thr * 255).astype(np.uint8).squeeze()
            # augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 0.0 if np.sum(msk) == 0 else 1.0

            return Image.fromarray(augmented_image), Image.fromarray(msk), np.array([has_anomaly], dtype=np.float32), 1 - beta
        else:
            mask = np.zeros_like(perlin_thr).astype(np.uint8).squeeze()
            return Image.fromarray(image.astype(np.uint8)), Image.fromarray(mask), np.array([0.0], dtype=np.float32), 0.0


    def generate_target_foreground_mask(self, img: np.ndarray, mode=1) -> np.ndarray:
        # Converting RGB into grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if mode == 1:  # USING THIS FOR NOT WHITE BACKGROUND
            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(bool).astype(int)
            # Inverting mask for foreground mask
            target_foreground_mask = -(target_background_mask - 1)

        elif mode == 2:  # USING THIS FOR DARK BACKGROUND
            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(bool).astype(int)
            target_foreground_mask = target_background_mask

        elif mode == 3:
            target_foreground_mask = np.ones(img_gray.shape)

        return target_foreground_mask




    def anomaly_source(self, img, anomaly_path_list, aug):
        p = np.random.uniform()
        if p < 0.5:
            # TODO: None texture
            idx = np.random.choice(len(anomaly_path_list))
            texture_source_img = cv2.imread(anomaly_path_list[idx])
            texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
            anomaly_source_img = cv2.resize(texture_source_img, self.input_size).astype(np.float32)
            anomaly_source_img = aug(image=img) ### 增强异常源图像
        else:
            structure_source_img = aug(image=img)

            assert self.input_size[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
            assert self.input_size[1] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
            grid_w = self.input_size[0] // self.structure_grid_size
            grid_h = self.input_size[1] // self.structure_grid_size

            structure_source_img = rearrange(
                tensor=structure_source_img,
                pattern='(h gh) (w gw) c -> (h w) gw gh c',
                gw=grid_w,
                gh=grid_h
            )
            disordered_idx = np.arange(structure_source_img.shape[0])
            np.random.shuffle(disordered_idx)

            anomaly_source_img = rearrange(
                tensor=structure_source_img[disordered_idx],
                pattern='(h w) gw gh c -> (h gh) (w gw) c',
                h=self.structure_grid_size,
                w=self.structure_grid_size
            ).astype(np.float32)

        return anomaly_source_img