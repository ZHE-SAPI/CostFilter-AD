from pathlib import Path
from torchvision import transforms

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import random
from typing import Sequence
from PIL.ImageOps import exif_transpose
import json
import os
import numpy as np
import torch
from einops import rearrange
import cv2
import glob
import imgaug.augmenters as iaa
from dataset.perlin_noise import rand_perlin_2d_np
import scipy.ndimage as ndimage
import json
import threading

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class MVTecDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            resize=512,
            img_size=512,
            center_crop=False,
            train=True,
            class_name="",
            encoder_hidden_states=None,
            pt_root="/home/sysmanager/customer/Desktop/ZZ/anomaly/GLAD-main/train_anomaldino_save_path_1234/MVTec-AD"
                    ):
        self.pt_root = pt_root
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states

        self.resize = resize
        self.img_size = img_size
        self.center_crop = center_crop
        self.instance_data_root = instance_data_root
        self.type = 'train' if train else 'test'

        self.pt_file_lists = {}
        self.pt_index = {}
        self.locks = {}  # 用于线程安全

        if class_name is not None and class_name != '':
            self.set = 'signal class'
            self.class_name = class_name
            self.instance_prompt = instance_prompt
            self.instance_images_paths, self.instance_masks_paths = self.get_data_single_class(self.instance_data_root)
        else:
            self.set = 'mutil class'
            self.instance_images_paths, self.instance_masks_paths, self.class_name_list = self.get_data_mutil_class(self.instance_data_root)
        # print(self.instance_images_paths, self.instance_masks_paths, self.class_name_list)
        self.num_instance_images = len(self.instance_images_paths)
        self.num_mask_images = len(self.instance_masks_paths)
        self.instance_prompt_start = instance_prompt
        self._length = self.num_instance_images
        self.structure_grid_size = 8

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
        self.masking_default = {
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


        # 遍历所有类别，预加载每个类别的 pt 文件路径，并随机打乱
        for object_name in os.listdir(self.pt_root):
            # print('dataset object_name', object_name)
            object_path = os.path.join(self.pt_root, object_name, "train", "good")
            if os.path.isdir(object_path):
                pt_files = [os.path.join(object_path, f) for f in os.listdir(object_path) if f.endswith(".pt")]
                if pt_files:
                    self.pt_file_lists[object_name] = pt_files
                    random.shuffle(self.pt_file_lists[object_name])  # 打乱顺序
                    self.pt_index[object_name] = 0  # 初始化索引
                    self.locks[object_name] = threading.Lock()  # 初始化锁


    def __len__(self):
        return self._length


    def _get_next_pt_file(self, object_name):
        """
        顺序选择 .pt 文件，遍历列表，循环取样
        """
        if object_name not in self.pt_file_lists:
            raise FileNotFoundError(f"No .pt files found for object: {object_name}")

        # 加锁确保线程安全
        with self.locks[object_name]:
            # 获取当前索引
            index = self.pt_index[object_name]
            pt_file = self.pt_file_lists[object_name][index]

            # 更新索引：循环回到列表开始
            self.pt_index[object_name] = (index + 1) % len(self.pt_file_lists[object_name])

        return pt_file


    def _get_first_pt_file(self, object_name):
        """
        返回指定对象的第一个 .pt 文件
        """
        if object_name not in self.pt_file_lists:
            raise FileNotFoundError(f"No .pt files found for object: {object_name}")

        # 直接返回第一个 .pt 文件
        return self.pt_file_lists[object_name][0]


    def _get_first_3_pt_file(self, object_name, m):
        """
        返回指定对象的第一个 .pt 文件
        """
        if object_name not in self.pt_file_lists:
            raise FileNotFoundError(f"No .pt files found for object: {object_name}")

        # 直接返回第一个 .pt 文件
        return self.pt_file_lists[object_name][0:m]



    def _get_random_pt_files(self, object_name, m):
        """
        随机选择 m 个 .pt 文件
        """
        # 检查是否存在该对象的文件列表
        if object_name not in self.pt_file_lists:
            raise FileNotFoundError(f"No .pt files found for object: {object_name}")

        # 获取文件列表并检查是否足够多的文件
        pt_files = self.pt_file_lists[object_name]
        if len(pt_files) < m:
            raise ValueError(f"Not enough .pt files for object: {object_name}. Requested: {m}, Available: {len(pt_files)}")

        # 加锁确保线程安全
        with self.locks[object_name]:
            # 随机选择 m 个文件
            selected_files = random.sample(pt_files, m)

        return selected_files


    def _get_fullshot_pt_files(self, object_name):
        """
        随机选择 m 个 .pt 文件
        """
        # 检查是否存在该对象的文件列表
        if object_name not in self.pt_file_lists:
            raise FileNotFoundError(f"No .pt files found for object: {object_name}")

        # 获取文件列表并检查是否足够多的文件
        pt_files = self.pt_file_lists[object_name]
        m = len(pt_files) 
        # 加锁确保线程安全
        with self.locks[object_name]:
            # 随机选择 m 个文件
            selected_files = random.sample(pt_files, m)

        return selected_files



    def __getitem__(self, index):


        example = {}

        instance_images_path = self.instance_images_paths[index]
        if self.set == 'mutil class':
            self.class_name = self.class_name_list[index]
            self.instance_prompt = self.instance_prompt_start + self.class_name

        instance_image = Image.open(instance_images_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_path"] = self.instance_images_paths[index]
        instance_mask_path = self.instance_masks_paths[index]

        if instance_mask_path:
            instance_mask = Image.open(instance_mask_path).convert('L')
            instance_mask = exif_transpose(instance_mask)
            example["instance_label"] = 1
        else:
            instance_mask = Image.fromarray(np.zeros([*instance_image.size]))
            example["instance_label"] = 0

        example["instance_images"], example["instance_masks"] = self.transformer_compose(instance_image, instance_mask)

        if self.encoder_hidden_states:
            example["encoder_hidden_states"] = self.encoder_hidden_states
        else:
            if self.tokenizer:
                text_inputs = self.tokenizer(
                    self.instance_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                example["instance_prompt_ids"] = text_inputs.input_ids
                example["instance_attention_mask"] = text_inputs.attention_mask

        if self.type == 'train':
            transforms_resize = transforms.Resize((self.resize, self.resize),
                                                  interpolation=transforms.InterpolationMode.BILINEAR)
            instance_image_resize = transforms_resize(instance_image)
            anomaly_image, anomaly_mask, anomaly_label, beta = self.generate_anomaly(np.array(instance_image_resize))
            example["anomaly_images"], example["anomaly_masks"] = self.transformer_compose(anomaly_image, anomaly_mask)
            # example["anomaly_masks"] = example["anomaly_masks"] * beta
            example["instance_label"] = anomaly_label
            example["instance_prompt"] = self.instance_prompt


            # 获取 object_name 并加载 .pt 文件
            object_name = self.get_object_name_from_path(example["instance_path"], self.masking_default)
            pt_file_paths = self._get_random_pt_files(object_name, 4)  # 随机选 m 个 .pt 文件
            
            features_ref_3_list = []
            features_ref_6_list = []
            features_ref_9_list = []
            features_ref_12_list = []
            
            preloaded_data = {path: torch.load(path, map_location='cpu') for path in pt_file_paths}

            for pt_file_path in pt_file_paths:
                # 加载每个 .pt 文件
                data_dict = preloaded_data[pt_file_path]
                
                # 将特征添加到对应列表
                features_ref_3_list.append(data_dict["features_ref_3"][0:1024]) # [0:1024]
                features_ref_6_list.append(data_dict["features_ref_6"][0:1024]) # [0:1024]
                features_ref_9_list.append(data_dict["features_ref_9"][0:1024]) # [0:1024]
                features_ref_12_list.append(data_dict["features_ref_12"][0:1024]) # [0:1024]

            example["features_ref_3"] = torch.cat(features_ref_3_list, dim=0)
            example["features_ref_6"] = torch.cat(features_ref_6_list, dim=0)
            example["features_ref_9"] = torch.cat(features_ref_9_list, dim=0)
            example["features_ref_12"] = torch.cat(features_ref_12_list, dim=0)

            # 保存选取的文件路径
            example["features_ref_path"] = pt_file_paths
             
        else:
            transforms_resize = transforms.Resize((self.resize, self.resize),
                                                  interpolation=transforms.InterpolationMode.BILINEAR)
            instance_image_resize = transforms_resize(instance_image)

            if self.class_name in ["screw", "bottle", "capsule", "zipper", "bracket_black", "bracket_brown",
                                "metal_plate"]:
                mode = 1
            elif self.class_name in ["hazelnut", "pill", "metal_nut", "toothbrush", "candle",  "cashew", "chewinggum",
                                        "fryum", "macaroni1", "macaroni2", "pipe_fryum", "bracket_white"]:
                mode = 2
            elif self.class_name in ["tile", "grid", "cable", "carpet", "leather", "wood", "transistor", "capsules",
                                       "pcb1", "pcb2", "pcb3", "pcb4", "connector", "tubes"]:
                mode = 3
            else:
                mode = 3
            foreground_mask = self.generate_target_foreground_mask(np.array(instance_image_resize), mode=mode).astype(np.float32)

            sigma = 6.0

            object_mask = ndimage.gaussian_filter(foreground_mask, sigma=sigma)
            object_mask = np.where(object_mask > 0, 1.0, 0.0)
            object_mask = ndimage.gaussian_filter(object_mask, sigma=sigma)
            example["object_mask"] = object_mask

            # 获取 object_name 并加载 .pt 文件
            object_name = self.get_object_name_from_path(example["instance_path"], self.masking_default)
            pt_file_paths = self._get_random_pt_files(object_name, 48)  # 随机选 m 个 .pt 文件
            
            features_ref_3_list = []
            features_ref_6_list = []
            features_ref_9_list = []
            features_ref_12_list = []
            template_image_list = []

            for pt_file_path in pt_file_paths:
                # 加载每个 .pt 文件
                data_dict = torch.load(pt_file_path, map_location='cpu')
        
                features_ref_3_list.append(data_dict["features_ref_3"])
                features_ref_6_list.append(data_dict["features_ref_6"])
                features_ref_9_list.append(data_dict["features_ref_9"])
                features_ref_12_list.append(data_dict["features_ref_12"])
                template_image_list.append(data_dict["instance_images"])
            
            # 沿维度 0 拼接所有特征
            example["features_ref_3"] = torch.cat(features_ref_3_list, dim=0)
            example["features_ref_6"] = torch.cat(features_ref_6_list, dim=0)
            example["features_ref_9"] = torch.cat(features_ref_9_list, dim=0)
            example["features_ref_12"] = torch.cat(features_ref_12_list, dim=0)
            # 保存选取的文件路径
            example["features_ref_path"] = pt_file_paths

            example["template_image"] = torch.cat(template_image_list, dim=0)

        return example

    def get_object_name_from_path(self, instance_images_path, masking_default):
        """
        从图像路径中提取 object_name，匹配 masking_default 字典中的键。

        Args:
            instance_images_path (str): 单个图像路径的字符串。
            masking_default (dict): 包含 object_name 及其对应 masking 值的字典。

        Returns:
            str: 匹配到的 object_name。如果没有匹配，返回 None。
        """
        # 遍历 masking_default 的键，检查路径中是否包含对应的 object_name
        for key in masking_default.keys():
            if f"/{key}/" in instance_images_path:  # 判断路径中是否包含 key
                # print(f"Matched object_name: {key}") # metal_nut
                return key  # 返回匹配到的 object_name

        # 如果没有匹配项，返回 None
        return None



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

        if self.class_name in ["screw", "bottle", "capsule", "zipper", "bracket_black", "bracket_brown",
                                "metal_plate"]:
            mode = 1
        elif self.class_name in ["hazelnut", "pill", "metal_nut", "toothbrush", "candle", "cashew", "chewinggum",
                                    "fryum", "macaroni1", "macaroni2", "pipe_fryum", "bracket_white"]:
            mode = 2
        elif self.class_name in ["tile", "grid", "cable", "carpet", "leather", "wood", "transistor", "capsules",
                                    "pcb1", "pcb2", "pcb3", "pcb4", "connector", "tubes"]:
            mode = 3
        else:
            mode = 3

        foreground_mask = self.generate_target_foreground_mask(image, mode=mode).astype(np.float32)
        perlin_thr = np.expand_dims(foreground_mask, axis=2) * perlin_thr

        anomaly_source_thr = anomaly_source_image * perlin_thr
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * anomaly_source_thr + beta * image * perlin_thr

        anomaly = torch.rand(1).numpy()[0]
        if anomaly > 0.5:
            augmented_image = augmented_image.astype(np.uint8)
            msk = (perlin_thr * 255).astype(np.uint8).squeeze()
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
            anomaly_source_img = cv2.resize(texture_source_img, (self.resize, self.resize)).astype(np.float32)
            anomaly_source_img = aug(image=img) ### 增强异常源图像
        else:
            structure_source_img = aug(image=img)

            assert self.resize % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
            grid_w = self.resize // self.structure_grid_size
            grid_h = self.resize // self.structure_grid_size

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

    def transformer_compose(self, image, mask):
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

    def get_data_single_class(self, instance_data_root):
        image_list = []
        mask_list = []
        object_fold = self.class_name
        mask_fold = '' if self.type == 'train' else 'ground_truth'

        foldnames = os.listdir(os.path.join(instance_data_root, self.class_name, self.type))
        for image_fold in foldnames:
            filenames = os.listdir(os.path.join(instance_data_root, self.class_name, self.type, image_fold))
            for img in filenames:
                image_list.append(os.path.join(instance_data_root, self.class_name, self.type, image_fold, img))
                if image_fold == 'good':
                    mask_list.append(None)
                else:
                    if 'VisA' in instance_data_root or 'PCBBank' in instance_data_root:
                        mask_name = img.split('.')[0] + ".png"
                    else:
                        mask_name = img.split('.')[0] + "_mask." + img.split('.')[1]
                    mask_list.append(os.path.join(instance_data_root, object_fold, mask_fold, image_fold, mask_name))
        return image_list, mask_list

    def get_data_single_test(self, instance_data_root):
        image_list = []
        mask_list = []
        for image_fold in os.listdir(os.path.join(instance_data_root, self.class_name)):
            for img in os.listdir(os.path.join(instance_data_root, self.class_name, image_fold)):
                image_list.append(os.path.join(instance_data_root, self.class_name, image_fold, img))
                mask_list.append(None)
        return image_list, mask_list

    def get_data_mutil_class(self, instance_data_root):
        image_list = []
        mask_list = []
        class_name = []
        for object_fold in os.listdir(instance_data_root):
            mask_fold = '' if self.type == 'train' else 'ground_truth'
            for image_fold in os.listdir(os.path.join(instance_data_root, object_fold, self.type)):
                for img in os.listdir(os.path.join(instance_data_root, object_fold, self.type, image_fold)):
                    image_list.append(os.path.join(instance_data_root, object_fold, self.type, image_fold, img))
                    if image_fold == 'good':
                        mask_list.append(None)
                    else:
                        mask_name = img.split('.')[0] + "_mask." + img.split('.')[1]
                        mask_list.append(
                            os.path.join(instance_data_root, object_fold, mask_fold, image_fold, mask_name))
                    class_name.append(object_fold)
        return image_list, mask_list, class_name

class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
