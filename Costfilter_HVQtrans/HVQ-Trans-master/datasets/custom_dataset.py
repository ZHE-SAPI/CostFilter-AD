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

logger = logging.getLogger("global_logger")

# cls_list = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
#             'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

# cls_list = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2',
#             'pcb1','pcb2','pcb3','pcb4','pipe_fryum']

cls_list = ['tubes','metal_plate','connector','bracket_white','bracket_brown','bracket_black']            


# cls_list = ['01','02','03']            


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
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

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
        input.update({"image": image, "mask": mask})
        return input
