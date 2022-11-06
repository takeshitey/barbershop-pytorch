import scipy
import os
import cv2
import itertools
import random
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms import transforms


def get_files(BASE_DIR):
    files = []
    images_dir = os.path.join(BASE_DIR, "raw", "train")
    masks_dir = os.path.join(BASE_DIR, "masks", "train")

    images = os.listdir(images_dir)

    for image in images:
        files.append((os.path.join(images_dir, image), os.path.join(masks_dir, image)))

    return files


def arcface_process(path):
    # ArcFace / ResNet50 trained on VGGFace2
    img = Image.open(path).convert("RGB")
    img = transforms.Resize((224, 224))(img)
    img = transforms.ToTensor()(img)  # [0, 1]
    img = img * 255  # [0, 255]
    img = transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])(img)
    img = img.float()
    return img


def dilate_erosion_mask(mask_path, size):
    # Mask
    mask = Image.open(mask_path).convert("RGB")
    mask = transforms.Resize((size, size))(mask)
    mask = transforms.ToTensor()(mask)  # [0, 1]

    # Hair mask + Hair image
    hair_mask = mask[0, ...]
    # hair_mask = torch.unsqueeze(hair_mask, axis=0)
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=5)
    hair_mask_erode = scipy.ndimage.binary_erosion(
        hair_mask, iterations=5
    )  # , structure=np.ones((3, 3)).astype(hair_mask.dtype))

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate), torch.from_numpy(hair_mask_erode)


def custom_decode_labels(mask, region_index: int):
    region_mask = np.where(mask == region_index, np.ones_like(mask), np.zeros_like(mask))
    return region_mask

def process_image_mask(img_path, mask_path, seg_parts, size=None, normalize=True):
    # Full image
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = transforms.Resize((size, size))(img)
    img = transforms.ToTensor()(img)
    if normalize:
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

    # Mask
    mask = np.array(Image.open(mask_path))
    parts_out = {}
    for part_id in seg_parts:
        part_mask = custom_decode_labels(mask, part_id)
        if size is not None:
            part_mask = cv2.resize(part_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        part_mask = torch.from_numpy(part_mask).unsqueeze(0)

        part_image = img * part_mask
        parts_out[part_id] = {'mask': part_mask, 'masked_image': part_image}

    outputs = {'img': img, 'img_mask': mask, 'parts_out': parts_out}

    return outputs