import os

import yaml
from easydict import EasyDict as edict

import os.path as path
import cv2
import pickle as pkl
import numpy as np

from tqdm import tqdm

import torch
import torch.nn.functional as F

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return edict(config_data)

def unify_image(img_raw, target_size=(192, 64), pad_color=(114, 114, 114)):
    src_h, src_w = img_raw.shape[:2]
    dst_h, dst_w = target_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if img_raw.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(img_raw, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = img_raw

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return image_dst

def unify_images(images_list):
    images_dst = []
    for image in images_list:
        images_dst.append(unify_image(image, (192, 64)))
    images_dst = np.array(images_dst)
    return images_dst

def compute_mean_std(img_path):
    img_files = os.listdir(img_path)
    imgs = []
    for img_file in tqdm(img_files, total=len(img_files)):
        with open(path.join(img_path, img_file), 'rb') as file:
            for img in pkl.load(file):
                imgs.append(img)

    cumulative_mean = np.zeros(3)
    cumulative_std = np.zeros(3)

    for img in tqdm(imgs, total=len(imgs)):
        img = img / 255
        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(imgs)
    std = cumulative_std / len(imgs)

    return mean, std

def positive_negative_sample_num(data):
    pos_num = 0
    neg_num = 0
    for label in data['crossing']:
        if label == 1:
            pos_num += 1
        else:
            neg_num += 1
    return pos_num, neg_num

def normalize_data(data):
    for key in data:
        if key == 'tte' or key == 'crossing' or key == 'seg' or key == 'ped_id' or key == 'image' or key == 'center':
            continue
        data[key] = (data[key] - data[key].mean()) / data[key].std()
    return data

def calculate_kl_divergence(mu, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl

def elbo_loss_local(output, target, bayesian_cnn, kl_weight=1.0):
    ce_loss = F.nll_loss(output, target)

    kl_loss = 0.0
    for module in bayesian_cnn.children():
        if hasattr(module, 'weight_mu') and hasattr(module, 'weight_rho'):
            kl_loss += calculate_kl_divergence(module.weight_mu, module.weight_rho)

    total_loss = ce_loss + kl_weight * kl_loss
    return total_loss


def elbo_loss_final(output_mean, output_std, target, kl_weight=1.0):
    ce_loss = F.nll_loss(output_mean, target)

    kl_loss = -0.5 * torch.sum(1 - output_std.pow(2) - output_mean.pow(2) + torch.log(output_std.pow(2)))

    total_loss = ce_loss + kl_weight * kl_loss
    return total_loss

def data_prepare(path):
    with open(path, 'rb') as file:
        data = pkl.load(file)
    for key in data:
        if key == 'tte' or key== 'crossing' or key == 'seg':
            continue
        data[key] = data[key][:, 0:16]
    try:
        del data['seg']
    except KeyError:
        pass
    with open(path, 'wb') as file:
        pkl.dump(data, file)