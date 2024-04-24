#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
import cv2

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': [
            'Background',
            'Hat',
            'Hair',
            'Glove',
            'Sunglasses',
            'Upper-clothes',
            'Dress',
            'Coat',
            'Socks',
            'Pants',
            'Jumpsuits',
            'Scarf',
            'Skirt',
            'Face', 'Left-arm', 'Right-arm',
            'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': [
            'Background',
            'Hat',
            'Hair',
            'Sunglasses',
            'Upper-clothes',
            'Skirt',
            'Pants',
            'Dress',
            'Belt',
            'Left-shoe',
            'Right-shoe',
            'Face',
            'Left-leg',
            'Right-leg',
            'Left-arm',
            'Right-arm',
            'Bag',
            'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip',
                        choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='',
                        help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='',
                        help="path of output image folder.")
    parser.add_argument("--logits", action='store_true',
                        default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def get_head_neck_palette_atr(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    
    print("====== ATRRRRRR")

    palette = [
        0, 0, 0,  # Background
        255, 255, 255,  # Hat
        255, 255, 255,  # Hair
        255, 255, 255,  # Sunglasseshttps://snafty-train-lora.s3.ap-northeast-1.amazonaws.com/img-to-img/medias/segment_images/segment_mask_transparent_20240423-17071713892039.png
        0, 0, 0,  # Upper-clothes
        0, 0, 0,  # Skirt
        0, 0, 0,  # Pants
        0, 0, 0,  # Dress
        0, 0, 0,  # Belt
        0, 0, 0,  # Left-shoe
        0, 0, 0,  # Right-shoe
        255, 255, 255,  # Face
        0, 0, 0,  # Left-leg
        0, 0, 0,  # Right-leg
        0, 0, 0,  # Left-arm
        0, 0, 0,  # Right-arm
        0, 0, 0,  # Bag
        0, 0, 0,  # Scarf
    ]

    # # transform palette red color to white color
    # palette[0] = 255

    return palette


def get_head_neck_palette_lip(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    
    print("====== LIPPPPPP")

    palette = [
        0, 0, 0,  # Background
        255, 255, 255,  # Hat
        255, 255, 255,  # Hair
        255, 255, 255,  # Sunglasses
        0, 0, 0,  # Upper-clothes
        0, 0, 0,  # Skirt
        0, 0, 0,  # Pants
        0, 0, 0,  # Dress
        0, 0, 0,  # Belt
        0, 0, 0,  # Left-shoe
        0, 0, 0,  # Right-shoe
        255, 255, 255,  # Face
        0, 0, 0,  # Left-leg
        255, 255, 255,  # Right-leg
        0, 0, 0,  # Left-arm
        0, 0, 0,  # Right-arm
        0, 0, 0,  # Bag
        0, 0, 0,  # Scarf
    ]

    # # transform palette red color to white color
    # palette[0] = 255

    return palette

def get_head_neck_palette_pascal(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    
    print("====== PASCAL")

    pascal = [
        0, 0, 0,  # Head
        0, 0, 0,  # Head
        255, 255, 255,  # Hat
        0, 0, 0,  # Background
        0, 0, 0,  # Background
        0, 0, 0,  # Upper-clothes
        0, 0, 0,  # Skirt
    ]

    # # transform palette red color to white color
    # palette[0] = 255

    return pascal


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(
        root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_head_neck_palette_pascal(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(
                size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(
                args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(
                np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            if args.logits:
                logits_result_path = os.path.join(
                    args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()


def gen_mask_scale(datasets="lip", model_restore="checkpoints/lip.pth", gpu="0", input_dir="inputs", output_dir="outputs", logits=False, mask_scale=8):

    # gpus = [int(i) for i in args.gpu.split(',')]
    # assert len(gpus) == 1
    # if not args.gpu == 'None':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[datasets]['num_classes']
    input_size = dataset_settings[datasets]['input_size']
    label = dataset_settings[datasets]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(
        root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    palette = get_head_neck_palette_lip(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(
                size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            # increase the size of logi

            parsing_result_path = os.path.join(
                output_dir, img_name[:-4] + '.segment-scale.png')
            output_img = Image.fromarray(
                np.asarray(parsing_result, dtype=np.uint8))

            output_img.putpalette(palette)
            output_img.save(parsing_result_path)

            # increase the area of white color from parsing result
            parsing_result = cv2.imread(parsing_result_path)
            parsing_result = cv2.cvtColor(parsing_result, cv2.COLOR_BGR2GRAY)
            parsing_result = cv2.threshold(
                parsing_result, 128, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(parsing_result, kernel,
                                  iterations=mask_scale)

            cv2.imwrite(parsing_result_path, dilation)

            if logits:
                logits_result_path = os.path.join(
                    output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)

    return parsing_result_path


def gen_mask_without_scale(datasets="lip", model_restore="checkpoints/lip.pth", gpu="0", input_dir="inputs", output_dir="outputs", logits=False):

    # gpus = [int(i) for i in args.gpu.split(',')]
    # assert len(gpus) == 1
    # if not args.gpu == 'None':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[datasets]['num_classes']
    input_size = dataset_settings[datasets]['input_size']
    label = dataset_settings[datasets]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(
        root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    palette = get_head_neck_palette_lip(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(
                size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            # increase the size of logi

            parsing_result_path = os.path.join(
                output_dir, img_name[:-4] + '-segment-without-scale.png')
            output_img = Image.fromarray(
                np.asarray(parsing_result, dtype=np.uint8))

            output_img.putpalette(palette)
            output_img.save(parsing_result_path)

            if logits:
                logits_result_path = os.path.join(
                    output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)

    return parsing_result_path
