
import os
import datetime
import random
import torch
import torch.nn as nn
import numpy as np
from utils.visualize import get_color_pallete
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from models import get_model
from global_config import nclass, task_num, classes_dict
from train import parse_args

def sync_transform_no_mask(img):
    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    crop_size = 480
    # random scale (short edge)
    short_size = random.randint(
        int(520 * 0.5), int(520 * 2.0))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    # gaussian blur as in PSP
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    # final transform
    img = img_transform(img)
    return img

def img_transform(img):
    return np.array(img)

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
    model = get_model(
        version=args.version,
        nclass=nclass,
        backbone=args.backbone,
        pretrained_base=args.pretrained_base,
        norm_layer=BatchNorm2d,
        jpu=args.jpu,
        pth_path=args.pth_path).to(device)
    if args.device != 'cpu' and len(args.gpus) > 1: # 多GPU训练
            model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoints, map_location=device))
    # model_param_count = count_parameters(model)
    print('Finished loading model!')
    model.eval()

    images_list = os.listdir(args.input_dir)

    for img in images_list:
        filenames = img.split('.')[0]
        img = os.path.join(args.input_dir, img)
        image = Image.open(img).convert('RGB')
        image = sync_transform_no_mask(image)
        images = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(images)
            img_labels = []
            for i in range(task_num):
                Keys = list(classes_dict[i].keys())
                img_labels.append(
                    Keys[outputs[i].max(1)[1]]
                )

            print('FileName : {} -- Labels : {}'.format(filenames, ' '.join(img_labels)))


if __name__ == '__main__':
    args = parse_args()
    args.distributed = False
    args.input_dir = 'running/demo/'
    # Pretrained model path
    args.save_dir = os.path.join('running', '2024-08-14_V1.0')
    args.log_dir = os.path.join(args.save_dir, 'logs')
    # args.visual = os.path.join(args.save_dir, 'logs', 'visual')
    args.checkpoints = os.path.join(args.save_dir, 'checkpoint', 'mtcnet_best.pth')
    demo(args)
