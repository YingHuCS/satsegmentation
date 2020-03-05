import os
import sys
import argparse
import collections
from contextlib import contextmanager
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision.transforms import  Normalize
from torch.optim import lr_scheduler

cwd = os.getcwd()
sys.path.append(cwd)
import models
from datasets.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomVerticalFlip,
    JointRandomResize,
    JointRandomCrop,
    JointRandomRotate,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor)
from datasets.dataset import TrainDataset
from core.metrics import Metrics
from core.losses import (
    CrossEntropyLoss2d, 
    mIoULoss2d, 
    FocalLoss2d, 
    LovaszLoss2d, 
    DiceLoss, 
    MixedLovaszCrossEntropyLoss)
from utils.log import Log
from utils.plot import plot

from mmcv import Config


@contextmanager
def no_grad():
    with torch.no_grad():
        yield


def parse_args():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--config', type=str, help='config file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = torch.device('cuda' if cfg.use_gpu else 'cpu')

    if cfg.use_gpu and not torch.cuda.is_available():
        sys.exit('Error: CUDA requested but not available')

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    assert cfg.model.num_classes == len(cfg.data.classes)
    num_classes = cfg.model.num_classes
    
    model_cfg = cfg.model.copy()
    model_name = model_cfg.pop('name')
    net = getattr(models, model_name)
    net = net(**model_cfg)
    net = DataParallel(net)
    net = net.to(device)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    try:
        weight = torch.Tensor(cfg.data.weights)
    except KeyError:
        if cfg.loss in ('CrossEntropy', 'mIoU', 'Focal'):
            sys.exit('Error: The loss function used, need dataset weights values')
    
    optimizer_cfg = cfg.optimizer.copy()
    optimizer_name = optimizer_cfg.pop('name')
    if optimizer_name == 'Adam':
        optimizer_cfg.pop('momentum')
    optimizer = getattr(optim, optimizer_name)
    optimizer = optimizer(net.parameters(), **optimizer_cfg)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, **cfg.scheduler)

    resume = 0
    if cfg.checkpoint:

        def map_location(storage, _):
            return storage.cuda() if cfg.use_gpu else storage.cpu()

        # https://github.com/pytorch/pytorch/issues/7178
        chkpt = torch.load(cfg.checkpoint, map_location=map_location)
        net.load_state_dict(chkpt['state_dict'])

        if cfg.resume:
            optimizer.load_state_dict(chkpt['optimizer'])
            resume = chkpt['epoch']

    if cfg.loss == 'CrossEntropy':
        criterion = CrossEntropyLoss2d(weight=weight).to(device)
    elif cfg.loss == 'mIoU':
        criterion = mIoULoss2d(weight=weight).to(device)
    elif cfg.loss == 'Focal':
        criterion = FocalLoss2d(weight=weight).to(device)
    elif cfg.loss == 'Lovasz':
        criterion = LovaszLoss2d().to(device)
    elif cfg.loss == 'Dice':
        criterion = DiceLoss().to(device)
    elif cfg.loss == 'Mix':
        criterion = MixedLovaszCrossEntropyLoss(weight=weight).to(device)
    else:
        sys.exit('Error: Unknown cfg.loss value !')
    
    train_loader, val_loader = get_dataset_loaders(cfg)

    num_epochs = cfg.num_epochs
    if resume >= num_epochs:
        sys.exit('Error: Epoch {} already reached by the checkpoint provided'.format(num_epochs))

    history = collections.defaultdict(list)
    log = Log(os.path.join(cfg.checkpoint_dir, 'log'))

    log.log('--- Hyper Parameters on this training: ---')
    log.log('Model:\t\t {}'.format(model_name))
    log.log('Backbone:\t {}'.format(cfg.model.backbone_name))
    log.log('Pretrained:\t {}'.format(cfg.model.pretrained))
    log.log('Loss function:\t {}'.format(cfg.loss))
    log.log('Batch Size:\t {}'.format(cfg.batch_size))
    log.log('optimizer:\t {}'.format(optimizer_name))
    log.log('Learning Rate:\t {}'.format(cfg.optimizer.lr))
    log.log('Momentum:\t {}'.format(cfg.optimizer.momentum))
    log.log('Weight Decay:\t {}'.format(cfg.optimizer.weight_decay))
    log.log('Step size:\t {}'.format(cfg.scheduler.step_size))
    log.log('Gamma:\t\t {}'.format(cfg.scheduler.gamma))
    log.log('Image Size:\t {}'.format(cfg.data.train.crop_size))
    log.log('Resize Scale:\t {}'.format(cfg.data.train.resize_scale))
    log.log('Flip Probability:\t {}'.format(cfg.data.train.flip_prob))
    log.log('Rotation Probability:\t {}'.format(cfg.data.train.rotation_prob))
    log.log('Rotation Degree:\t {}'.format(cfg.data.train.rotation_degree))
    log.log('Rotate Degree:\t {}'.format(cfg.data.train.rotate_degree))
    
    if 'weight' in locals():
        log.log('Weights:\t {}'.format(cfg.data.weights))
    log.log('------------------------------------------')

    for epoch in range(resume, num_epochs):
        log.log('Epoch: {}/{}'.format(epoch + 1, num_epochs))
        
        train_hist = train(train_loader, num_classes, device, net, optimizer, criterion, exp_lr_scheduler)
        log.log(
            'Train    loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}'.format(
                train_hist['loss'],
                train_hist['miou'],
                cfg.data.classes[1],
                train_hist['fg_iou'],
                train_hist['mcc'],
            )
        )

        for k, v in train_hist.items():
            history['train ' + k].append(v)

        val_hist = validate(val_loader, num_classes, device, net, criterion)
        log.log(
            'Validate loss: {:.4f}, mIoU: {:.3f}, {} IoU: {:.3f}, MCC: {:.3f}'.format(
                val_hist['loss'], val_hist['miou'], cfg.data.classes[1], val_hist['fg_iou'], val_hist['mcc']
            )
        )

        for k, v in val_hist.items():
            history['val ' + k].append(v)

        visual = 'history-{:05d}-of-{:05d}.png'.format(epoch + 1, num_epochs)
        plot(os.path.join(cfg.checkpoint_dir, visual), history)

        checkpoint = 'checkpoint-{:05d}-of-{:05d}.pth'.format(epoch + 1, num_epochs)

        states = {'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}

        torch.save(states, os.path.join(cfg.checkpoint_dir, checkpoint))


def train(loader, num_classes, device, net, optimizer, criterion, scheduler):
    num_samples = 0
    running_loss = 0

    metrics = Metrics()

    scheduler.step()
    print('current lr: ', scheduler.get_lr())
    net.train()
    
    for images, masks in tqdm(loader, desc='Train', unit='batch', ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], 'resolutions for images and masks are in sync'

        num_samples += int(images.size(0))

        optimizer.zero_grad()
        outputs = net(images)
        assert outputs.size()[2:] == masks.size()[1:], 'resolutions for predictions and masks are in sync'
        assert outputs.size()[1] == num_classes, 'classes for predictions and dataset are in sync'

        loss = criterion(outputs, masks)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    assert num_samples > 0, 'dataset contains training images and labels'

    return {
        'loss': running_loss / num_samples,
        'miou': metrics.get_miou(),
        'fg_iou': metrics.get_fg_iou(),
        'mcc': metrics.get_mcc(),
    }


@no_grad()
def validate(loader, num_classes, device, net, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics()

    net.eval()

    for images, masks in tqdm(loader, desc='Validate', unit='batch', ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], 'resolutions for images and masks are in sync'

        num_samples += int(images.size(0))

        outputs = net(images)

        assert outputs.size()[2:] == masks.size()[1:], 'resolutions for predictions and masks are in sync'
        assert outputs.size()[1] == num_classes, 'classes for predictions and dataset are in sync'

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

    assert num_samples > 0, 'dataset contains validation images and labels'

    return {
        'loss': running_loss / num_samples,
        'miou': metrics.get_miou(),
        'fg_iou': metrics.get_fg_iou(),
        'mcc': metrics.get_mcc(),
    }


def get_dataset_loaders(cfg):
    resize_scale = cfg.data.train.resize_scale
    crop_size = cfg.data.train.crop_size
    flip_prob = cfg.data.train.flip_prob
    rotation_prob = cfg.data.train.rotation_prob
    rotate_degree = cfg.data.train.rotate_degree
    rotation_degree=cfg.data.train.rotation_degree
    train_img_mean = cfg.data.train.img_mean
    train_img_std = cfg.data.train.img_std
    val_img_mean = cfg.data.val.img_mean
    val_img_std = cfg.data.val.img_std


    train_transform = JointCompose(
        [
            JointTransform(ConvertImageMode('RGB'), ConvertImageMode('P')),
            #JointRandomResize(resize_scale),
            #JointRandomCrop(crop_size),
            JointRandomHorizontalFlip(flip_prob),
            JointRandomVerticalFlip(flip_prob),
            #JointRandomRotate(rotate_degree),
            JointRandomRotation(rotation_prob, rotation_degree),
            JointRandomRotation(rotation_prob, rotation_degree),
            JointRandomRotation(rotation_prob, rotation_degree),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=train_img_mean, std=train_img_std), None),
        ]
    )

    val_transform = JointCompose(
        [
            JointTransform(ConvertImageMode('RGB'), ConvertImageMode('P')),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=val_img_mean, std=val_img_std), None),
        ]
    )


    train_img_dir = cfg.data.train.img_dir
    train_mask_dir = cfg.data.train.mask_dir
    val_img_dir = cfg.data.val.img_dir
    val_mask_dir = cfg.data.val.mask_dir

    train_dataset = TrainDataset(train_img_dir, train_mask_dir, train_transform)
    val_dataset = TrainDataset(val_img_dir, val_mask_dir, val_transform) 

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=cfg.num_workers)

    return train_loader, val_loader

if __name__ == '__main__':
    main()
