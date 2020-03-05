import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

cwd = os.getcwd()
sys.path.append(cwd)
import models
from datasets.dataset import TestDataset
from datasets.transforms import ConvertImageMode, ImageToTensor
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='inference model')
    parser.add_argument('--config', type=str, help='config file path')
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = torch.device("cuda" if cfg.use_gpu else "cpu")

    def map_location(storage, _):
        return storage.cuda() if cfg.use_gpu else storage.cpu()



    if cfg.use_gpu and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    num_classes = len(cfg.data.classes)

    # https://github.com/pytorch/pytorch/issues/7178

    chkpt = torch.load(cfg.checkpoint, map_location=map_location)
    model_cfg = cfg.model
    model_name = model_cfg.pop('name')
    net = getattr(models, model_name)
    net = net(**model_cfg)
    net = DataParallel(net)
    net = net.to(device)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    '''
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for key in chkpt["state_dict"].keys():
        new_key = key.replace('resnet', 'backbone')
        new_checkpoint[new_key] = chkpt["state_dict"][key]
    '''

    net.load_state_dict(chkpt["state_dict"])
    #net.load_state_dict(new_checkpoint)
    net.eval()

    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=cfg.data.test.img_mean, std=cfg.data.test.img_std)])
    test_dataset = TestDataset(cfg.data.test.img_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for imgs, img_names in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = net(imgs)

            # manually compute segmentation mask class probabilities per pixel

            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()
            
            for prob, img_name in zip(probs, img_names):
                # Quantize the floating point probabilities in [0,1] to [0,255] and store
                assert prob.shape[0] == 2, "single channel requires binary model"
                assert np.allclose(np.sum(prob, axis=0), 1.), "single channel requires probabilities to sum up to one"
                
                fg = prob[1, :, :]
                anchors = np.linspace(0, 1, 256)
                out = np.digitize(fg, anchors).astype(np.uint8)
                out[out>=128] = 255
                out[out<128] = 0
                      
                out = Image.fromarray(out, mode="P")
                out.save(os.path.join(cfg.data.test.out_dir, img_name.split('.')[0]+'.png'), optimize=True)


if __name__ == '__main__':
    main()

