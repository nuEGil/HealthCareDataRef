import os
import csv
import json
import torch
import torch.nn as nn 
import torch.optim as optim 

import argparse
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from torchvision.models import resnet50, ResNet50_Weights

from fineTuneResNet import loadResNet50

def getImagePaths():
    img_dir =  os.environ['CHESTXRAY8_BASE_DIR']
    mass_set = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/Mass_set.csv')
    paths = []
    with open(mass_set, mode='r', newline='', encoding='utf-8') as ff:
        reader = csv.reader(ff)
        next(reader) # skip the header. 
        for row in reader:
            paths.append(os.path.join(img_dir,row[0]))
    return paths

def to_heatmap(outdir,out, name_ = 'heatmap'):
    # # optional: normalize to [0,255] for visualization
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    out = (out * 255).astype(np.uint8)
    print('ouputshape ', out.shape)
    out_pil = Image.fromarray(out, mode="L")
    out_pil.save(os.path.join(outdir,f"{name_}.png"))

def pad_left_top(img, pad_left, pad_top):
    H, W = img.shape
    padded = np.zeros((H + pad_top, W + pad_left), dtype=img.dtype)
    padded[pad_top:pad_top+H, pad_left:pad_left+W] = img
    return padded

def pad_ring(img, pad):
    """
    pad = total shift (e.g. 64)
    pads pad//2 on each side
    """
    p = pad // 2
    return np.pad(img, pad_width=((p, p), (p, p)), mode="reflect")

def patch_generator(img, patch_size=128, stride=None):
    # do it this way so you can sample the patches instead of 
    # saving a list of patches to work on. 
    if stride is None:
        stride = patch_size

    H, W = img.shape[:2] 
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            yield img[i:i+patch_size, j:j+patch_size], i, j

class model_runner():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"------ Using device: {self.device} ------")

        moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/pytorch_resnet_v2')
        output_dir = os.path.join(moddir, 'outputs') 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir


        model_path = os.path.join(moddir, 'mod_tag-v2_steps-95.pt')
        
        ckpt = torch.load(model_path, map_location=self.device)

        # you need to copy the architecture exactly for this to work. 
        # remember we have the json file for parame setting --> use this when making custom archiectures. 
        self.model, self.transforms = loadResNet50(num_classes=2) # use model arch from training
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()
    
    def process_img(self, img0):

        outimg0 = torch.from_numpy(np.zeros([img0.shape[0], img0.shape[1]])).float()
        outimg0 = outimg0.to(self.device)
        print('outimg0 shape', outimg0.shape)
        
        patch_size = 128
        # so now gen is compatible with next -> next(gen)
        gen = patch_generator(img0, patch_size=patch_size) # gonna make a new patch generator every time.
        
        thr = 0.85
        with torch.no_grad():
            for patch,i,j in gen:
                xx = torch.from_numpy(patch).permute(2,0,1).float()
                xx = self.transforms(xx)
                xx = xx.unsqueeze(0).to(self.device)

                y = self.model(xx) # [1, num_classes]
                # print('y_outputshape ', y.shape)
                probs = torch.softmax(y, dim=1)
                score = probs[0, 1]
                
                score = torch.where(score >= thr, score, torch.zeros_like(score))
                outimg0[i:i+patch_size, j:j+patch_size] = score
        
        out = outimg0.detach().cpu().numpy()
        return out

if __name__ =='__main__':
    
    img_paths = getImagePaths()
    # print('img paths ', img_paths)
    img0 = Image.open(img_paths[0]).convert("RGB")
    img0 = np.array(img0) / 255.0
    
    H, W = img0.shape[:2]
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    
    M00 = model_runner()
    out0 = M00.process_img(img0)
    acc += out0
    cnt += 1

    stride = 2
    for pad in range(2, 512 - 128, stride):
        st = pad // 2
        if st >= H // 2 or st >= W // 2:
            break

        cropped = img0[st:-st, st:-st, :]
        out1 = M00.process_img(cropped)
        out1 = pad_ring(out1, pad=2 * st)

        mask = np.zeros((H, W), dtype=np.float32)
        mask[st:-st, st:-st] = 1

        acc += out1 * mask
        cnt += mask

    full_output = acc / np.clip(cnt, 1, None)
    to_heatmap(M00.output_dir, full_output, name_='heatmap')
    to_heatmap(M00.output_dir, img0, name_ = 'sourceimg')
    
    
    # overlay
    # base = Image.fromarray((img0 * 255).astype(np.uint8))
    # overlay = out_pil.resize(base.size)

    # blended = Image.blend(base.convert("RGB"), overlay.convert("RGB"), alpha=0.4)
    # blended.save("overlay.png")