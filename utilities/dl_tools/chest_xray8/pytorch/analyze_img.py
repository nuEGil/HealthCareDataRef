import os
import csv
import json
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp

import argparse
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from torchvision.models import resnet50, ResNet50_Weights

from fineTuneResNet import loadResNet50
from PIL import ImageFilter



'''
split the loop up and see if you can run with multiple processes since 
pytorch doesnt allocate the entire gpu 

use an inverse hanning window, or down weight as you get closer to the center. 
heat map will bias towards the center of the image. 

add argparse to set threshold on model probability. 

likely you need to go lower in the patch size during training to try to get 
a finer map. 

other thing to try is to stop before the last max pool layer and use that 
to go fully convolutional resnet...  try this first. 
think about network surgery
'''

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

def overlay_heatmap_simple(img, heatmap, alpha=0.4):
    # base image
    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

    # normalize heatmap
    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-8)
    hm_u8 = (hm * 255).astype(np.uint8)

    # put heatmap in red channel
    overlay = np.zeros((*hm_u8.shape, 3), dtype=np.uint8)
    overlay[..., 0] = hm_u8  # red

    overlay = Image.fromarray(overlay).resize(base.size)
    return Image.blend(base, overlay, alpha)

def norm(out):
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    return (out * 255).astype(np.uint8)

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
        print(self.model)
        
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()
    
    def process_img(self, img0):

        outimg0 = torch.from_numpy(np.zeros([img0.shape[0], img0.shape[1]])).float()
        outimg0 = outimg0.to(self.device)
        # print('outimg0 shape', outimg0.shape)
        
        patch_size = 128
        # so now gen is compatible with next -> next(gen)
        gen = patch_generator(img0, patch_size=patch_size) # gonna make a new patch generator every time.
        
        thr = 0.9
        c = patch_size // 2
        with torch.no_grad():
            for patch,i,j in gen:
                xx = torch.from_numpy(patch).permute(2,0,1).float()
                xx = self.transforms(xx)
                xx = xx.unsqueeze(0).to(self.device)

                y = self.model(xx) # [1, num_classes]
                # print('y_outputshape ', y.shape)
                probs = torch.softmax(y, dim=1)
                score = probs[0, 1]
                
                # # hard threshold
                score = torch.where(score >= thr, score, torch.zeros_like(score))
                outimg0[i:i+patch_size, j:j+patch_size] = score
                
                # # soft threshold
                # score = torch.clamp((score - thr) / (1 - thr), 0, 1) # clamps all elements into input range. 
                # outimg0[i:i+patch_size, j:j+patch_size] += score
                
                # center pixel only
                # outimg0[i + c, j + c] = score
        
        out = outimg0.detach().cpu().numpy()
        return out

# serial heat map generation with shrinking crop towards center. 
# ok for serial compute - bad for parallel - imbalanced load. 
def heat_map_from_patches():
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
    to_heatmap(M00.output_dir, np.mean(img0, axis = -1), name_ = 'sourceimg')

# new version - parallelized
def get_subimg_inds(img_size=1024, stride=16):
    pad_up = img_size // stride
    padded_shape = (img_size + pad_up, img_size + pad_up)

    inds = []
    for y in range(0, padded_shape[0] // stride, stride):
        for x in range(0, padded_shape[1] // stride, stride):
            inds.append((y, y+img_size, x, x+img_size))

    return np.array(inds), pad_up

def worker(inds_chunk, padimg):
    # each worker has its own model  .
    M00 = model_runner() 
    acc = np.zeros(padimg.shape[:2], dtype=np.float32)
    mask = np.zeros(padimg.shape[:2], dtype=np.float32)

    for (y0, y1, x0, x1) in inds_chunk:
        view = padimg[y0:y1, x0:x1, :]
        out = M00.process_img(view)

        acc[y0:y1, x0:x1] += out
        mask[y0:y1, x0:x1] += (out > 0)
    return acc, mask

def make_padded_image(img, padded_shape):
    padimg = np.zeros(padded_shape, dtype=img.dtype)

    H, W = img.shape[:2]
    PH, PW = padded_shape[:2]

    y0 = (PH - H) // 2
    x0 = (PW - W) // 2

    padimg[y0:y0+H, x0:x0+W,:] = img
    return padimg

def crop_center_region(padded_map, original_shape):
    """
    padded_map: HxW or HxWxC
    original_shape: (H, W) or img.shape
    """
    H, W = original_shape[:2]
    PH, PW = padded_map.shape[:2]

    y0 = (PH - H) // 2
    x0 = (PW - W) // 2

    return padded_map[y0:y0+H, x0:x0+W]

def heatmap_gen_dist():
    img_paths = getImagePaths()
    # print('img paths ', img_paths)
    img0 = Image.open(img_paths[0]).convert("RGB")
    img0 = np.array(img0) / 255.0
    
    img_size = img0.shape[0]
    inds, pad_up  = get_subimg_inds(img_size = img_size, stride = 16)

    print('number of large image reps, bbox ',inds.shape)
    # this one already distributes the chunks pretty well
    
    processes = 12
    chunks = np.array_split(inds, processes)
    padimg0 = make_padded_image(img0, [img_size + pad_up, img_size + pad_up,3])

    
    mp.set_start_method("spawn", force=True)
    with mp.get_context("spawn").Pool(processes) as pool:
        results = pool.starmap(worker, 
                               [(chunks[i], padimg0) for i in range(processes)]
        )
    

    acc_final  = np.zeros(padimg0.shape[:2], dtype=np.float32)
    mask_final = np.zeros(padimg0.shape[:2], dtype=np.float32)

    # aggregate results
    for acc, mask in results:
        acc_final  += acc*mask # bias towards pixels that were counted more often. 
        mask_final += mask

    # heatmap = acc_final /  np.clip(mask_final, 1, None)
    heatmap = acc_final / np.maximum(mask_final, 1)
    # in the fast api server you should just pass the user the single pixel locations
    # then blur and render on the ui side. 

    
    # # use only for single pixel mapping. - bluring at the display stage is meaningful - its - 
    # # what are you reporting the the clinician that looks at this thing. - probably want to 
    # # implement a feature where you get the value as you hover. above the pixel. 
    # heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    # heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=2)) # radius could be attached to a slider... interesting. 
    # heatmap = np.array(heatmap)
    # heatmap = norm(heatmap)
    
    moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/pytorch_resnet_v2')
    output_dir = os.path.join(moddir, 'outputs') 
    to_heatmap(output_dir, heatmap, name_ = 'torch_mp_heatmap')
    overlay_ = overlay_heatmap_simple(img0, heatmap, alpha=0.5)
    overlay_.save(os.path.join(output_dir, f"torch_mp_heatmap_overlay.png"))

if __name__ =='__main__':
    heatmap_gen_dist()
    
    
    
    