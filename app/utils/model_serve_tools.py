import os
import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision.models import resnet50, ResNet50_Weights
from jobs.dl_tools.chest_xray8.training.models import loadResNet50_add_convheadMLP

def get_subimg_inds(img_size=1024, stride=16):
    pad_up = img_size // stride
    padded_shape = (img_size + pad_up, img_size + pad_up)

    inds = []
    for y in range(0, padded_shape[0] // stride, stride):
        for x in range(0, padded_shape[1] // stride, stride):
            inds.append((y, y+img_size, x, x+img_size)) # this already is the offset. 

    return np.array(inds), pad_up

def patch_generator(img, patch_size=128, stride=None):
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

        moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/ResNet50_add_c_headMLP_mass_v0')
        output_dir = os.path.join(moddir, 'outputs') 
        self.output_dir = output_dir
        model_path = os.path.join(moddir, 'mod_tag-mass_v0_steps-80.pt')
        ckpt = torch.load(model_path, map_location=self.device)

        # you need to copy the architecture exactly for this to work. 
        # remember we have the json file for parame setting --> use this when making custom archiectures. 
        self.model, self.transforms = loadResNet50_add_convheadMLP(num_classes=1) # use model arch from training
        # print(self.model)
        
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()

        # this should be the same across models. 
        self.patch_size = 128
        self.c = self.patch_size // 2
        self.thr = 0.8 # set this in the model class init

    def process_img(self, img0, offset_x=0, offset_y=0, batch_size=32):
        # now batching images
        gen = patch_generator(img0, patch_size=self.patch_size)
        out = defaultdict(float)
        
        batch_patches = []
        batch_coords = []
        sigm = torch.nn.Sigmoid()
        with torch.no_grad():
            for patch, y, x in gen:
                patch_ = torch.from_numpy(patch).permute(2,0,1).float()
                patch_ = self.transforms(patch_)
                batch_patches.append(patch_)
                batch_coords.append((y, x))
                
                # Process when batch is full
                if len(batch_patches) == batch_size:
                    batch_tensor = torch.stack(batch_patches).to(self.device)
                    model_pred = self.model(batch_tensor)  # [batch_size, num_classes]
                    scores = sigm(model_pred).cpu().numpy()  # Move to CPU once
                    for batch_idx, (y, x) in enumerate(batch_coords):
                        score = scores[batch_idx]  # Single score per patch
                        if score > self.thr:
                            new_y = int(y + self.c + offset_y)
                            new_x = int(x + self.c + offset_x)
                            out[(new_y, new_x)] += score
                    
                    # Clear batches
                    batch_patches = []
                    batch_coords = []
            
            # Process remaining patches (last incomplete batch)
            if batch_patches:
                batch_tensor = torch.stack(batch_patches).to(self.device)
                model_pred = self.model(batch_tensor)  # [batch_size, num_classes]
                scores = sigm(model_pred).cpu().numpy()  # Move to CPU once
                
                for batch_idx, (y, x) in enumerate(batch_coords):
                    score = scores[batch_idx]  # Single score per patch
                    if score > self.thr:
                        new_y = int(y + self.c + offset_y)
                        new_x = int(x + self.c + offset_x)
                        out[(new_y, new_x)] += score
        
        return out

def make_padded_image(img, padded_shape):
    padimg = np.zeros(padded_shape, dtype=img.dtype)

    H, W = img.shape[:2]
    PH, PW = padded_shape[:2]

    y0 = (PH - H) // 2
    x0 = (PW - W) // 2

    padimg[y0:y0+H, x0:x0+W,:] = img
    return padimg

def mapmaker(padimg0, points_list, patch_size=128):
    H, W = padimg0.shape[:2]
    val_img   = np.zeros((H, W), dtype=np.float32)
    count_img = np.zeros((H, W), dtype=np.float32)
    c = patch_size // 2
    
    for point in points_list:
        y = point["yx"][0]
        x = point["yx"][1]
        v = point["score"]
        
        y0 = max(0, y - c)
        y1 = min(H, y + c)
        x0 = max(0, x - c)
        x1 = min(W, x + c)
        if y0 >= y1 or x0 >= x1:
            continue
        val_img[y0:y1, x0:x1]   += v
        count_img[y0:y1, x0:x1] += 1.0
    
    return val_img * count_img

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

def to_heatmap(outdir,out, name_ = 'heatmap'):
    # # optional: normalize to [0,255] for visualization
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    out = (out * 255).astype(np.uint8)
    print('ouputshape ', out.shape)
    out_pil = Image.fromarray(out, mode="L")
    out_pil.save(os.path.join(outdir,f"{name_}.png"))


    
    