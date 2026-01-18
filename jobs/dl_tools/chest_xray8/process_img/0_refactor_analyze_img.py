import os
import csv
import time
import numpy as np 

import torch
import torch.multiprocessing as mp
from collections import defaultdict

from PIL import Image
from jobs.dl_tools.chest_xray8.training.models import loadResNet50_add_convhead


'''
second refactor --> since we trained a new model that has a conv step 
the output from that layer on ResNet is gonna end up being 8x8 with 1 channel . 
BCE loss so that 1 channel activity maps directly to the class. 

this algorithm already pads, then picks overlaping 1024x1024 regions. 
so what we need to do. is use less of those. --> before we only got 1 score - center pixel 
8x8 means we can pick more pixels from the resulting map to add to the point list
ie. we build the point list faster --> less 1024x1024 regions needed. 


python -m jobs.dl_tools.chest_xray8.process_img.0_refactor_analyze_img
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

def get_subimg_inds(img_size=1024, stride=16):
    pad_up = img_size // stride
    padded_shape = (img_size + pad_up, img_size + pad_up)

    inds = []
    for y in range(0, padded_shape[0] // stride, stride):
        for x in range(0, padded_shape[1] // stride, stride):
            inds.append((y, y+img_size, x, x+img_size)) # this already is the offset. 

    return np.array(inds), pad_up

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

        moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/ResNet50_add_c_head_mass_v0')
        
        output_dir = os.path.join(moddir, 'outputs') 
        
        self.output_dir = output_dir

        model_path = os.path.join(moddir, 'mod_tag-mass_v0_steps-40.pt')
        
        ckpt = torch.load(model_path, map_location=self.device)

        # you need to copy the architecture exactly for this to work. 
        # remember we have the json file for parame setting --> use this when making custom archiectures. 
        self.model, self.transforms = loadResNet50_add_convhead(num_classes=1) # use model arch from training
        print(self.model)
        
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()

        # this should be the same across models. 
        self.patch_size = 128
        self.c = self.patch_size // 2
        self.thr = 0.95 # set this in the model class init

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
                    probs = sigm(model_pred)
                    # print('=-------Probs shape ', probs.shape)
                    scores = probs.cpu().numpy()
                    
                    for (y, x), score in zip(batch_coords, scores):
                        score = score if score > self.thr else 0.0
                        if score > 0:
                            new_y = int(y + self.c + offset_y)
                            new_x = int(x + self.c + offset_x)
                            out[(new_y, new_x)] += score
                    
                    # Clear batches
                    batch_patches = []
                    batch_coords = []
            
            # Process remaining patches (last incomplete batch)
            if batch_patches:
                batch_tensor = torch.stack(batch_patches).to(self.device)
                model_pred = self.model(batch_tensor)
                probs = sigm(model_pred)
                # print('=-------Probs shape ', probs.shape)
                scores = probs.cpu().numpy()
                
                for (y, x), score in zip(batch_coords, scores):
                    score = score if score > self.thr else 0.0
                    if score > 0:
                        new_y = int(y + self.c + offset_y)
                        new_x = int(x + self.c + offset_x)
                        out[(new_y, new_x)] += score
        
        return out

class model_gridRun():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"------ Using device: {self.device} ------")

        moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/ResNet50_add_c_head_mass_v0')
        
        output_dir = os.path.join(moddir, 'outputs') 
        
        self.output_dir = output_dir

        model_path = os.path.join(moddir, 'mod_tag-mass_v0_steps-40.pt')
        
        ckpt = torch.load(model_path, map_location=self.device)

        # you need to copy the architecture exactly for this to work. 
        # remember we have the json file for parame setting --> use this when making custom archiectures. 
        self.model, self.transforms = loadResNet50_add_convhead(num_classes=1) # use model arch from training
        print(self.model)
        self.model.deploy_ =True
        
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()

        # this should be the same across models. 
        self.patch_size = 128
        self.c = self.patch_size // 2
        self.thr = 0.9 # set this in the model class init

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

                    # can get 
                    model_pred, spatial_pred = self.model(batch_tensor)  # [batch_size, num_classes]
                    
                    spatial_probs = sigm(spatial_pred)[:,0,:,:] # should get [batch,8x8]
                    print('=-------Probs shape ', spatial_probs.shape)
                    
                    scores = spatial_probs.cpu().numpy()
                    # Find all points above threshold
                    high_conf_points = torch.where(spatial_probs > self.thr)
                    # print('high conf points ', high_conf_points.shape , high_conf_points)
                    # Add each point to your output dict
                    for batch_idx, point_y, point_x in zip(high_conf_points[0], high_conf_points[1], high_conf_points[2]):
                        # batch_idx tells you which patch in the batch this point came from
                        y, x = batch_coords[batch_idx]  # Get the correct patch's coordinates
                        
                        # Convert local patch coordinates to global image coordinates
                        global_y = int(y + point_y.item() + offset_y)
                        global_x = int(x + point_x.item() + offset_x)
                        
                        # Get the actual score from the spatial map
                        score = spatial_probs[batch_idx, point_y, point_x].item()
                        out[(global_y, global_x)] += score
                    
                    # Clear batches
                    batch_patches = []
                    batch_coords = []
            
            # Process remaining patches (last incomplete batch)
            if batch_patches:
                batch_tensor = torch.stack(batch_patches).to(self.device)
                # can get 
                model_pred, spatial_pred = self.model(batch_tensor)  # [batch_size, num_classes]
                
                spatial_probs = sigm(spatial_pred)[:,0,:,:] # should get [batch,8x8]
                print('=-------Probs shape ', spatial_probs.shape)
                
                scores = spatial_probs.cpu().numpy()
                # Find all points above threshold
                high_conf_points = torch.where(spatial_probs > self.thr)
                # print('high conf points ', high_conf_points.shape , high_conf_points)
                # Add each point to your output dict
                for batch_idx, point_y, point_x in zip(high_conf_points[0], high_conf_points[1], high_conf_points[2]):
                    # batch_idx tells you which patch in the batch this point came from
                    y, x = batch_coords[batch_idx]  # Get the correct patch's coordinates
                    
                    # Convert local patch coordinates to global image coordinates
                    global_y = int(y + point_y.item() + offset_y)
                    global_x = int(x + point_x.item() + offset_x)
                    
                    # Get the actual score from the spatial map
                    score = spatial_probs[batch_idx, point_y, point_x].item()
                    out[(global_y, global_x)] += score
                
        return out

def worker(inds_chunk, padimg):
    # each worker has its own model  .
    M00 = model_runner()
    # M00 = model_gridRun() 
    
    all_out = defaultdict(float)
    start_time = time.time()
    for (y0, y1, x0, x1) in inds_chunk:
        view = padimg[y0:y1, x0:x1, :]
        out = M00.process_img(view, offset_y=y0, offset_x=x0)

        # accumulate score.     
        for k_,v_ in out.items():
            all_out[k_]+=v_
    elapsed_time = time.time() - start_time
    print(f'Inference took {elapsed_time:.3f} seconds')
    return all_out

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

def heatmap_gen_dist():
    moddir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/ResNet50_add_c_head_mass_v0')
    output_dir = os.path.join(moddir, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_paths = getImagePaths()
    # print('img paths ', img_paths)
    img0 = Image.open(img_paths[0]).convert("RGB")
    img0 = np.array(img0) / 255.0
    
    img_size = img0.shape[0]
    inds, pad_up  = get_subimg_inds(img_size = img_size, stride = 64)

    print('number of large image reps, bbox ',inds.shape)
    # this one already distributes the chunks pretty well
    
    processes = 2
    chunks = np.array_split(inds, processes)
    padimg0 = make_padded_image(img0, [img_size + pad_up, img_size + pad_up,3])

    mp.set_start_method("spawn", force=True)
    with mp.get_context("spawn").Pool(processes) as pool:
        results = pool.starmap(worker, [(chunks[i], padimg0) for i in range(processes)])
    
    accumulated = defaultdict(float)
    for rr in results:
        for k_, v_ in rr.items():
            accumulated[k_] += v_

    print('LEN accum ', len(accumulated))
    # Convert tuple keys to lists or strings for JSON serialization
    points_list = [{"yx": list(k), "score": float(v)} for k, v in accumulated.items()]
    
    heatmap = mapmaker(padimg0, points_list, patch_size=128)
    
    to_heatmap(output_dir, heatmap, name_ = 'torch_mp_heatmap')
    overlay_ = overlay_heatmap_simple(img0, heatmap, alpha=0.5)
    overlay_.save(os.path.join(output_dir, f"torch_mp_heatmap_overlay.png"))

if __name__ =='__main__':
    heatmap_gen_dist()
    
    
    
    