import os 
import sys
import csv
import glob 
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw
'''
probably there's a cleaner way to write this
rewrite this with ... well it's file i/o bound, likely threading.. 
propper way to do this is to do it with threading. but sprint... bash s
'''

def draw_bounding_boxes():
    mass_patch = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/Mass_set.csv')
    box = []
    j = 0
    with open(mass_patch, "r", encoding='utf-8') as ff:
        reader = csv.reader(ff)
        next(reader) # skip the header. 
        for row in reader:
            # box.append(row)
            img_name = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], row[0])
            img = (
                    Image.open(img_name)
                    .convert("RGB")
                )


            bbox = list(map(float, row[2:6]))
            bbox = list(map(int, bbox))
            x,y,w,h = bbox
            draw = ImageDraw.Draw(img)
            # bbox coords
            x, y, w, h = bbox
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="green",
                width=3
                    )
            
            outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'],f'user_meta_data/patch_sets/patch_{j}.png')
            img.save(outpath)
            j+=1
            if j>25:
                break

def read_a_few(tag = 0):
    np.random.seed(42)

    odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/')
    fnames = ["No Finding_set.csv", "Effusion_set.csv", 
              "Infiltrate_set.csv", "Mass_set.csv"]
    # not the right way to do this but 
    ff = fnames[tag]
    ff_ = os.path.join(odir, ff)
    dat = pd.read_csv(ff_)
    if tag ==0:
        dat["x"] = 0.0
        dat["y"] = 0.0
        dat["w"] = 1024.0
        dat["h"] = 1024.0
    dat = dat.sample(n=85) # smallest set has 85 images. 
    print(dat.head())    
    print(dat.shape)
    return dat

def saveSubPatches(outpath, imname, img, bbox, patch_size = 128, stride = 128, tag = 0):
    # print('OUT PATH ', outpath)
    
    x,y,w,h = bbox
    # was gonna do the roi but this will be better . 
    W, H = img.size

    base_name = os.path.basename(imname).split('/')[0].split('.')[0]
    # print(base_name, x,y,w,h)
    for yy in range(0, H - patch_size + 1, stride):
        for xx in range(0, W - patch_size + 1, stride):
            frac_overlap = 0.0
            overlap_A = 0.0
            sub = img.crop((
                xx, yy,
                xx + patch_size, yy + patch_size
            ))
            
            # intersection box
            ix1 = max(xx, x)
            iy1 = max(yy, y)
            ix2 = min(xx + patch_size, x + w)
            iy2 = min(yy + patch_size, y + h)

            if ix1 <= ix2 and iy1 <= iy2:
                overlap_A = (ix2 - ix1) * (iy2 - iy1)
                frac_overlap = overlap_A / (w * h)

            if frac_overlap>0.25 or w>1023:
                lab = tag
                       
                print(f"area overlap :{overlap_A} frac overlap:{frac_overlap}")
                outname = os.path.join(outpath, f"{base_name}_x-{xx}_y-{yy}_label-{lab}.png")
                print('OUT NAME ', outname, 'stride ', stride)
                sub.save(outname)

def cropToPatch(dat_, tag = 0):
        
    outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/patch_sets/NoF_Eff_Inf_Mas/tag_{tag}')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    for j in range(dat_.shape[0]):
        row = dat_.iloc[j, :]
    
        img_name = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], row.iloc[0])
        img = (Image.open(img_name).convert("RGB"))

        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])
        bbox = [x, y, w, h]
        bbox = list(map(int, bbox))
        
        # over sample the positive classes; they're less common than the background
        stride_ = 128 if tag==0 else 32    
        saveSubPatches(outpath, img_name, img, bbox, patch_size = 128, stride = stride_, tag = tag)

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--i", type=int, required=True)
    args = p.parse_args()

    print("Running job", args.i)
    
    tag = args.i
    data_ = read_a_few(tag = tag)
    cropToPatch(data_, tag = tag)  