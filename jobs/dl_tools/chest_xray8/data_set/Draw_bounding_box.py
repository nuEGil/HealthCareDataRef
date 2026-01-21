import os 
import pandas as pd
from PIL import Image, ImageDraw
from jobs.dl_tools.chest_xray8.data_set.Patch_maker import read_a_few

def draw_bounding_boxes(dat_, tag = 0):
    samples = 20
    sub_dat = dat_.iloc[0:samples,:]

    for ii in range(samples):
        row = sub_dat.iloc[ii]
        fname_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], row["path"])
        base_name = os.path.basename(fname_)
        img = Image.open(fname_).convert("RGB")

        if tag != 0:
            bbox_img = img.copy()
            x, y, w, h = map(int, [row["x"], row["y"], row["w"], row["h"]])
            draw = ImageDraw.Draw(bbox_img)
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
            
            bb_name = base_name.replace('.png', f'_bb_tag_{tag}.png')
            bb_outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/patch_sets/bbox_check_1/{bb_name}')
            bbox_img.save(bb_outpath)
            
        oname = base_name.replace('.png', f'_o_tag_{tag}.png')
        outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/patch_sets/bbox_check_1/{oname}')
        img.save(outpath)
        

if __name__ =='__main__':
    NoF_data = read_a_few(tag = 0)
    Mass_data = read_a_few(tag = 3)

    draw_bounding_boxes(NoF_data, tag = 0)
    draw_bounding_boxes(Mass_data, tag = 3)
    