import os 
import csv 
from PIL import Image, ImageDraw

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


def saveSubPatches(outpath, imname, img, bbox, patch_size = 128, stride = 128):
    print('OUT PATH ', outpath)
    
    x,y,w,h = bbox
    # was gonna do the roi but this will be better . 
    W, H = img.size

    base_name = os.path.basename(imname).split('/')[0].split('.')[0]
    
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

            if ix1 < ix2 and iy1 < iy2:
                overlap_A = (ix2 - ix1) * (iy2 - iy1)
                frac_overlap = overlap_A / (w * h)

            if frac_overlap>0:
                lab = 1
                sub_p = 'finding'
            else:
                lab = 0
                sub_p = 'bg'
            
            print(f"area overlap :{overlap_A} frac overlap:{frac_overlap}")
            outname = os.path.join(outpath, f"{sub_p}/{base_name}_x-{xx}_y-{yy}_label-{lab}.png")
            print('OUT NAME ', outname)
            sub.save(outname)

def cropToPatch():
    mass_patch = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/Mass_set.csv')
    
    j = 0
    with open(mass_patch, "r", encoding='utf-8') as ff:
        reader = csv.reader(ff)
        next(reader) # skip the header. 
        
        outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'],f'user_meta_data/patch_sets/mass1')
        if not os.path.exists(outpath):
            os.makedirs(outpath+'/finding')
            os.makedirs(outpath+'/bg')

        for row in reader:
            # box.append(row)
            img_name = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], row[0])
            img = (Image.open(img_name)
                   .convert("RGB"))
            bbox = list(map(float, row[2:6]))
            bbox = list(map(int, bbox))
             
            saveSubPatches(outpath, img_name, img, bbox, patch_size = 128, stride = 128)
            # outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'],f'user_meta_data/patch_sets/patch_{j}.png')
            # img.save(outpath)
            j+=1
            if j>25:
                break


if __name__ == '__main__':
    cropToPatch()   