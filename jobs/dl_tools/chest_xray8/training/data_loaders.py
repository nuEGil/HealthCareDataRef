import os
import csv
import torch 
import numpy as np 
from PIL import Image

class simCLRLoader():
    def __init__(self, device, batch_size = 5):
        dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data')
        csv_name = os.path.join(dir_, 'set1.csv')
        self.batch_size = batch_size
        self.target_size = (128, 128)  # (W, H) for PIL
        self.device = device
        paths = []
        with open(csv_name, mode='r', newline='', encoding='utf-8') as ff:
            reader = csv.reader(ff)
            next(reader) # skip the header. 
            for row in reader:
                paths.append(os.path.join(os.environ['CHESTXRAY8_BASE_DIR'],row[0]))

        self.paths = np.array(paths) # works on strings. -- can shuffle in place now. 
        

    def load_samples(self, id=0):
        xx = torch.zeros((2*self.batch_size,3,128,128), dtype = torch.float32)
        for ii in range(0, self.batch_size):
            img = Image.open(self.paths[id+ii]).convert('RGB').resize(self.target_size, Image.BILINEAR)
            img = np.array(img) / 255.0
            # augmentations right in the loop
            aug1 = img**2
            aug2 = np.sqrt(img)
            xx[2*ii,...] =  torch.from_numpy(aug1).permute(2, 0, 1).float()
            xx[2*ii+1,...] =  torch.from_numpy(aug2).permute(2, 0, 1).float() 
            
        xx = xx.to(self.device)
        return xx
    
    def shuffle(self):
        np.random.shuffle(self.paths)

class ClassifierLoader():
    def __init__(self, device, csv_name, batch_size = 5):
        self.batch_size = batch_size
        self.target_size = (128, 128)  # (W, H) for PIL
        self.device = device
        paths = []
        with open(csv_name, mode='r', newline='', encoding='utf-8') as ff:
            reader = csv.reader(ff)
            next(reader) # skip the header. 
            for row in reader:
                paths.append(row)

        self.paths = np.array(paths) # works on strings. -- can shuffle in place now.   

        self.NN = self.paths.shape[0] // batch_size # need this for the training loop
        print('number of steps per epoch :', self.NN)
                  
    def load_samples(self, id=0):
        xx = torch.zeros((self.batch_size, 3, 128, 128), dtype=torch.float32)
        yy = torch.zeros((self.batch_size,), dtype=torch.long)

        for ii in range(self.batch_size):
            subrow = self.paths[id + ii]

            img = (
                Image.open(subrow[1])
                .convert("RGB")
                .resize(self.target_size, Image.BILINEAR)
            )
            img = np.array(img) / 255.0

            lab = int(subrow[2])   # 0 or 1

            xx[ii] = torch.from_numpy(img).permute(2, 0, 1)
            yy[ii] = lab

        return xx.to(self.device), yy.to(self.device)
    
    def shuffle(self):
        np.random.shuffle(self.paths)