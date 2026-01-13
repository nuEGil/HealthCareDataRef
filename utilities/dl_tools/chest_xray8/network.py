import os
import csv
import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
'''
Use pathches instead of full images --> full xray data at 128x128 likely wont be informative
Implement augmentation policies - paper has cropping and color distortion as it's main 2

argparse for hyper parameter tuning
use torch data set - has next method for this thing
implement timing to measure batch load training time. 

refactor to make a task.py. make it so you can train a classifier or do sim clr. 
'''

class Block(nn.Module):
    def __init__(self, input_shape = [256, 256], n_kerns_in = 64, n_kerns_out=64, cc = 'enc'):
        super().__init__() # this makes sure the stuff in the keras layer init function still runs
        
        # convolutional type - I want to make this so that if i use this block in a decoder section
        # I can change one argument and have a new layer type
        convtype = {'enc' : nn.Conv2d,
                    'dec' : nn.ConvTranspose2d}

        self.conv0 = convtype[cc](n_kerns_in, n_kerns_in, kernel_size =(1, 1), 
                                  stride = 1, groups = 1, padding = 'same', bias = True,)

        self.conv1 = convtype[cc](n_kerns_in, n_kerns_in, kernel_size = (7, 7), 
                                  stride = 1, padding = 'same', bias = True,)
        
        self.act0 = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(input_shape) # for [batch size, chans, h=128,w=128] operated over last 2 dims
        self.conv2 = convtype[cc](n_kerns_in, n_kerns_out, kernel_size = (1, 1), 
                                  stride = 1, padding = 'same', bias = True,)
            
    def forward(self, x):
        # so this is the function call 
        # define an intermediate x_ so we can sum the output later
        x_ = self.conv0(x) # 1x1 convolution - no activation
        x_ = self.conv1(x_) # ksize convolution - actiation
        x_ = x_ + x # summing block 
        x_ = self.act0(x_) # relu activation
        x_ = self.LayerNorm(x_) # layer normalization 
        x_ = self.conv2(x_) # set the number of filters on the way out. - no activation
        return x_

# Multilayer perceptron - as a custom layer -- this is normally 
# the last few layers for your classificaiton model
class MLP(nn.Module):
    def __init__(self,  neurons_in = 64, dropout_rate = 0.5, n_out = 1, activation='sigmoid'):
        super(MLP, self).__init__() # make sure the nn.module init funciton works
        activation_types = {'sigmoid': nn.Sigmoid(),
                            'softmax': nn.Softmax(dim = 1)}  
        
        self.neurs = [(neurons_in, neurons_in), 
                      (neurons_in,neurons_in*2),
                      (neurons_in*2, neurons_in)]
        
        # make a module list for these things -- dont have to spplit it up like this
        self.layers = nn.ModuleList() 
        self.dropout_lays = nn.ModuleList()
        
        for neur in self.neurs:
            # dense layers are also refered to as fully connected layers
            self.layers.append(nn.Linear(neur[0], neur[1]))
            self.dropout_lays.append(nn.Dropout(p=dropout_rate, inplace=False))

        # shape of object yo be normalized
        self.LayerNorm = nn.LayerNorm(neurons_in)
        
        # final layer
        self.final_lin = nn.Linear(neurons_in, n_out)
        self.output_activation = activation_types[activation]
    
    def forward(self, x):
        # applies the layers to x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.dropout_lays[i](x)
        x = self.LayerNorm(x)
        x = self.final_lin(x)
        x = self.output_activation(x)
        return x

# Network backbone is just a block stack
class BlockStack(nn.Module):
    # ok so this part works.. just need to keep flattening it out. 
    def __init__(self, input_shape = [3, 128,128], nblocks = 4, n_kerns = 32, 
                 pool_rate = 2, n_out=3, last_activation = 'sigmoid'):
        
        super(BlockStack, self).__init__() # make sure the nn.module init funciton works
        
        # create a module list
        self.layers = nn.ModuleList() 
        # remember its chanells in, channels out, then the rest
        self.layers.append(nn.Conv2d(3, n_kerns, kernel_size = (1, 1), 
                          stride = 1, padding = 'same', bias = True,))

        num_pools = int((nblocks / pool_rate) - 1) # number of pools
        
        # mod to bring back width parameter later. 
        # itterate through the blocks
        input_shape = [n_kerns, *input_shape[1::]]
        for n in range(nblocks):
            self.layers.append(Block(input_shape = input_shape, 
                                     n_kerns_in = n_kerns, 
                                     n_kerns_out = n_kerns, cc = 'enc'))
            if n%pool_rate==0:
                self.layers.append(nn.MaxPool2d([2, 2]))
                # halve the input shape
                input_shape = [n_kerns, int(input_shape[1]/2), int(input_shape[2]/2)]
        
        # once we are at the end. we wnat to flatten 
        self.layers.append(nn.Flatten())

        # the shape is going to be 
        flatten_shape = n_kerns * input_shape[1] * input_shape[2]
        # this one reduces the flatten shape to something we can work with
        self.layers.append(nn.Linear(flatten_shape, 64))
        # then this one calls the MLP layer
        self.layers.append(MLP(neurons_in = 64, dropout_rate = 0.5,
                                n_out = n_out, activation=last_activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   

def NTXEntLoss(z, n_samps, device):
    # follow https://arxiv.org/abs/2002.05709
    # set xk including a positive pair of example xj and xk
    tau = 0.5 # temperature param
    cos = nn.CosineSimilarity(dim=0, eps=1e-6) # compute along dim 
    sij = torch.zeros((2*n_samps, 2*n_samps),
        dtype=torch.float32, device=device)
    # print('sij shape ', sij.shape)
    for i in range(2*n_samps):
        for j in range(2*n_samps):
            sij[i, j] = cos(z[i,...], z[j,...]) # reference makes this (features,) not (batch, features)
    
    numer = torch.exp(sij/tau)
    
    mask = (1-torch.eye(2*n_samps, device=device))
    denom = mask * torch.exp(sij/tau)
    denom = denom.sum(dim = 1, keepdim=True)
    
    # lij 
    lij = -torch.log(numer /denom.squeeze())

    loss = 0
    for k in range(1, n_samps+1):
        a = (2*k-1) -1
        b = (2*k) -1
        loss+= (lij[a, b] + lij[b, a])
    loss = loss / (2*n_samps)
    # print('batch loss : ', loss)
    return loss

class DataSampler():
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
            
        xx = xx.to(device)
        return xx
    
    def shuffle(self):
        np.random.shuffle(self.paths)

class ClassifierSampler():
    def __init__(self, device, batch_size = 5):
        dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/patch_sets/mass1')
        csv_name = os.path.join(dir_, 'train_set.csv')
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

        return xx.to(device), yy.to(device)
    
    def shuffle(self):
        np.random.shuffle(self.paths)

@dataclass
class log_file():
    step: list = field(default_factory=list)
    training_loss: list = field(default_factory=list)

def modeltraining(optimizer, loader, model, epochs):
    NN = loader.paths.shape[0] // loader.batch_size
    log_ = log_file()
    odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/model_0')
    tag = '00'
    for steps in range(epochs):
        total_loss = 0
        for sub in range(0, loader.paths.shape[0]-loader.batch_size, loader.batch_size):
            # sample a batch of data 
            X = loader.load_samples(id=sub)
            y_pred = model(X)
            loss = NTXEntLoss(y_pred, n_samps= loader.batch_size, device = device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{steps+1}/{epochs}], Loss: {total_loss/NN:.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        log_.step.append(steps)
        log_.training_loss.append(total_loss/NN)
        
        if steps%5 == 0: 
            torch.save({
                        'epoch': steps,
                        'model_state': model.state_dict(),
                        # 'optimizer_state': optimizer.state_dict(),
                    }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    torch.save({
                'epoch': steps,
                'model_state': model.state_dict(),
                # 'optimizer_state': optimizer.state_dict(),
            }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    return log_, model, optimizer

def classifier_training(optimizer, loader, model, epochs):
    model.train()
    NN = loader.paths.shape[0] // loader.batch_size
    log_ = log_file()
    odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/model_classifier_1')
    tag = '00'
    lossf = nn.CrossEntropyLoss()
    
    for steps in range(epochs):
        total_loss = 0
        loader.shuffle()
        for sub in range(0, loader.paths.shape[0]-loader.batch_size, loader.batch_size):
            # sample a batch of data 
            X, Y_true= loader.load_samples(id=sub)
            y_pred = model(X)
            loss = lossf(y_pred, Y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{steps+1}/{epochs}], Loss: {total_loss/NN:.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        log_.step.append(steps)
        log_.training_loss.append(total_loss/NN)
        
        if steps%5 == 0: 
            torch.save({
                        'epoch': steps,
                        'model_state': model.state_dict(),
                        # 'optimizer_state': optimizer.state_dict(),
                    }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    torch.save({
                'epoch': steps,
                'model_state': model.state_dict(),
                # 'optimizer_state': optimizer.state_dict(),
            }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    return log_, model, optimizer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------ Using device: {device} ------")
    
    # define the model and sent it to the device
    model = BlockStack( input_shape = [3, 128,128], nblocks = 16, n_kerns = 32, 
                        pool_rate = 4, n_out = 2, last_activation='sigmoid')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # print a copy of the model.
    print(model)

    # DSampler = DataSampler(device, batch_size=20)
    # modeltraining(optimizer, DSampler, model, epochs=20)

    CSampler = ClassifierSampler(device, batch_size=50)
    classifier_training(optimizer, CSampler, model, epochs=100)
