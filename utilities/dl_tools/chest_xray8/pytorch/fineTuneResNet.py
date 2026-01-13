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
'''
ok resnet with frozen weidths except the last layer fine tunes on these image patches 
great. means the training loop is fine... 

so now we have the option to try a few different training set ups. patches vs full image
vs. n classes. etc. 

finetuning with supervised learning -- you are restricted to bg+ the labeled classes in the data set 
finetuning with NTXEnt and SimCLR -- you can pick the number of features in the final layer ... 

practically - the output for both is still a vector. for segmentation you need to do post processing
for full image classification you get one number out. 

Remember - goal is a model server that does xyz to the image. 

difference between tf and pytorch in memory allocation. I can run 
multiple instances of this training script on the same GPU
TF preallocates all the available gpu memory. Pytorch only allocates what you need. 
need to look to see if there's any memory leak issues from training like this. 
need to look into why TF allocates the entire GPU... could be that it has some
optimizations. for loading data as quickly as possible.  

need augmentation policies. 

'''
def manageArgs():
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument("--tag", type=str, required=True, help="model tag")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")

    args = parser.parse_args()
    assert args.learning_rate > 0, "learning_rate must be > 0"
    return args

class ClassifierSampler():
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

@dataclass
class log_file():
    training_step: list = field(default_factory=list)
    training_loss: list = field(default_factory=list)
    
    testing_step: list = field(default_factory=list)
    testing_loss: list = field(default_factory=list)


    # should really consolidate these into 1 function. 
    def save_training_log(self, fname_full):
        with open(fname_full, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "training_loss"])  # header

            for s, l in zip(self.training_step, self.training_loss):
                writer.writerow([s, l])

    def save_testing_log(self, fname_full):
        with open(fname_full, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "testing_loss"])  # header

            for s, l in zip(self.testing_step, self.testing_loss):
                writer.writerow([s, l])

def loadResNet50(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # replace final FC
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    # freeze backbone (optional but recommended for sanity check)
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    return model, weights.transforms()

class Trainer():
    def __init__(self, batch_size = 50, learning_rate = 1e-3, epochs = 100, tag = 'v0' ):
        self.tag = tag
        # set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"------ Using device: {self.device} ------")
        
        # output directory 
        self.odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/pytorch_resnet_{tag}')
        if not os.path.exists(self.odir):
            os.makedirs(self.odir)
        
        with open(os.path.join(self.odir, "args.json"), "w") as f:
            json.dump({'batch_size':batch_size, 'learning_rate':learning_rate,
                       'epochs':epochs, 'tag':tag}, f, indent=2)

        # load training and testing data files
        dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/patch_sets/mass1')
        csv_name_train = os.path.join(dir_, 'train_set.csv')
        csv_name_test = os.path.join(dir_, 'test_set.csv')

        # intanciate the sampler classes
        self.batch_size = batch_size # arg
        self.train_sampler = ClassifierSampler(self.device, csv_name_train, batch_size=batch_size)
        self.test_sampler = ClassifierSampler(self.device, csv_name_test, batch_size=batch_size)

        # load the model
        model, self.preprocess = loadResNet50(2)
        model = model.to(self.device)
        model.train()
        # print a copy of the model.
        print(model)
        self.model = model
        self.epochs = epochs # arg

        # optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.lossf = nn.CrossEntropyLoss()

        # log file and tags. 
        self.log_file_ = log_file()
        
    def train_one_epoch(self, steps):
        total_loss = 0
        self.train_sampler.shuffle()
        NN = self.train_sampler.NN
         
        for sub in range(0, self.train_sampler.paths.shape[0]-self.batch_size, self.batch_size):
            # sample a batch of data 
            X, Y_true = self.train_sampler.load_samples(id=sub)
            X = self.preprocess(X) # preprocess should happen in the dataloader  
            y_pred = self.model(X)
            loss = self.lossf(y_pred, Y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{steps+1}/{self.epochs}], Train Loss: {total_loss/NN:.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        self.log_file_.training_step.append(steps)
        self.log_file_.training_loss.append(total_loss/NN)

    def test_model(self, steps):
        print('Testing...')
        total_loss = 0
        NN = self.test_sampler.NN
         
        for sub in range(0, self.test_sampler.paths.shape[0] - self.batch_size, self.batch_size):
            # sample a batch of data 
            X, Y_true = self.test_sampler.load_samples(id=sub)
            X = self.preprocess(X) # preprocess should happen in the dataloader  
            y_pred = self.model(X)
            loss = self.lossf(y_pred, Y_true)
            total_loss += loss.item()
            # no gradient step or anything like that. 

        print(f"Epoch [{steps+1}/{self.epochs}], Test Loss: {total_loss/NN:.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        self.log_file_.testing_step.append(steps)
        self.log_file_.testing_loss.append(total_loss/NN)   
        print('saving model and logs')

        torch.save({
                    'epoch': steps,
                    'model_state': self.model.state_dict(),
                    #'optimizer_state': optimizer.state_dict(), # needed to continue training. 
                    }, f'{self.odir}/mod_tag-{self.tag}_steps-{steps}.pt')
        
        self.log_file_.save_training_log(os.path.join(self.odir, 'trainlog.csv'))
        self.log_file_.save_testing_log(os.path.join(self.odir, 'testlog.csv'))
        print('artifacts saved')

    def classifier_training(self):

        for steps in range(self.epochs):
            self.train_one_epoch(steps)
            
            if steps%5 == 0: 
                self.test_model(steps)

if __name__ == '__main__':
    xargs = manageArgs()
    mod_train = Trainer(batch_size = xargs.batch_size, 
                        learning_rate = xargs.learning_rate,
                        epochs = xargs.epochs, tag = xargs.tag )
    mod_train.classifier_training()