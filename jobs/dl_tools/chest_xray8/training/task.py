import os 
import csv
import json
import torch
import torch.nn as nn 
import torch.optim as optim 

import argparse
from dataclasses import dataclass, field

from data_loaders import ClassifierLoader
from models import loadResNet50_unfreeze

def manageArgs():
    parser = argparse.ArgumentParser(description="train a model")
    
    parser.add_argument("--tag", type=str, required=True, help="model tag")
    parser.add_argument("--data_set_dir", type=str, required=True, help="directory with train test csv \
                        - just the name. have path management to patch_set from base image dir")
    parser.add_argument("--n_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")

    args = parser.parse_args()
    assert args.learning_rate > 0, "learning_rate must be > 0"
    return args

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

class Trainer():
    def __init__(self, n_classes=2, batch_size = 50, learning_rate = 1e-3, epochs = 100, tag = 'v0', data_set_dir='' ):
        self.tag = tag
        # set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"------ Using device: {self.device} ------")
        
        # output directory 
        self.odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/pytorch_resnet_{tag}')
        if not os.path.exists(self.odir):
            os.makedirs(self.odir)
        
        # check this before running training. I want to make sure that the hyperparameters for the model are saved. 
        params = {'n_classes':n_classes, 'batch_size':batch_size, 
                  'learning_rate':learning_rate, 'epochs':epochs, 
                  'tag':tag, 'data_set_dir':data_set_dir}
        
        with open(os.path.join(self.odir, "args.json"), "w") as f:
            json.dump(params, f, indent=2)

        # load training and testing data files
        dir_ = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/patch_sets/{data_set_dir}')
        csv_name_train = os.path.join(dir_, 'train_set.csv')
        csv_name_test = os.path.join(dir_, 'test_set.csv')

        # intanciate the Loader classes
        self.batch_size = batch_size # arg
        self.train_Loader = ClassifierLoader(self.device, csv_name_train, batch_size=batch_size)
        self.test_Loader = ClassifierLoader(self.device, csv_name_test, batch_size=batch_size)

        # load the model
        model, self.preprocess = loadResNet50_unfreeze(n_classes)
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
        self.train_Loader.shuffle()
        NN = self.train_Loader.NN
         
        for sub in range(0, self.train_Loader.paths.shape[0]-self.batch_size, self.batch_size):
            # sample a batch of data 
            X, Y_true = self.train_Loader.load_samples(id=sub)
            X = self.preprocess(X) # preprocess should happen in the dataloader  
            y_pred = self.model(X)
            loss = self.lossf(y_pred, Y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print('sub loss : ', total_loss)

        print(f"Epoch [{steps+1}/{self.epochs}], Train Loss: {total_loss/NN:.4f}")
        # print(f'steps: {steps}/{N_steps}, training loss: ', loss.item())
        self.log_file_.training_step.append(steps)
        self.log_file_.training_loss.append(total_loss/NN)

    def test_model(self, steps):
        print('Testing...')
        total_loss = 0
        NN = self.test_Loader.NN
         
        for sub in range(0, self.test_Loader.paths.shape[0] - self.batch_size, self.batch_size):
            # sample a batch of data 
            X, Y_true = self.test_Loader.load_samples(id=sub)
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
    mod_train = Trainer(n_classes=xargs.n_classes,
                        batch_size = xargs.batch_size, 
                        learning_rate = xargs.learning_rate,
                        epochs = xargs.epochs, tag = xargs.tag,
                        data_set_dir=xargs.data_set_dir )
    
    mod_train.classifier_training()