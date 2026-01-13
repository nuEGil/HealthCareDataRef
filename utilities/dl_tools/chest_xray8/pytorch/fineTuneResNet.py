import os
import csv
import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from PIL import Image
from dataclasses import dataclass, field, asdict
from torchvision.models import resnet50, ResNet50_Weights
'''
ok resnet with frozen weidths except the last layer fine tunes on these image patches 
great. means the training loop is fine... 

so now we have the option to try a few different training set ups. patches vs full image
vs. n classes. etc. 
'''
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

def save_training_log(fname_full, obj):
    with open(fname_full, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "training_loss"])  # header

        for s, l in zip(obj.step, obj.training_loss):
            writer.writerow([s, l])

def make_res_net_compatible(num_classes):
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

def classifier_training(optimizer, loader, model, preprocess, epochs):
    model.train()
    NN = loader.paths.shape[0] // loader.batch_size
    log_ = log_file()
    odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], 'user_meta_data/pytorch_resnet_v0')
    if not os.path.exists(odir):
        os.makedirs(odir)

    tag = '00'
    lossf = nn.CrossEntropyLoss()
    
    for steps in range(epochs):
        total_loss = 0
        loader.shuffle()
        for sub in range(0, loader.paths.shape[0]-loader.batch_size, loader.batch_size):
            # sample a batch of data 
            X, Y_true = loader.load_samples(id=sub)
            X =preprocess(X) # preprocess should happen in the dataloader  
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
            save_training_log(os.path.join(odir, 'trainlog.csv'), log_)

    torch.save({
                'epoch': steps,
                'model_state': model.state_dict(),
                # 'optimizer_state': optimizer.state_dict(),
            }, f'{odir}/mod_tag-{tag}_steps-{steps}.pt')

    return log_, model, optimizer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------ Using device: {device} ------")
    
   
    model, preprocess = make_res_net_compatible(2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # print a copy of the model.
    print(model)

    CSampler = ClassifierSampler(device, batch_size=50)
    classifier_training(optimizer, CSampler, model, preprocess, epochs=100)
