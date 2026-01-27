import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# quick and dirty. 
if __name__ =='__main__':

    odir = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/ResNet50_add_c_headMLP_mass_v0')
    train_dat = pd.read_csv(os.path.join(odir, 'trainlog.csv'))
    test_dat = pd.read_csv(os.path.join(odir, 'testlog.csv'))

        
    plt.figure()
    plt.plot(train_dat['step'].values, train_dat['training_loss'].values, label="train loss")
    plt.plot(test_dat['step'].values, test_dat['testing_loss'].values, label="test loss")
    
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Testing loss for finetuning ResNet50+MLP ")
    # Save the figure
    plt.savefig(os.path.join(odir, "lineplot.png"), dpi=300, bbox_inches="tight")