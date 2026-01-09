import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np

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
                 width_param = 2, pool_rate = 2, n_out=3):
        
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

            self.layers.append(nn.MaxPool2d([2, 2]))
            # halve the input shape
            input_shape = [n_kerns, int(input_shape[1]/2), int(input_shape[2]/2)]
        
        # once we are at the end. we wnat to flatten 
        self.layers.append(nn.Flatten())

        # the shape is going to be 
        flatten_shape = int(((input_shape[1]/num_pools)**2) * n_kerns)
        # this one reduces the flatten shape to something we can work with
        self.layers.append(nn.Linear(flatten_shape, 64))
        # then this one calls the MLP layer
        self.layers.append(MLP(neurons_in = 64, dropout_rate = 0.5,
                                n_out = n_out, activation='softmax'))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"------ Using device: {device} ------")
    
    # define the model and sent it to the device
    model = BlockStack( input_shape = [3, 128,128], nblocks = 4, n_kerns = 32, 
                        width_param = 2, pool_rate = 2, n_out = 3)
    model = model.to(device)
    

    # print a copy of the model.
    print(model)
    xx = np.random.rand(1, 3, 128,128).astype(np.float32)
    input_ = torch.from_numpy(xx).to(device)
    
    output = model(input_)
    # on calling you want (Nsamples, Channels in, Height in, Width in)
    print(output.shape)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)