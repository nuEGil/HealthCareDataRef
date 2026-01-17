import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

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

class ResNet50WithHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # print shows the order of definition here - not the order of execution inforward. 
        weights = ResNet50_Weights.DEFAULT
        base = resnet50(weights=weights)
        self.transforms = weights.transforms()

        self.layer4 = base.layer4

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
            self.layer4,
        )

        # 2048 → 1 channel
        self.conv_head = nn.Conv2d(2048, 1, kernel_size=1) # channel is supervised by new labels. 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_head(x)   # [B, 1, H, W]
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

def loadResNet50_unfreeze(num_classes):
    model = ResNet50WithHead(num_classes)

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last conv block
    for p in model.layer4.parameters():
        p.requires_grad = True

    # unfreeze head
    for p in model.conv_head.parameters():
        p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True

    return model, model.transforms


