import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

def init_weights(m):
    """Better initialization for training from scratch"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Block(nn.Module):
    def __init__(self, 
                 n_kerns_in: int, 
                 n_kerns_out: int,
                 norm_groups: int = 8):
        super().__init__()

        Conv = nn.Conv2d

        self.conv0 = Conv(n_kerns_in, n_kerns_in,
            kernel_size=1, stride=1, padding=0, bias=True)

        self.conv1 = Conv(n_kerns_in, n_kerns_in,
            kernel_size=7, stride=1, padding=3, bias=True)

        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

        self.norm = nn.GroupNorm(num_groups=min(norm_groups, n_kerns_in),
                                 num_channels=n_kerns_in)

        self.conv2 = Conv(n_kerns_in, n_kerns_out,
                          kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        identity = x

        out = self.conv0(x)   # kernel size 1 
        out = self.conv1(out) # kernel size 7
        out = out + identity  # reidual sum
        out = self.act(out)   # relu 1
        out = self.norm(out)  # group normalize
        out = self.conv2(out) # convolve for new channels out
        out = self.act2(out)  # relu2 
        return out

# Multilayer perceptron - as a custom layer -- this is normally 
# the last few layers for your classificaiton model
class MLP(nn.Module):
    def __init__(self,  neurons_in = 64, dropout_rate = 0.5, 
                 n_out = 1, activation='sigmoid'):
        super().__init__()
         
        dims = [(neurons_in, neurons_in),
                (neurons_in, neurons_in * 2),
                (neurons_in * 2, neurons_in),]

        self.blocks = nn.ModuleList()
        for nin, nout in dims:
            self.blocks.extend([
                nn.Linear(nin, nout),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ])

        self.norm = nn.LayerNorm(neurons_in)
        self.final = nn.Linear(neurons_in, n_out)

        self.out_act = {
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1),
            None: nn.Identity(),
        }[activation]
    
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.norm(x)
        x = self.final(x)
        return self.out_act(x)

# Network backbone is just a block stack
class BlockStack(nn.Module):
    # ok so this part works.. just need to keep flattening it out. 
    def __init__(self, input_shape = [3, 128,128], nblocks = 4, n_kerns = 32, 
                 pool_rate = 2, n_out=3, last_activation = 'sigmoid'):
        
        super().__init__() # make sure the nn.module init funciton works
        
        # so i can save this later. 
        self.hyper_params_0 = { 'input_shape':input_shape, 
                              'nblocks':nblocks, 
                              'n_kerns': n_kerns,
                              'pool_rate':pool_rate,
                              'n_out':n_out,
                              'last_activation':last_activation}

        # create a module list
        self.layers = nn.ModuleList() 
        # remember its chanells in, channels out, then the rest
        self.layers.append(nn.Conv2d(3, n_kerns, kernel_size = (1, 1), 
                          stride = 1, padding = 'same', bias = True,))
        
        # mod to bring back width parameter later. 
        # itterate through the blocks
        input_shape = [n_kerns, *input_shape[1::]]
        for n in range(nblocks):
            self.layers.append(Block(n_kerns_in = n_kerns, 
                                     n_kerns_out = n_kerns))
            if n%pool_rate==0:
                self.layers.append(nn.MaxPool2d([2, 2]))
                # halve the input shape
                input_shape = [n_kerns, int(input_shape[1]/2), int(input_shape[2]/2)]
        
        # --- final 1×1 conv (channel mixer / head prep) ---
        self.layers.append(
        nn.Conv2d(n_kerns, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # once we are at the end. we wnat to flatten 
        self.layers.append(nn.Flatten())

        # the shape is going to be 
        # flatten_shape = n_kerns * input_shape[1] * input_shape[2]
        flatten_shape = 1 * input_shape[1] * input_shape[2]
        # this one reduces the flatten shape to something we can work with
        self.layers.append(nn.Linear(flatten_shape, 64))
        # then this one calls the MLP layer
        self.layers.append(MLP(neurons_in = 64, dropout_rate = 0.5,
                                n_out = n_out, activation=last_activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   

def loadBlockStack(num_classes):
    model = BlockStack(input_shape = [3, 128,128], 
                       nblocks = 32, n_kerns = 32, 
                       pool_rate = 8, n_out = num_classes,
                       last_activation = None)
    model.apply(init_weights)
    # to stay consistent with the other model loads. 
    def transform(x):
        return x
    
    return model, transform

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

    return model, weights.transforms

class ResNet50WithHead(nn.Module):
    def __init__(self, num_classes, input_size=128):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        base = resnet50(weights=weights)
        
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        
        # Calculate spatial size: ResNet50 reduces by 32×
        spatial_size = input_size // 32
        num_features = spatial_size * spatial_size
        
        self.conv_head = nn.Conv2d(2048, 1, kernel_size=1)
        
        # Upsample from 4×4 to 8×8 (for 128×128 input)
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        # Now we have 8×8 = 64 features instead of 4×4 = 16
        self.fc = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.5)

        self.deploy_ = False
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_head(x)      # [B, 1, 4, 4]
        upsamp = self.upsample(x)       # [B, 1, 8, 8]
        out = self.flat(upsamp)           # [B, 64]
        out = self.fc(out)             # [B, num_classes]
        out = self.drop(out)
        
        if self.deploy_:
            return out, upsamp
        else:
            return out

class ResNet50WithHeadMLP(nn.Module):
    def __init__(self, num_classes, input_size=128):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        base = resnet50(weights=weights)
        
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        
        # Calculate spatial size: ResNet50 reduces by 32×
        spatial_size = input_size // 32
        num_features = spatial_size * spatial_size
        
        self.conv_head = nn.Conv2d(2048, 1, kernel_size=1)
        # Upsample from 4×4 to 8×8 (for 128×128 input)
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        # Now we have 8×8 = 64 features instead of 4×4 = 16
        self.mlp = MLP(neurons_in = 64, dropout_rate = 0.5,
                                n_out = 64, activation=None)
        
        self.fc = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.5)

        self.deploy_ = False
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_head(x)      # [B, 1, 4, 4]
        upsamp = self.upsample(x)       # [B, 1, 8, 8]
        out = self.flat(upsamp)           # [B, 64]
        out = self.mlp(out)
        out = self.fc(out)             # [B, num_classes]
        out = self.drop(out)

        if self.deploy_:
            return out, upsamp
        else:
            return out

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

def loadResNet50_add_convhead(num_classes, input_size=128):
    model = ResNet50WithHead(num_classes, input_size=input_size)
    
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze head
    for p in model.conv_head.parameters():
        p.requires_grad = True
    
    for p in model.upsample.parameters():
        p.requires_grad = True
        
    for p in model.fc.parameters():
        p.requires_grad = True
    
    # Transforms for TENSORS (not PIL Images)
    # Assumes input is already a tensor in [0, 1] range
    tensor_transforms = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
    
    return model, tensor_transforms


def loadResNet50_add_convheadMLP(num_classes, input_size=128):
    model = ResNet50WithHeadMLP(num_classes, input_size=input_size)
    
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze head
    for p in model.conv_head.parameters():
        p.requires_grad = True
    
    for p in model.upsample.parameters():
        p.requires_grad = True

    for p in model.mlp.parameters():
        p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True
    
    # Transforms for TENSORS (not PIL Images)
    # Assumes input is already a tensor in [0, 1] range
    tensor_transforms = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
    
    return model, tensor_transforms