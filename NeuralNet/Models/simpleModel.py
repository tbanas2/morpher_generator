import torch.nn as nn
import torch.nn.functional as F

#Adapted from https://github.com/nalbert9/Facial-Keypoint-Detection

class FaceKeypointModel1(nn.Module):
    def __init__(self):
        '''
        A CNN for face landmarks with 3 convolutional layers.
        '''
        super(FaceKeypointModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 30) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool(x)
         x = F.relu(self.conv2(x))
         x = self.pool(x)
         x = F.relu(self.conv3(x))
         x = self.pool(x)
         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc1(x) 
         return out
    
# Adapted from https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
class FaceKeypointModel2(nn.Module):
    '''
    A CNN for classifying faces using 5 convolutions.
    '''
    def __init__(self):
        super(FaceKeypointModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 58*2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
