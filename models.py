## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # maxpool that uses a square window of kernel_size=2, stride=2
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.dropout1 = nn.Dropout(p=0.1)
   
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)  
        self.dropout2 = nn.Dropout(p=0.15)   
        
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2, 2)  
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.pool4 = nn.MaxPool2d(2, 2)  
        self.dropout4 = nn.Dropout(p=0.25)
        
        self.conv5 = nn.Conv2d(256, 512, 5)
        self.pool5 = nn.MaxPool2d(2, 2)  
        self.dropout5 = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(4608, 2560)
        self.dropout6 = nn.Dropout(p=0.35)
        
        self.fc2 = nn.Linear(2560, 1280)
        self.dropout7 = nn.Dropout(p=0.4)
        
        self.fc3 = nn.Linear(1280, 640)
        self.dropout8 = nn.Dropout(p=0.45)
        
        self.fc4 = nn.Linear(640, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))   (self.conv1_bn
        
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.relu(self.conv4(x))))
        x = self.dropout5(self.pool5(F.relu(self.conv5(x))))
        
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = self.dropout6(F.relu(self.fc1(x)))
        x = self.dropout7(F.relu(self.fc2(x)))
        x = self.dropout8(F.relu(self.fc3(x)))
        
        x = self.fc4(x)
        #x = F.log_softmax(x, dim=1)
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
# instantiate and print your Net
#net = Net()
#print(net)