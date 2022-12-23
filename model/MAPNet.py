
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import numpy as np
import torchvision.transforms as transforms

import os
import argparse
import sys

#from models import *
sys.path.append("../..")

    
import torch.nn as nn
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    
import torch
import torch.nn as nn
from torch.nn import functional as F
class RB1(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(RB1, self).__init__()
        aa=2
        self.conv1 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=1,dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=3,dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=4,dilation=4)
        self.bn4 = nn.BatchNorm2d(out_channels//aa)        
        
        self.conv6 = nn.Conv2d(out_channels*4//aa, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(out_channels)  

        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(out_channels)          
        
        #self.Drop =torch.nn.Dropout(p=0.2, inplace=False)
        self.p = nn.PReLU()
        self.se = SEWeightModule(out_channels*2,2)

    def forward(self, x):
        output1 = self.conv1(x)
        output1 = self.p(self.bn1(output1))
        
        output2 = self.conv2(x)
        output2 = self.p(self.bn2(output2))       
        
        output3 = self.conv3(x)
        output3 = self.p(self.bn3(output3))     
        
        output4 = self.conv4(x)
        output4 = self.p(self.bn4(output4))
        
        output = torch.cat([output1,output2,output3,output4], 1)
        Woutput= self.se(output)
        output = output * Woutput
        
        xd = self.conv7(x)
        xd = self.p(self.bn7(xd))  
        
        #print((output+xd).shape)
        
        #output6 = self.conv6( output+xd )
        output6 = self.conv6(output) 
        output6 = self.p(self.bn6(output6)+xd )

        return output6 
                         



class RB2(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(RB2, self).__init__()
        aa=2
        self.conv1 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=1,dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=3,dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels//aa)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels//aa, kernel_size=3, stride=1, padding=4,dilation=4)
        self.bn4 = nn.BatchNorm2d(out_channels//aa)          
        
        self.conv6 = nn.Conv2d(out_channels*4//aa, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(out_channels)  

        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(out_channels)          
        
        #self.Drop =torch.nn.Dropout(p=0.2, inplace=False)
        self.p = nn.PReLU()
        self.se = SEWeightModule(out_channels*2,2)

    def forward(self, x):
        output1 = self.conv1(x)
        output1 = self.p(self.bn1(output1))
        
        output2 = self.conv2(x)
        output2 = self.p(self.bn2(output2))       
        
        output3 = self.conv3(x)
        output3 = self.p(self.bn3(output3))     
        
        output4 = self.conv4(x)
        output4 = self.p(self.bn4(output4))
        
        output = torch.cat([output1,output2,output3,output4], 1)
        Woutput= self.se(output)
        output = output * Woutput
        
        xd = self.conv7( x)
        xd = self.p(self.bn7(xd))  
        
        output6 = self.conv6(output)    
        output6 = self.p(self.bn6(output6)+xd )       

        return output6 



class MAPB(nn.Module):
    def __init__(self):
        
        super(MAPB, self).__init__()
        self.conv1 = RB1(1,16)
        self.conv2 = RB2(16,32)
        self.conv3 = RB2(32,32)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(32, 14)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        
        out = self.avgpool(out)
        out1 = out.reshape(x.shape[0], -1)
        out = self.fc1(out1)

        return out1,out        
    
    