import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.convNet = vgg16.features

        self.FC = nn.Sequential(
            nn.Linear(512*4*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.output = nn.Sequential(
            nn.Linear(4096+2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )

        # replace the maxpooling layer in VGG
        self.convNet[4] = nn.MaxPool2d(kernel_size=2, stride=1)
        self.convNet[9] = nn.MaxPool2d(kernel_size=2, stride=1)
      

    def forward(self, x_in):
        feature = self.convNet(x_in['eye'])
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)

        feature = torch.cat((feature, x_in['head_pose']), 1)
        gaze = self.output(feature)

        return gaze

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

if __name__ == '__main__':
    m = model().cuda()
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2).cuda(),
               "eye": torch.zeros(10, 3, 36, 60).cuda()
               }
    a = m(feature)
    print(m)

