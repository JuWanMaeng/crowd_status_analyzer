import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet18
from torchsummary import summary

class MultiTaskModel(nn.Module):
    def __init__(self,phase=None):
        super(MultiTaskModel, self).__init__()
        if phase=='train':
            resnet=resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        else:
            print('not pretrained model loaded')
            resnet=resnet18()
        
        self.feature_extractor=resnet
        self.phase=phase
        # Define the task-specific output layers
        self.gender_fc = nn.Sequential(
            #nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 2)
        )

        self.emo_fc = nn.Sequential(
            #nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 7)
        )

    def forward(self, x, task=None):  #['A_G', 'E_P']
        
        features=self.feature_extractor(x)

        if self.phase != 'test':
            if task=='A_G':
                gender_output=self.gender_fc(features)
                gender=gender_output
                return gender
            
            else:
                emo_output=self.emo_fc(features)
                emo=emo_output
                return emo

        else: # test mode
            gender=self.gender_fc(features)
            emo=self.emo_fc(features)
            
            return gender,emo
