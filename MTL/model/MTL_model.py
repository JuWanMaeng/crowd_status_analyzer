import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet18,efficientnet_b1
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
        self.age_fc = nn.Sequential(
            #nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 4)
        )

    def forward(self, x, task=None):  #['A_G', 'E_P']
        
        features=self.feature_extractor(x)

        if self.phase != 'test':
            if task=='gender':
                gender_output=self.gender_fc(features)
                return gender_output
            elif task=='age':
                age_output=self.age_fc(features)
                return age_output                
            else:
                emo_output=self.emo_fc(features)
                return emo_output

        else: # test mode
            gender=self.gender_fc(features)
            emo=self.emo_fc(features)
            age=self.age_fc(features)
            
            return gender,emo,age
