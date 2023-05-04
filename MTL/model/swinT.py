import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import swin_t
from torchsummary import summary


    
class EmotionModel(nn.Module):
    def __init__(self,phase='train',num_emotions=7):
        super(EmotionModel, self).__init__()
        if phase=='train':
            swin = swin_t(weights='Swin_T_Weights.IMAGENET1K_V1')
        else:
            swin = swin_t()
        
        self.feature_extractor = swin
        
        self.fc_emotion =nn.Sequential(nn.Linear(1000,512),
                                       nn.ReLU(),
                                       nn.Linear(512,num_emotions)) 
    
        

    def forward(self, x):
        features = self.feature_extractor(x)
        
        emotion_logits = self.fc_emotion(features)

        return emotion_logits


if __name__ == '__main__':
    model=EmotionModel()
    print(summary(model,input_size=(3,128,128),device='cpu'))