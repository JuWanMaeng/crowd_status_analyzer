import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import mobilenet_v3_large,mobilenet_v3_small
from torchsummary import summary

class EmotionModel(nn.Module):
    def __init__(self,num_emotions=7):
        super(EmotionModel, self).__init__()
        
        self.feature_extractor = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        
        self.fc_emotion =nn.Sequential(nn.Linear(1000,512),
                                       nn.ReLU(),
                                       nn.Linear(512,7)) 
        
        # apply 함수를 사용하여 모든 nn.ReLU를 nn.GELU로 바꾸기.
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                setattr(module, 'inplace', False)
                new_module = nn.GELU()
                setattr(module, 'new_module', new_module)
        


    def forward(self, x):
        features = self.feature_extractor(x)
        
        emotion_logits = self.fc_emotion(features)

        return emotion_logits


if __name__ == '__main__':
    model=EmotionModel()
    input_data = torch.randn(1, 3, 224, 224)
    output = model(input_data)
    print(output)