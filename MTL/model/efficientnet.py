import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import efficientnet_b0,efficientnet_b1
from torchsummary import summary


    
class EmotionModel(nn.Module):
    def __init__(self,phase='train',num_emotions=7):
        super(EmotionModel, self).__init__()
        if phase=='train':
            efficientnet = efficientnet_b1(weights='EfficientNet_B1_Weights.IMAGENET1K_V1')
        else:
            efficientnet = efficientnet_b1()
        
        self.feature_extractor = efficientnet
        
        self.fc_emotion =nn.Sequential(nn.Linear(1000,512),
                                       nn.ReLU(),
                                       nn.Linear(512,num_emotions)) 
    
        

    def forward(self, x):
        features = self.feature_extractor(x)
        
        emotion_logits = self.fc_emotion(features)

        return emotion_logits


class GenderModel(nn.Module):
    def __init__(self,phase='train'):
        super(GenderModel, self).__init__()
        
        if phase=='train':
            efficientnet = efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
        else:
            efficientnet = efficientnet_b0()
        
        self.feature_extractor = efficientnet
        
        self.fc_age =nn.Sequential(nn.Linear(1000,512),
                                       nn.ReLU(),
                                       nn.Linear(512,2)
                                       ) 
        
        # apply 함수를 사용하여 모든 nn.ReLU를 nn.GELU로 바꾸기.
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                setattr(module, 'inplace', False)
                new_module = nn.GELU()
                setattr(module, 'new_module', new_module)

    def forward(self, x):
        features = self.feature_extractor(x)
        
        gender_logits = self.fc_age(features)


        return gender_logits


    
if __name__ == '__main__':
    model=GenderModel().to(device='cpu')

    
    print(summary(model,(3,128,128),device='cpu'))
    print(model)
    
    
    # def print_layer_output(module, input, output):
    #     print(module.__class__.__name__, "output shape:", output.shape)


    # Register the forward hook on each layer of the model
    # for name, module in model.named_modules():
    #     module.register_forward_hook(print_layer_output)

    # # Feed an example input tensor through the model
    # input_tensor = torch.randn(1, 3, 128,128)
    # output = model(input_tensor)
    
    