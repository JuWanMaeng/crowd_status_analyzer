'''
모델의 input, output size를 확인할수 있는 코드
'''



import torch
from MTL.model import resnet,MobileNetv3




# Define a forward hook function to print the output of each layer
def print_layer_output(module, input, output):
    print(module.__class__.__name__, "output shape:", output.shape)





if __name__=='__main__':
    model=efficientnet.EmotionAgeGenderModel()

    # Register the forward hook on each layer of the model
    for name, module in model.named_modules():
        module.register_forward_hook(print_layer_output)

    # Feed an example input tensor through the model
    input_tensor = torch.randn(1, 3, 128,128)
    output = model(input_tensor)
