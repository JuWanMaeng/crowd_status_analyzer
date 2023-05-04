import torch
import torch.nn as nn
from MTL.model.MTL_model import MultiTaskModel
from data.datasets import ExpDataset,GenderDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm
from utils.train_utils import get_lr
from utils.loss import FocalLoss
import copy
import wandb,cv2,time
import numpy as np
import glob

from torchsummary import summary
wandb.init(project='emotoin+gender',entity='kookmin_ai')


device='cuda:1'
state_dict=torch.load('weight/MTL/resnet18_only_FC_best.pt')
test_model=MultiTaskModel(phase='test')
test_model.load_state_dict(state_dict)
test_model.to(device)

example_images=[]
emo_label={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender_label={0:'man',1:'woman'}

test_model.eval()
a=glob.glob('testimgs/*.png')
for image in a:
    with torch.no_grad():

        img=cv2.imread(image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img=cv2.resize(img,(128,128))
        img=np.transpose(img,(2,0,1))
        img=torch.FloatTensor(img).to(device)
        img=torch.unsqueeze(img,0)/255.0
        
        start=time.time()
        gender_output,emotion_output=test_model(img)
        infer_time=time.time()-start
        
        emo_pred=emotion_output.argmax(1,keepdim=True)
        gender_pred=gender_output.argmax(1,keepdim=True)
        emotion=emo_label[emo_pred.item()]
        gender=gender_label[gender_pred.item()]
    example_images.append(wandb.Image(
                    img, caption=f'Pred:{gender},{emotion}, inference_time:{infer_time:.4f}'))
wandb.log({"Image": example_images})