import cv2
import wandb
import numpy as np
import time
import torch
from MTL.model import resnet,MTL_model
import glob
from torchvision.models import  resnet18
import torch.nn as nn
from data.datasets import EmotionDataset,GenderDataset,AgeDataset
from torch.utils.data import DataLoader
import random

emo_label={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
gender_label={0:'man',1:'woman'}
age_label={0:'youth', 1: 'student', 2: 'adult', 3:'elder'}

test_gender_dataset=GenderDataset(phase='test')
test_emo_dataset=EmotionDataset(phase='test')
test_age_dataset=AgeDataset(phase='test')

test_gender_loader=DataLoader(test_gender_dataset,batch_size=16,shuffle=True,num_workers=1)
test_emo_loader=DataLoader(test_emo_dataset,batch_size=16,shuffle=True,num_workers=1)
test_age_loader=DataLoader(test_age_dataset,batch_size=16,shuffle=True,num_workers=1)
device='cuda:3' if torch.cuda.is_available() else 'cpu'

gender_model=resnet18()
gender_model.fc=nn.Linear(512,2)
gender_wt=torch.load('weight/classification/UTK_gender_best_model.pt')

emotion_model=resnet.EmotionModel(phase='test')
emotion_wt=torch.load('weight/classification/emotion128_resnet18.pt')

age_model=resnet.AgeModel(phase='test')
age_wt=torch.load('weight/classification/age128_4class_merge_resnet18.pt')


gender_model.to(device)
emotion_model.to(device)
age_model.to(device)



emotion_model.load_state_dict(emotion_wt)
gender_model.load_state_dict(gender_wt)
age_model.load_state_dict(age_wt)


emotion_model.eval()
gender_model.eval()
age_model.eval()

MTL_model=MTL_model.MultiTaskModel(phase='test')
MTL_wt=torch.load('weight/MTL/resnet18_MTL_212.pt')
MTL_model.to(device)
MTL_model.load_state_dict(MTL_wt)
MTL_model.eval()

length =30
unique_list = []
while len(unique_list) < length:
    num = random.randint(1, 9000)
    if num not in unique_list:
        unique_list.append(num)
        

wandb.init(project='single tasks result',entity='kookmin_ai')
wandb.run.name=(f'emotion_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        emo_img=test_emo_dataset[i][0]
        emo_answer=emo_label[test_emo_dataset[i][1]]
        emo_img=emo_img.to(device)
        emo_img=emo_img.unsqueeze(0)
        
        start=time.time()
        emo_output=emotion_model(emo_img)
        infer_time=time.time()-start
        
        emo_pred=emo_output.argmax(1,keepdim=True)
        emotion=emo_label[emo_pred.item()]
    example_images.append(wandb.Image(
                    emo_img, caption=f'Pred:{emotion},  Answer:{emo_answer}'))
wandb.log({"Image": example_images})

wandb.init(project='single tasks result',entity='kookmin_ai')
wandb.run.name=(f'age_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        age_img=test_age_dataset[i][0]
        age_answer=age_label[test_age_dataset[i][1]]
        age_img=age_img.to(device)
        age_img=age_img.unsqueeze(0)
        
        start=time.time()
        age_output=age_model(age_img)
        infer_time=time.time()-start
        
        age_pred=age_output.argmax(1,keepdim=True)
        age=age_label[age_pred.item()]
    example_images.append(wandb.Image(
                    age_img, caption=f'Pred:{age},  Answer:{age_answer}'))
wandb.log({"Image": example_images})

wandb.init(project='single tasks result',entity='kookmin_ai')
wandb.run.name=(f'gender_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        gender_img=test_gender_dataset[i][0]
        gender_answer=gender_label[test_gender_dataset[i][1]]
        gender_img=gender_img.to(device)
        gender_img=gender_img.unsqueeze(0)
        
        start=time.time()
        gender_output=gender_model(gender_img)
        infer_time=time.time()-start
        
        gender_pred=gender_output.argmax(1,keepdim=True)
        gender=gender_label[gender_pred.item()]
    example_images.append(wandb.Image(
                    gender_img, caption=f'Pred:{gender},  Answer:{gender_answer}'))
wandb.log({"Image": example_images})

wandb.init(project='multi tasks result',entity='kookmin_ai')
wandb.run.name=(f'emotion_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        emo_img=test_emo_dataset[i][0]
        emo_answer=emo_label[test_emo_dataset[i][1]]
        emo_img=emo_img.to(device)
        emo_img=emo_img.unsqueeze(0)
        
        start=time.time()
        outputs=MTL_model(emo_img)
        emo_output=outputs[1]
        infer_time=time.time()-start
        
        emo_pred=emo_output.argmax(1,keepdim=True)
        emotion=emo_label[emo_pred.item()]
    example_images.append(wandb.Image(
                    emo_img, caption=f'Pred:{emotion},  Answer:{emo_answer}'))
wandb.log({"Image": example_images})

wandb.init(project='multi tasks result',entity='kookmin_ai')
wandb.run.name=(f'age_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        age_img=test_age_dataset[i][0]
        age_answer=age_label[test_age_dataset[i][1]]
        age_img=age_img.to(device)
        age_img=age_img.unsqueeze(0)
        
        start=time.time()
        outputs=MTL_model(age_img)
        age_output=outputs[2]
        infer_time=time.time()-start
        
        age_pred=age_output.argmax(1,keepdim=True)
        age=age_label[age_pred.item()]
    example_images.append(wandb.Image(
                    age_img, caption=f'Pred:{age},  Answer:{age_answer}'))
wandb.log({"Image": example_images})

wandb.init(project='multi tasks result',entity='kookmin_ai')
wandb.run.name=(f'gender_test')

example_images=[]
for i in unique_list:
    with torch.no_grad():

        gender_img=test_gender_dataset[i][0]
        gender_answer=gender_label[test_gender_dataset[i][1]]
        gender_img=gender_img.to(device)
        gender_img=gender_img.unsqueeze(0)
        
        start=time.time()
        outputs=MTL_model(gender_img)
        gender_output=outputs[0]
        infer_time=time.time()-start
        
        gender_pred=gender_output.argmax(1,keepdim=True)
        gender=gender_label[gender_pred.item()]
    example_images.append(wandb.Image(
                    gender_img, caption=f'Pred:{gender},  Answer:{gender_answer}'))
wandb.log({"Image": example_images})