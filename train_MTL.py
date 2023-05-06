import torch
import torch.nn as nn
from MTL.model.MTL_model import MultiTaskModel
from data.datasets import EmotionDataset,GenderDataset,AgeDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm
from utils.train_utils import get_lr
from utils.loss import FocalLoss
import copy
import wandb,cv2,time
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from torchsummary import summary

###################################################################
wandb.init(project='MultiTask',entity='kookmin_ai')
device='cuda:2' if torch.cuda.is_available() else 'cpu'
backbone='resnet18'
model=MultiTaskModel(phase='train')
emo_weight=1
gender_weight=1
age_weight=1
num_epochs=100
##################################################################
model_name=f'weight/MTL/{backbone}_MTL_step.pt'
wandb.run.name=(f'{backbone}_step')
print(f'device:{device},backbone:{backbone}')

# build model
model=MultiTaskModel(phase='train')
model.to(device)

# Define the loss functions for each task

gender_criterion=nn.CrossEntropyLoss()
emotion_criterion = nn.CrossEntropyLoss()
age_criterion=nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Define dataloader
train_gender_dataset=GenderDataset(phase='train')
train_emo_dataset=EmotionDataset(phase='train')
train_age_dataset=AgeDataset(phase='train')

val_gender_dataset=GenderDataset(phase='val')
val_emo_dataset=EmotionDataset(phase='val')
val_age_dataset=AgeDataset(phase='val')

test_gender_dataset=GenderDataset(phase='test')
test_emo_dataset=EmotionDataset(phase='test')
test_age_dataset=AgeDataset(phase='test')

train_gender_loader=DataLoader(train_gender_dataset,batch_size=128,shuffle=True,num_workers=4)
train_emo_loader=DataLoader(train_emo_dataset,batch_size=128,shuffle=True,num_workers=4)
train_age_loader=DataLoader(train_age_dataset,batch_size=128,shuffle=True,num_workers=4)

val_gender_loader=DataLoader(val_gender_dataset,batch_size=64,shuffle=True,num_workers=2)
val_emo_loader=DataLoader(val_emo_dataset,batch_size=64,shuffle=True,num_workers=2)
val_age_loader=DataLoader(val_age_dataset,batch_size=64,shuffle=True,num_workers=2)

test_gender_loader=DataLoader(test_gender_dataset,batch_size=16,shuffle=True,num_workers=1)
test_emo_loader=DataLoader(test_emo_dataset,batch_size=16,shuffle=True,num_workers=1)
test_age_loader=DataLoader(test_age_dataset,batch_size=16,shuffle=True,num_workers=1)

print(f'total gender train data:{len(train_gender_loader.dataset)}')
print(f'total emotion train data:{len(train_emo_loader.dataset)}')
print(f'total age train data:{len(train_age_loader.dataset)}')

print(f'total gender val data:{len(val_gender_loader.dataset)}')
print(f'total emotion val data:{len(val_emo_loader.dataset)}')
print(f'total age val data:{len(val_age_loader.dataset)}')

print(f'total gender test data:{len(test_gender_loader.dataset)}')
print(f'total emotion test data:{len(test_emo_loader.dataset)}')
print(f'total age test data:{len(test_age_loader.dataset)}')

'''
total gender train data:71144
total emotion train data:73437
total age train data:80370

total gender val data:13844
total emotion val data:9180
total age val data:10047

total gender test data:9844
total emotion test data:9176
total age test data:10041

'''

best_loss=1000


for epoch in range(num_epochs):
    print('*'*60)
    current_lr=get_lr(optimizer)
    print(f'{epoch+1}/{num_epochs} current_lr:{current_lr}')
    
    train_total_loss=0
    train_gender_loss=0
    train_emotion_loss=0
    train_age_loss=0
    
    emo_corrects=0
    gender_corrects=0
    age_corrects=0
    
    gender_tq=tqdm(train_gender_loader,ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    
    model.train()
    for i, (gender_data, emo_data, age_data) in enumerate(zip(gender_tq, train_emo_loader, train_age_loader)):
        

        gender_img,gender=gender_data
        emo_img,emotion=emo_data
        age_img,age=age_data

        gender_img=gender_img.to(device)
        emo_img=emo_img.to(device)
        age_img=age_img.to(device)
        gender,emotion,age=gender.to(device),emotion.to(device),age.to(device)
        
        # set optimizer zero grad
        optimizer.zero_grad()

        # Forward pass
        gender_output=model(gender_img,task='gender')
        gender_pred=torch.argmax(gender_output,dim=1)   # [64]
        gender_corrects+=gender_pred.eq(gender).sum().item()
        gender_loss=gender_criterion(gender_output,gender)
        
        age_output=model(age_img,task='age')
        age_pred=torch.argmax(age_output,dim=1)   # [64]
        age_corrects+=age_pred.eq(age).sum().item()
        age_loss=age_criterion(age_output,age)
 
        
        emo_output=model(emo_img,task='emotion')
        emo_pred=torch.argmax(emo_output,dim=1)
        emo_corrects+=emo_pred.eq(emotion).sum().item()
        emo_loss=emotion_criterion(emo_output,emotion)
  
  
        # Compute the total loss
        common_loss = gender_weight * gender_loss+ emo_weight*emo_loss + age_weight + age_loss
        common_loss_value=common_loss.item()
        common_loss.backward(retain_graph=True)
        optimizer.step()
        
        # 각각의 total loss 저장
        train_total_loss+=common_loss_value
        train_gender_loss+=gender_loss.item()
        train_emotion_loss +=emo_loss.item()
        train_age_loss+=age_loss.item()
    
    # epoch 당 각각의 loss
    train_loss = train_total_loss/len(train_gender_loader)
    train_gender_loss = train_gender_loss/len(train_gender_loader)
    train_emotoin_loss = train_emotion_loss/len(train_gender_loader)
    train_age_loss=train_age_loss/len(train_age_loader)   
    # epoch 당 각각의 accuracy
    train_gender_accuracy=gender_corrects/len(train_gender_loader.dataset) * 100
    train_emotion_accuracy=emo_corrects/len(train_gender_loader.dataset) * 100
    train_age_accuracy=age_corrects/len(train_gender_loader.dataset) * 100
    
    # validation
    val_emo_tq=tqdm(val_emo_loader,ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
    val_total_loss=0
    val_gender_loss=0
    val_emotion_loss=0
    val_age_loss=0
    
    gender_corrects=0
    emo_corrects=0
    age_corrects=0
    model.eval()
    with torch.no_grad():
        for i, (emo_data, gender_data,age_data) in enumerate(zip(val_emo_tq,val_gender_loader,val_age_loader)):

            gender_img,gender=gender_data
            emo_img,emotion=emo_data
            age_img,age=age_data

            gender_img=gender_img.to(device)
            emo_img=emo_img.to(device)
            age_img=age_img.to(device)
            gender,emotion,age=gender.to(device),emotion.to(device),age.to(device)
            
            
            # Forward pass
            gender_output=model(gender_img,task='gender')
            gender_pred=torch.argmax(gender_output,dim=1)   # [64]
            gender_corrects+=gender_pred.eq(gender).sum().item()
            gender_loss=gender_criterion(gender_output,gender)
            
            age_output=model(age_img,task='age')
            age_pred=torch.argmax(age_output,dim=1)   # [64]
            age_corrects+=age_pred.eq(age).sum().item()
            age_loss=age_criterion(age_output,age)

            emo_output=model(emo_img,task='emotion')
            emo_pred=torch.argmax(emo_output,dim=1)
            emo_corrects+=emo_pred.eq(emotion).sum().item()
            emo_loss=emotion_criterion(emo_output,emotion)
    
 
    
            # Compute the total loss
            val_loss = gender_loss + emo_loss + age_loss
            val_total_loss+=val_loss.item()
            val_gender_loss+=gender_loss.item()
            val_emotion_loss+=emo_loss.item()
            val_age_loss+=age_loss.item()
    
    
    val_loss=val_total_loss/len(val_emo_loader)
    val_gender_loss=val_gender_loss/len(val_emo_loader)
    val_emotion_loss=val_emotion_loss/len(val_emo_loader)
    val_age_loss=val_age_loss/len(val_age_loader)
    
    val_gender_accuracy=gender_corrects/len(val_emo_loader.dataset) * 100
    val_emotion_accuracy=emo_corrects/len(val_emo_loader.dataset) * 100
    val_age_accuracy=age_corrects/len(val_age_loader.dataset) * 100
    
    wandb.log({'train_total_loss':train_loss,
               'train_gender_loss':train_gender_loss,
               'train_emotion_loss':train_emotoin_loss,
               'train_age_loss':train_age_loss,
                'val_total_loss':val_loss,
                'val_gender_loss':val_gender_loss,
                'val_emotoin_loss':val_emotion_loss,
                'val_age_loss':val_age_loss,
                'train_gender_acc':train_gender_accuracy,
                'val_gender_acc':val_gender_accuracy,
                'train_emo_acc':train_emotion_accuracy,
                'val_emo_acc':val_emotion_accuracy,
                'train_age_acc':train_age_accuracy,
                'val_age_acc':val_age_accuracy},step=epoch+1)
    
    
    if val_loss<best_loss:
        best_loss=val_loss
        best_gender_acc=val_gender_accuracy
        best_emo_acc=val_emotion_accuracy
        best_age_acc=val_age_accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(),model_name)
        print('Copied best model weights!')
        
    scheduler.step()
    if current_lr != get_lr(optimizer):
        print('Loading best model weights')
        model.load_state_dict(best_model_wts)
    
    
    
    print(f'train loss:{train_loss:.4f}, val loss:{val_loss:.4f}, val_gender_accuracy:{val_gender_accuracy:.2f}, val_emotion_accuracy:{val_emotion_accuracy:.2f}, val_age_accuracy:{val_age_accuracy:.2f} ')

print(f'best loss:{best_loss:.4f}, best_gender_acc:{best_gender_acc:.2f}, best_emo_acc:{best_emo_acc:.2f}, best_age_acc:{best_age_acc:.2f}')

wandb.log({'best validation gender acc':best_gender_acc,
           'best validatoin emotion acc':best_emo_acc,
           'best validation age acc':best_age_acc})

print('@@@@@@@@@@@@@@test start@@@@@@@@@@@@@')
model.eval()
with torch.no_grad():
    test_emo_tq=tqdm(test_emo_loader,ncols=80, smoothing=0, bar_format='test: {desc}|{bar}{r_bar}')
    for i, (emo_data, gender_data,age_data) in enumerate(zip(test_emo_tq,test_gender_loader,test_age_loader)):
        gender_img,gender=gender_data
        emo_img,emotion=emo_data
        age_img,age=age_data

        gender_img=gender_img.to(device)
        emo_img=emo_img.to(device)
        age_img=age_img.to(device)
        gender,emotion,age=gender.to(device),emotion.to(device),age.to(device)

                # Forward pass
        gender_output=model(gender_img,task='gender')
        gender_pred=torch.argmax(gender_output,dim=1)   # [64]
        gender_corrects+=gender_pred.eq(gender).sum().item()

        
        age_output=model(age_img,task='age')
        age_pred=torch.argmax(age_output,dim=1)   # [64]
        age_corrects+=age_pred.eq(age).sum().item()


        emo_output=model(emo_img,task='emotion')
        emo_pred=torch.argmax(emo_output,dim=1)
        emo_corrects+=emo_pred.eq(emotion).sum().item()

    test_gender_accuracy=gender_corrects/len(test_emo_loader.dataset) * 100
    test_emotion_accuracy=emo_corrects/len(test_emo_loader.dataset) * 100
    test_age_accuracy=age_corrects/len(test_age_loader.dataset) * 100
    
wandb.log({'best test gender acc':test_gender_accuracy,
           'best test emotion acc':test_emotion_accuracy,
           'best test age acc':test_age_accuracy})