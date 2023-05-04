import torch
import cv2
import numpy as np
import wandb
import time


# wandb에 image 결과를 plot하는 코드
def vis(model,best_model_wts,device,max_epoch,train_losses,val_losses,train_acc,val_acc,test_imgs):
    example_images=[]
    cat={0:'sad', 1:'happy', 2:'angry', 3:'disgust', 4:'surprise', 5:'fear', 6:'neutral'}
    model.load_state_dict(best_model_wts)
    model.eval()
    
    for image in test_imgs:
        with torch.no_grad():

            img=cv2.imread(image)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            img=cv2.resize(img,(128,128))
            img=np.transpose(img,(2,0,1))
            img=torch.FloatTensor(img).to(device)
            img=torch.unsqueeze(img,0)/255.0
            
            start=time.time()
            emo_output=model(img)
            infer_time=time.time()-start
            
            emo_pred=emo_output.argmax(1,keepdim=True)
            emotion=cat[emo_pred.item()]
        example_images.append(wandb.Image(
                        img, caption=f'Pred:{emotion}, inference_time:{infer_time:.4f}'))
    wandb.log({"Image": example_images})
    # 여기까지

    # wandb에 loss와 acc를 train 이랑 val을 함께 plot하는 코드
    wandb.log({
        'train_val_loss': wandb.plot.line_series(
            xs=list(range(1,max_epoch+1)),
            ys=[train_losses, val_losses],
            keys=['Train Loss', 'Validation Loss'],
            title='Train vs. Validation Loss',
            xname='Epoch'
            
        )
    })
    wandb.log({
        'train_val_acc': wandb.plot.line_series(
            xs=list(range(1,max_epoch+1)),
            ys=[train_acc, val_acc],
            keys=['Train acc', 'Validation acc'],
            title='Train vs. Validation accuracy',
            xname='Epoch'
            
        )
    })