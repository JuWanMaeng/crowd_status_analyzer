import numpy as np
import torch
from tqdm import tqdm



# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl,device, phase=None, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl)
    if phase=='train':
        tq=tqdm(dataset_dl, ncols=80, smoothing=0, bar_format='train: {desc}|{bar}{r_bar}')
    elif phase=='test':
        tq=tqdm(dataset_dl, ncols=80, smoothing=0, bar_format='test: {desc}|{bar}{r_bar}')
    else:
        tq=tqdm(dataset_dl, ncols=80, smoothing=0, bar_format='val: {desc}|{bar}{r_bar}')
    for data in tq:
        img=data[0]
        label=data[1]
        img=img.to(device)
        label=label.to(device)
        output = model(img)
        

        loss_b, metric_b = loss_batch(loss_func, output, label, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

  
    loss = running_loss / len_data
    metric = running_metric / len(dataset_dl.dataset)
    return loss, metric