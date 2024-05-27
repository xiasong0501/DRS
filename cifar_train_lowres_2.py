import numpy as np 
from datasets import get_normalize_layer
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
import datetime
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import MultiStepLR
from datasets import get_dataset, DATASETS
import pandas as pd
from numpy import random
import torch
from datasets import get_normalize_layer
from architectures import ARCHITECTURES, get_architecture
from torch.nn import CrossEntropyLoss,NLLLoss,MSELoss
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD,Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
import copy
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

class GetLoader_train(torch.utils.data.Dataset): #load data
	
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        transform=transforms.Compose([transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        data=transform(data)
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

def get_res_index(img_shape):
    H,W=img_shape.shape[-1],img_shape.shape[-2]
    pixel_amount=H*W
    shape=H*W
    false_tensor_A = torch.zeros(shape, dtype=torch.bool)
    false_tensor_B = torch.zeros(shape, dtype=torch.bool)
    count=0
    for i in range(pixel_amount):
        if i%2==0:
            if count%2==0:
                false_tensor_A[i]=True
                false_tensor_B[i+1]=True

            else:
                false_tensor_A[i+1]=True
                false_tensor_B[i]=True      
        if i%H==0:
            count=count+1
    false_tensor_A=false_tensor_A.reshape(1,H,W).repeat(3,1,1)
    false_tensor_B=false_tensor_B.reshape(1,H,W).repeat(3,1,1)
    index_tensor=torch.cat((false_tensor_A,false_tensor_B),dim=0).reshape(2,3,H,W)
    return index_tensor


def get_low_res_data(data,index_tensor,i):
    return_data=torch.tensor([])
    temp_data=torch.tensor([])
    temp=torch.tensor([])
    index=index_tensor[0].reshape(1,3,32,32).repeat(len(data),1,1,1)
    end1 = time.time() 
    data=data.cuda(non_blocking=True)
    index_tensor=index_tensor.cuda(non_blocking=True)
    temp=data*index.cuda(non_blocking=True)
    temp=F.interpolate(temp.reshape(len(data),3,32,32),size=(32,16), mode='bilinear',align_corners=False)*2 
    # temp=data
    ran=len(index_tensor)
    temp_data=temp
    while ran>1:
        ran=ran-1
        temp_data=torch.cat((temp_data,temp),dim=1)
    return_data=temp_data.reshape(len(index_tensor)*len(data),3,32,16)
    # return_data=temp_data.reshape(len(index_tensor)*len(data),3,32,32)
    return return_data

def get_low_res_data_mn(data,index_tensor,l_r,noise_sd):
    return_data=torch.tensor([])
    temp_data=torch.tensor([])
    temp=torch.tensor([])
    index=index_tensor[0].reshape(1,3,32,32).repeat(len(data),1,1,1)
    end1 = time.time() 
    data=data.cuda(non_blocking=True)
    index_tensor=index_tensor.cuda(non_blocking=True)
    temp=data*index.cuda(non_blocking=True)

    ran=len(index_tensor)
    temp_data=temp
    while ran>1:
        ran=ran-1
        temp_data=torch.cat((temp_data,temp),dim=1)
    return_data=temp_data.reshape(len(index_tensor)*len(data),3,32,32)
    return_data=return_data.cuda()
    # assert len(index_tensor)==len(return_data),print(index.shape,return_data.shape)
    index=index.repeat(len(index_tensor),1,1,1)
    # assert len(index_tensor)==len(return_data),print(index.shape,return_data.shape)
    noise = torch.randn_like(return_data, device='cuda')*noise_sd*index.cuda() 
    return_data=return_data+noise 
    if l_r=='r':
        return_data_mean=torch.nn.functional.max_pool2d(return_data, 2)
        return_data_mean=F.interpolate(return_data_mean,size=(32,32), mode='nearest')
        return_data_mean=return_data_mean*index.cuda()
        return_data=return_data+return_data_mean
    else:
        return_data_mean=torch.nn.functional.max_pool2d(return_data, 2)
        return_data_mean=F.interpolate(return_data_mean,size=(32,32), mode='nearest')
        return_data_mean=return_data_mean*index.cuda()
        return_data=return_data+return_data_mean       
    return return_data

def get_upper_res_data(data):
    data_len=len(data)
    return_data=torch.zeros([data_len,3,32,32])
    # for i in range(data_len):
    for j in range(16):
        return_data[:,:,:,2*j]=data[:,:,:,j]
        return_data[:,:,:,2*j+1]=data[:,:,:,j]
    return return_data

def kl_div(outputs_sf, outputs_sf_mean):
    assert outputs_sf.shape==outputs_sf_mean.shape, print(outputs_sf.shape,outputs_sf_mean.shape)
    return F.kl_div(torch.log(torch.clamp(outputs_sf, min=0.000001)), outputs_sf_mean, reduction='none').sum(1)
def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent
def consistency_loss(outputs_sf,outputs_sf_mean,num,w):
    outputs_sf_resized=outputs_sf.reshape((len(outputs_sf_mean), num, 10))
    # print(outputs_sf_resized.shape)
    loss_kl=[]
    for n in range(num):
        loss_kl += [kl_div(outputs_sf_resized[:,n,:], outputs_sf_mean)]
    loss_kl = sum(loss_kl) / num
    loss_ent = entropy(outputs_sf_mean)
    # print(loss_kl.shape,loss_ent.shape)
    loss_consistency = w * (loss_kl + 0.05* loss_ent)
    return loss_consistency.mean()

def My_train(loader: DataLoader, model: torch.nn.Module, model_target: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,N,LAR,l_r,cut):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        loss_p=0
        loss_acc_p=0
        model.train()
        model_target.eval()
        acc=0
        # index_tensor_whole=get_res_index()
        random_number = 0
        for i, (inputs, targets) in enumerate(loader):
            if i==0:
                index_tensor_whole=get_res_index(inputs)
            # measure data loading time
            data_time.update(time.time() - end)
            if l_r=='l':
                random_number = 0
            if l_r=='r':
                random_number = 1
            # index_tensor=index_tensor_whole.reshape(2,3,32,32)
            index_tensor=index_tensor_whole[random_number].reshape(1,3,32,32)
            inputs=inputs.cuda()
            input_size=len(inputs)
            targets = targets.cuda()

            inputs=get_low_res_data_mn(inputs,index_tensor,l_r,noise_sd)
            num=len(index_tensor)

            target_r=targets.reshape(len(targets),1)
            target_r = target_r.repeat(1,num)
            target_r=target_r.reshape(len(targets)*num)
            target_r=target_r.cuda()
            outputs = model(inputs)

            outputs_sf=F.softmax(outputs,dim=1)
            log_outputs_sf=torch.log(torch.clamp(outputs_sf, min=0.000001))
            outputs_sf_resized=outputs_sf.reshape((input_size, num, 10))
            outputs_sf_mean=outputs_sf_resized.mean(1)
            outputs_sf_mean = torch.clamp(outputs_sf_mean, min=0.000001)
            log_outputs_mean = torch.log(outputs_sf_mean)

            criterion_ind=nn.CrossEntropyLoss()
            loss_ind=criterion_ind(outputs,target_r)
            criterion_acc=nn.NLLLoss()
            loss_acc=criterion_acc(log_outputs_mean,targets)
            loss = 1.0*loss_ind
            loss_acc=loss_acc
          
            acc1, acc5 = accuracy(outputs_sf, target_r, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            acc=acc+acc1
            loss_p=loss+loss_p
            loss_acc_p=loss_acc+loss_acc_p
            if i % 20 == 0:
                #print("acc=",acc/(i+1),"cl_loss=",loss/(i+1),"rl_loss=",rl_loss1/(i+1),"robustness=",rl_loss_real1/(i+1))
                print("acc=",acc/(i+1),"loss_acc=",loss_acc_p/(i+1),"loss_kl=",loss_p/(i+1))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)



def My_train_kl(loader: DataLoader, model: torch.nn.Module, model_target: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,N,LAR,l_r,cut):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        # cl_loss1=0
        # rl_loss1=0
        loss_p=0
        # rl_loss=0
        loss_acc_p=0
        # rl_loss_real1=0
        # switch to train mode
        model.train()
        model_target.eval()
        acc=0
        # index_tensor_whole=get_res_index()
        random_number = 0
        for i, (inputs, targets) in enumerate(loader):
            if i==0:
                index_tensor_whole=get_res_index(inputs)
            # measure data loading time
            data_time.update(time.time() - end)
            if l_r=='l':
                random_number = 0
            if l_r=='r':
                random_number = 1
            # index_tensor=index_tensor_whole.reshape(2,3,32,32)
            index_tensor=index_tensor_whole[random_number].reshape(1,3,32,32)
            index_tensor=torch.cat((index_tensor,index_tensor),dim=0).reshape(2,3,32,32)
            index_tensor=torch.cat((index_tensor,index_tensor),dim=0).reshape(4,3,32,32)
            # index_tensor=torch.cat((index_tensor,index_tensor),dim=0).reshape(8,3,32,32)
            inputs_org=inputs.cuda()
            inputs=inputs.cuda()
            input_size=len(inputs)
            targets = targets.cuda()

            inputs=get_low_res_data_mn(inputs,index_tensor,l_r,noise_sd)
            num=len(index_tensor)
            target_r=targets.reshape(len(targets),1)
            target_r = target_r.repeat(1,num)
            target_r=target_r.reshape(len(targets)*num)

            inputs=inputs.cuda()
            targets = targets.cuda()
            target_r=target_r.cuda()
            outputs = model(inputs)

            outputs_sf=F.softmax(outputs,dim=1)
            log_outputs_sf=torch.log(torch.clamp(outputs_sf, min=0.000001))
            outputs_sf_resized=outputs_sf.reshape((input_size, num, 10))
            outputs_sf_mean=outputs_sf_resized.mean(1)

            outputs_sf_mean = torch.clamp(outputs_sf_mean, min=0.000001)
            log_outputs_mean = torch.log(outputs_sf_mean)
            criterion_ind=nn.CrossEntropyLoss()
            loss_ind=criterion_ind(outputs,target_r)
            
            criterion_acc=nn.NLLLoss()
            loss_acc=criterion_acc(log_outputs_mean,targets)
            # inputs_org=inputs_org.cuda()

            w=15
            loss_consistency=consistency_loss(outputs_sf,outputs_sf_mean,num,w)
            loss = 0*loss_acc + 1.0*loss_ind+1*loss_consistency
            # +0*(loss_ind_org+w*loss_kl_org_rob)
            loss_acc=loss_acc

          
            acc1, acc5 = accuracy(outputs_sf, target_r, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            acc=acc+acc1
            loss_p=loss+loss_p
            loss_acc_p=loss_acc+loss_acc_p

            if i % 20 == 0:
                #print("acc=",acc/(i+1),"cl_loss=",loss/(i+1),"rl_loss=",rl_loss1/(i+1),"robustness=",rl_loss_real1/(i+1))
                print("acc=",acc/(i+1),"loss_acc=",loss_acc_p/(i+1),"loss_kl=",loss_p/(i+1))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)

def My_test(loader: DataLoader, model: torch.nn.Module, criterion, epoch: int, noise_sd: float,N,LAR,l_r):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        cl_loss1=0
        rl_loss1=0
        cl_loss=0
        rl_loss=0
        rl_loss_real=0
        rl_loss_real1=0
        # switch to train mode
        model.eval()
        acc=0
        # index_tensor_whole=get_res_index()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if i==0:
                    index_tensor_whole=get_res_index(inputs)
                # measure data loading time
                if l_r=='l':
                    random_number = 0
                else:
                    random_number = 1
                
                data_time.update(time.time() - end)
                inputs=inputs.cuda()
                input_size=len(inputs)
                targets = targets.cuda()
                #assert 1==0, print(inputs.max(),inputs.min())
                index_tensor=index_tensor_whole[random_number].reshape(1,3,32,32)

                inputs=get_low_res_data_mn(inputs,index_tensor,l_r,noise_sd)
                num=len(index_tensor)
                # num=1
                #targets=targets-1
                target_r=targets.reshape(len(targets),1)
                target_r = target_r.repeat(1,num)
                target_r=target_r.reshape(len(targets)*num)


                #inputs=get_upper_res_data(inputs)
                inputs=inputs.cuda()
                targets = targets.cuda()
                target_r=target_r.cuda()
                outputs = model(inputs)
                outputs_sf=F.softmax(outputs,dim=1)

                outputs_sf_resized=outputs_sf.reshape((input_size, num, 10))
                outputs_mean=outputs_sf_resized.mean(1)
                outputs_mean_clamped = torch.clamp(outputs_mean, min=0.000001)
                log_outputs_mean = torch.log(outputs_mean_clamped)
                criterion_acc=nn.NLLLoss()
                loss=criterion_acc(log_outputs_mean,targets.long())
                # acc1, acc5 = accuracy(log_outputs_mean, targets, topk=(1, 5))
                acc1, acc5 = accuracy(outputs_sf, target_r, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))

                # compute gradient and do SGD step
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                acc=acc+acc1
                if i % 20 == 0:
                    #print("acc=",acc/(i+1),"cl_loss=",loss/(i+1),"rl_loss=",rl_loss1/(i+1),"robustness=",rl_loss_real1/(i+1))
                    print("acc=",acc/(i+1),"cl_loss=",loss/(i+1))
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)


def My_train_clean(loader: DataLoader, model: torch.nn.Module, model_target: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,N,LAR,cut):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        loss_kl_p=0
        loss_acc_p=0
        m = Normal(torch.tensor([0.0]).to('cuda'),
             torch.tensor([1.0]).to('cuda'))
        model.train()
        model_target.eval()
        acc=0
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            num=N
            targets = targets.cuda()
            input_size=len(inputs)
            # augment inputs with noise
            new_shape = [input_size * num] #64*16
            new_shape.extend(inputs[0].shape)   #size 

            inputs = inputs.repeat((1, num, 1, 1)).view(new_shape)
            inputs=inputs.reshape(len(inputs),3,32,32).cuda()
            #inputs=torch.tensor(inputs).cuda().float()
            noise = torch.randn_like(torch.tensor(inputs), device='cuda')*noise_sd 
            inputs=inputs+noise 

            outputs = model(inputs)


            outputs_sf=F.softmax(outputs,dim=1)
            log_outputs_sf=torch.log(torch.clamp(outputs_sf, min=0.000001))
            pre_label=outputs_sf.argmax(dim=1)
            correct_indict=pre_label==targets
            outputs_rob=outputs_sf[correct_indict].max(dim=1).values
            outputs_rob=torch.clamp(outputs_rob, max=0.99)
            rob_loss=-m.icdf(outputs_rob).sum()/len(outputs_sf)
            
            criterion_acc=nn.CrossEntropyLoss()
            loss_acc=criterion_acc(outputs,targets)
            loss_kl=rob_loss
           
            # loss_kl=F.kl_div(log_outputs_mean, pre_outputs, reduction='mean')
            # loss_kl_rob=F.kl_div(log_outputs_sf, acc_outputs_mean, reduction='mean')
            loss = loss_acc + 0.5*loss_kl
            #loss_acc=(0.1*loss_acc +loss_kl_rob)
            # loss=criterion_acc(log_outputs,pre_targets.long())
          
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            acc=acc+acc1
            loss_kl_p=loss_kl+loss_kl_p
            loss_acc_p=loss_acc+loss_acc_p
            if i % 20 == 0:
                #print("acc=",acc/(i+1),"cl_loss=",loss/(i+1),"rl_loss=",rl_loss1/(i+1),"robustness=",rl_loss_real1/(i+1))
                print("acc=",acc/(i+1),"loss_acc=",loss_acc_p/(i+1),"loss_kl=",loss_kl_p/(i+1))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)