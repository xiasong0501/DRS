import numpy as np 
from datasets import get_normalize_layer
import argparse
from datasets import get_dataset, DATASETS, get_num_classes
import datetime
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
from numpy import random
import torch
# from architectures_lowres import ARCHITECTURES, get_architecture
from architectures import ARCHITECTURES, get_architecture
from torch.nn import CrossEntropyLoss,NLLLoss,MSELoss
from torch.optim import SGD,Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
from train_utils import AverageMeter, accuracy, init_logfile, log
import copy
import torch.nn as nn
import torch.nn.functional as F
from cifar_train_lowres_2 import My_train,My_test,My_train_clean,My_train_kl
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='train resnet110 on cifar10')
parser.add_argument("datasets", type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument("--sigma", type=float, default=0.25, help="noise hyperparameter")
parser.add_argument("--lr", type=float, default=0.01, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=128, help="batch size")
parser.add_argument("--N", type=int, default=4)
parser.add_argument("--N0", type=int, default=4, help="number of samples for map network")
parser.add_argument("--LAR", type=int, default=50, help="the robustness and accuracy rate")
parser.add_argument("--train_method", type=str, default='ours', help="the training method")
parser.add_argument("--out_dir", type=str, default='models/cifar10/resnet110/noise_0.25/low_res_2/kl', help="noise hyperparameter")
parser.add_argument("--ckpt_path", type=str, default='models_old/cifar10/resnet110/pre_trained.pth.tar', help="noise hyperparameter")
parser.add_argument("--mode", type=str, default='org', help="org or consistency training")
parser.add_argument("--pos", type=str, default='', help="the left_right model")
parser.add_argument("--scracth", type=str, default='', help="scracth or not")
args = parser.parse_args()

if __name__ == "__main__":
      model_test=get_architecture(args.arch, args.datasets)
      model_target=get_architecture(args.arch, args.datasets)
      out_dir=args.out_dir + 'noise_' + str(args.sigma) + '/' + str(args.mode)
      checkpoint_pre = torch.load(args.ckpt_path)

     
      if args.train_method != 'from_scracth':
          model_test.load_state_dict(checkpoint_pre['state_dict'])
    #   model_target.load_state_dict(checkpoint_org['state_dict'])
      modified_network=model_test
      l_r=args.pos
      scracth=args.scracth
      ######
      noise_sd=args.sigma
      train_dataset = get_dataset(args.datasets, 'train')
      test_dataset = get_dataset(args.datasets, 'test')
      train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,num_workers=2,pin_memory=True)
      test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch,num_workers=2,pin_memory=True)
      
      num_epoch=40
      criterion_my =CrossEntropyLoss().cuda()
      optimizer=Adam(model_test.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

      scheduler = MultiStepLR(optimizer, milestones=[25 ,35], gamma=0.1)
      if not os.path.exists(out_dir):

          os.makedirs(out_dir)
          print(f"dir created:{out_dir}")
      else:
          print(f"dir exists:{out_dir}")
      print('data_set is ready')
      #train the network###

      start_epoch=0
      
      before = time.time()
      after = time.time()
      acc=np.zeros([100])
      loss=np.ones([100])
      file_name='ds_rs_'l_r+'.txt'
      logfilename = os.path.join(out_dir, file_name)#path to save log file 
      model_name='checkpoint_'+l_r+'.pth.tar'
      if not os.path.exists(logfilename):
          with open(logfilename, 'w') as file:
              file.write('')  # 
          print(f"fiel created:{logfilename}")
      folder = os.path.exists(out_dir)
      if not folder:                  
        os.makedirs(out_dir)
        print ("---  new folder...  ---")

      for i in range(num_epoch):
            epoch=start_epoch+i
            before = time.time()
            scheduler.step(epoch)
            lr=scheduler.get_lr()
            #### original training for our method 
            test_loss, test_acc = My_test(test_loader, modified_network, criterion_my, epoch, (noise_sd),args.N,args.LAR,l_r)
            if args.train_method =='ours':
              if args.mode=='org':
                train_loss, train_acc = My_train(train_loader, modified_network,model_target, criterion_my, optimizer, epoch, (noise_sd),args.N,args.LAR,l_r,0)
              if args.mode=='consistency':
                train_loss, train_acc = My_train_kl(train_loader, modified_network,model_target, criterion_my, optimizer, epoch, (noise_sd),args.N,args.LAR,l_r,0)
            if args.train_method =='clean':
                train_loss, train_acc = My_train_clean(train_loader, modified_network,model_target, criterion_my, optimizer, epoch, (noise_sd),args.N,args.LAR,0)           
            

            acc[epoch]=test_acc
            loss[epoch]=test_loss
            after = time.time()
            print('learning rate=',lr,'acc=',test_acc)
            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))),
                scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
        
            if i >=0 and i%2==0:
                 torch.save({
                    'epoch': epoch + 1,
                    'arch': 'cifar10',
                    'state_dict': modified_network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(out_dir, model_name)) #path to save model
