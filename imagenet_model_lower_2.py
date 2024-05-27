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
from architectures import ARCHITECTURES, get_architecture
from torch.nn import CrossEntropyLoss,NLLLoss,MSELoss
from torch.optim import SGD,Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
from train_utils import AverageMeter, accuracy, init_logfile, log
import copy
import torch.nn as nn
import torch.nn.functional as F
from imagenet_train_lowres_2 import My_train,My_test,My_train_clean
from torchvision import transforms
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from folder2lmdb import ImageFolderLMDB
from progress.bar import Bar
import torch.multiprocessing as mp
from imagenet_dali import get_imagenet_iter_dali
# from data_loader_dali import create_dali_pipeline
# from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# CUDA_VISIBLE_DEVICES=0,1 python imagenet_model_lower_2.py imagenet resnet50 --sigma 0.25  --batch 64 --N 1 --lr 0.0001 --out_dir models/imagenet/resnet50/noise_0.25/low_res_2/kl  --pos l --scracth kl_rob
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
parser.add_argument("--pos", type=str, default='', help="the left_right model")
parser.add_argument("--scracth", type=str, default='', help="the left_right model")
parser.add_argument('--local_rank', default=0, type=int,help='node rank for distributed training')
parser.add_argument('--lmdb', default=0, type=int,help='node rank for distributed training')
parser.add_argument('--dali', default=0, type=int,help='node rank for distributed training')
args = parser.parse_args()

def return_imagenet():
    traindir = os.path.join('ImageNet_lmdb', 'train.lmdb')
    valdir = os.path.join('ImageNet_lmdb', 'val.lmdb')
    train_dataset = ImageFolderLMDB(
        traindir,
        transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))        
    val_dataset = ImageFolderLMDB(
        valdir,
        transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    return val_dataset,train_dataset
def get_data_set(type='train'):
    if type == 'train':
        return get_imagenet_iter_dali('train', 'ImageNet/train',args.batch,num_threads=4, crop=224, device_id=0, num_gpus=1)
    else:
        return get_imagenet_iter_dali('val', 'ImageNet/val', args.batch,num_threads=4, crop=224, device_id=0, num_gpus=1)
def dali_data(batch_size,workers):
        train_pipe = create_dali_pipeline(batch_size=batch_size,num_threads=workers,device_id=0,seed=12,
                                          data_dir='ImageNet/train',crop=224,size=256,dali_cpu=False,shard_id=0,num_shards=1,is_training=True)
        train_pipe.build()
        train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",last_batch_policy=LastBatchPolicy.PARTIAL,auto_reset=True)

        val_pipe = create_dali_pipeline(batch_size=batch_size,num_threads=workers,device_id=0,seed=12,data_dir='ImageNet/val',
                                        crop=224,size=256,dali_cpu=False,shard_id=0,num_shards=1,is_training=False)
        val_pipe.build()
        val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader",last_batch_policy=LastBatchPolicy.PARTIAL,auto_reset=True)
        return train_loader,val_loader
if __name__ == "__main__":
      mp.set_start_method('spawn')
      torch.cuda.set_device(args.local_rank)
      model_test=get_architecture(args.arch, args.datasets)
      
      checkpoint_pre = torch.load('models/imagenet/resnet50/noise_0.00/checkpoint_1_0.0lkl_rob_frombi.pth.tar')
      if args.train_method != 'clean':
          model_test.load_state_dict(checkpoint_pre['state_dict'])

      modified_network=model_test
      l_r=args.pos
      scracth=args.scracth
      ######
      noise_sd=args.sigma       
      criterion_my =CrossEntropyLoss().cuda()
      optimizer=Adam(model_test.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
      print('data_set is ready')
      #train the rec_network###

      ###################################
      start_epoch=0
      num_epoch=15
      scheduler = MultiStepLR(optimizer, milestones=[5 ,10], gamma=0.1)
      before = time.time()
      after = time.time()
      acc=np.zeros([100])
      loss=np.ones([100])
      fine_name='ds_rs_'+str(args.N)+'_'+str(noise_sd)+l_r+scracth+'.txt'
      logfilename = os.path.join(args.out_dir, fine_name)#path to save log file 
      model_name='checkpoint_'+str(args.N)+'_'+str(noise_sd)+l_r+scracth+'.pth.tar'
      folder = os.path.exists(args.out_dir)
      if not folder:                   
        os.makedirs(args.out_dir)
        print ("---  new folder...  ---")
      if args.lmdb==1:
            test_dataset,train_dataset=return_imagenet()
      if args.dali==1:
            train_loader = get_data_set('train')
            val_loader = get_data_set('test')
      else: 
            train_dataset = get_dataset(args.datasets, 'train')
            test_dataset = get_dataset(args.datasets, 'test')
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=True)
            train_loader = DataLoaderX(train_dataset, shuffle=(train_sampler is None), batch_size=args.batch,sampler=train_sampler, num_workers=4,pin_memory=True,persistent_workers=True)
            test_loader = DataLoaderX(test_dataset, shuffle=(test_sampler is None), batch_size=args.batch,sampler=test_sampler, num_workers=4,pin_memory=True,persistent_workers=True)


#       init_logfile('model/cifar10/resnet110/noise_0.25/log_0.25_SGD_rob.txt', "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")#
      for i in range(num_epoch):
            epoch=start_epoch+i
            if args.dali==0:
                train_sampler.set_epoch(i)
                test_sampler.set_epoch(i)
            # train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), batch_size=args.batch,sampler=train_sampler, num_workers=8,pin_memory=True,persistent_workers=True)
            before = time.time()
            scheduler.step(epoch)
            lr=scheduler.get_lr()
            if args.train_method =='ours':       
                train_loss, train_acc = My_train(train_loader, modified_network, criterion_my, optimizer, epoch, (noise_sd),args.N,args.LAR,l_r,0,args.dali)     
                test_loss, test_acc = My_test(val_loader, modified_network, criterion_my, epoch, (noise_sd),args.N,args.LAR,l_r,args.dali)
                # test_loss, test_acc=train_loss, train_acc
                if i >=0 and i%1==0:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': 'imagenet',
                        'state_dict': modified_network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(args.out_dir, model_name)) #path to save model    
                
                # test_loss, test_acc = My_test(test_loader, modified_network, criterion_my, epoch, (noise_sd),args.N,args.LAR,l_r,0)     
            torch.cuda.empty_cache()
            ### training for cleaning model
            acc[epoch]=test_acc
            loss[epoch]=test_loss
            if test_acc==0:
                test_loss, test_acc=train_loss, train_acc

            after = time.time()
            print('learning rate=',lr,'acc=',test_acc)
            if test_acc!=0:
                log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                    epoch, str(datetime.timedelta(seconds=(after - before))),
                 scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
