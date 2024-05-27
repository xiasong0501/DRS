# evaluate a smoothed classifier on a dataset

import numpy as np 
from datasets import get_normalize_layer
import argparse
import os
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core_imagenet_lowres_2 import Smooth
from time import time
import torch
import datetime
# from architectures import get_architecture
from architectures_lowres import get_architecture
import numpy as np
# from models.model import resnet110
# CUDA_VISIBLE_DEVICES=2 python certify_imagenet_lowres_2.py imagenet models/imagenet/resnet50/noise_0.25/low_res_2/checkpoint_1_0.25 data/certify/imagenet/resnet50/noise_0.25/our/N1 --skip 100
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=2000, help="batch size")
parser.add_argument("--skip", type=int, default=100, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    base_classifier_l=args.base_classifier+'lkl_rob.pth.tar'
    base_classifier_r=args.base_classifier+'rkl_rob.pth.tar'
    checkpoint_l = torch.load(base_classifier_l)
    checkpoint_r = torch.load(base_classifier_r)
    model_test_l=get_architecture('resnet50', 'imagenet')
    model_test_r=get_architecture('resnet50', 'imagenet')
#     model_test = torch.nn.DataParallel(model_test)
    # base_classifier=model_test
    model_test_l.load_state_dict(checkpoint_l['state_dict'])
    model_test_r.load_state_dict(checkpoint_r['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(model_test_l,model_test_r, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    ave=np.zeros(10000)
    ave_org=np.zeros(10000)
    acc=np.zeros(10000)
    count=0
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
 
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius,radius_org = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        # prediction, radius = smoothed_classifier.smoothed_certify(x, args.N0, args.N, args.alpha, args.batch)
        correct = int(prediction == label)
        if prediction == label:
            ave[count]=radius
            ave_org[count]=radius_org
        acc[count]=correct
        count=count+1
        after_time = time()
        print(sum(acc)/count,sum(ave)/count,sum(ave_org)/count)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
