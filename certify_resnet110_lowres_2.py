# evaluate a smoothed classifier on a dataset

import numpy as np 
from datasets import get_normalize_layer
import argparse
import os
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core1_lowres_2 import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture as get_architecture1
from architectures_lowres import get_architecture
import numpy as np
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=2000, help="batch size")
parser.add_argument("--skip", type=int, default=15, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=200)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--mode", type=str, default='org', help="org or consistency training")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    base_classifier_dir=args.base_classifier+ 'noise_' + str(args.sigma) + '/' + str(args.mode)
    base_classifier_l=base_classifier_dir+'/'+'checkpoint_l.pth.tar'
    base_classifier_r=base_classifier_dir+'/'+'checkpoint_r.pth.tar'
    checkpoint_l = torch.load(base_classifier_l)
    checkpoint_r = torch.load(base_classifier_r)
    model_test_l=get_architecture1('cifar_resnet110', 'cifar10')
    model_test_r=get_architecture1('cifar_resnet110', 'cifar10')
    model_test_l.load_state_dict(checkpoint_l['state_dict'])
    model_test_r.load_state_dict(checkpoint_r['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(model_test_l,model_test_r, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    folder_path = os.path.dirname(args.outfile)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"dir created{folder_path}")
    with open(args.outfile, 'w') as file:
        file.write('')  #
        print(f"file created{args.outfile}")
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    ave=np.zeros(10000)
    acc=np.zeros(10000)
    acc_clean=np.zeros(10000)
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
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        # prediction, radius = smoothed_classifier.smoothed_certify(x, args.N0, args.N, args.alpha, args.batch)
        correct = int(prediction == label and radius>0)
        correct_clean = int(prediction == label)
        if prediction == label and radius>=0:
            ave[count]=radius
        acc[count]=correct
        acc_clean[count]=correct_clean
        count=count+1
        after_time = time()
        print('the clean acc; the rob acc; the acr',sum(acc_clean)/count,sum(acc)/count,sum(ave)/count)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
