import os 
import argparse
import time 
import datetime 
from torchvision import transforms, datasets
from datasets import get_dataset, DATASETS, get_num_classes
from core_org import Smooth 
from DRM_org import DiffusionRobustModel
import numpy as np

IMAGENET_DATA_DIR = "ImageNet"

def main(args):
    model = DiffusionRobustModel()

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='ImageNet/val', transform=transform)

    # Get the timestep t corresponding to noise level sigma
    # dataset = get_dataset('imagenet', 'test')
    target_sigma = args.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a

    # Define the smoothed classifier 
    smoothed_classifier = Smooth(model, 1000, args.sigma, t)

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    total_num = 0
    corrects = 0
    ave=np.zeros(10000)
    ave_org=np.zeros(10000)
    acc=np.zeros(10000)
    acc_clean=np.zeros(10000)
    count=0
    for i in range(len(dataset)):
        if i % args.skip != 0:
            continue

        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time.time()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        correct = int(prediction == label and radius>0)
        if prediction == label and radius>=0:
            ave[count]=radius
        acc[count]=correct
        # acc_clean[count]=correct_clean
        count=count+1
        print('the rob acc; the acr',sum(acc)/count,sum(ave)/count)
        after_time = time.time()

        corrects += int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        total_num += 1

        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, corrects, time_elapsed), file=f, flush=True)

    print("sigma %.2f accuracy of smoothed classifier %.4f "%(args.sigma, correct/float(total_num)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--skip", type=int, default=10, help="how many examples to skip")
    parser.add_argument("--N0", type=int, default=100, help="number of samples to use")
    parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--outfile", type=str, help="output file")
    args = parser.parse_args()

    main(args)