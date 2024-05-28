
This is the code for paper "Mitigating the Curse of Dimensionality for Certified Robustness via Dual Randomized Smoothing"

##### pls download our trained model via https://drive.google.com/drive/folders/1IZFqJm58d9Sp54zVEE3tMcsXAouFri_l?usp=sharing
##### pls download the cifar10 dataset and put it into dir "dataset_cache"

###### to evaluate the certified robustness
CUDA_VISIBLE_DEVICES=0 python certify_resnet110_lowres_2.py cifar10 models/cifar10/resnet110/ 0.18 data/certify/cifar10/resnet110/org/sigma_0.18 --skip 5 --N 10000 --mode org

### 0.18 is the sigma value, "models/cifar10/resnet110/" is the folder for the ckpt  'data/certify/cifar10/resnet110/boost_ours/diff_smoothadv/sigma_0.18' is the log_file path
### To test different models trained by Gaussian and consistency, please change "--mode org" to "--mode consistency" and 0.18 to 0.36




### to train the model on cifar10, please first download the pre-train model via https://drive.google.com/file/d/1V1xsapLoaqXLYUHdKIbY3ridO53oUmuh/view?usp=sharing, which is trained on the original cifar10 images, and put this model on model/cifar_10/resnet110/pre_trained.pth.tar

# Then train the left and right model for dual randomized smoothing, you can also modify the pre_trained model path by --ckpt_path
CUDA_VISIBLE_DEVICES=0 python cifar_model_lower_2.py cifar10 cifar_resnet110 --sigma 0.18  --batch 256 --N 1 --lr 0.001 --out_dir models/cifar10/resnet110/  --pos r --scracth c_smoothadv_0.25 --mode org
CUDA_VISIBLE_DEVICES=1 python cifar_model_lower_2.py cifar10 cifar_resnet110 --sigma 0.18  --batch 256 --N 1 --lr 0.001 --out_dir models/cifar10/resnet110/  --pos l --scracth c_smoothadv_0.25 --mode org

### If you want to combine your own method with our DS_RS, we strongly recommend using the proposed method to train a model on the original cifar10 image first and then fine-tuning DRS based on the pre-trained model


if you find this code help, please consider cite:
@inproceedings{
xia2024mitigating,
title={Mitigating the Curse of Dimensionality for Certified Robustness via Dual Randomized Smoothing},
author={Song Xia and Yi Yu and Xudong Jiang and Henghui Ding},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=C1sQBG6Sqp}
}
