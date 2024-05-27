###### to evaluate the certified robustness

### 0.18 is the sigma value, "models/cifar10/resnet110/" is the folder for the ckpt  'data/certify/cifar10/resnet110/boost_ours/diff_smoothadv/sigma_0.18' is the log_file path
### to test different model trianed by gaussian and consistency, please change "--mode org" to "--mode consistency" and 0.18 to 0.36
CUDA_VISIBLE_DEVICES=0 python certify_resnet110_lowres_2.py cifar10 models/cifar10/resnet110/ 0.18 data/certify/cifar10/resnet110/org/sigma_0.18 --skip 5 --N 10000 --mode org



### to train the model on cifar10, please first download the pre-train model via xxx, which is trained on the original cifar10 images, and put this model on model/cifar_10/resnet110/pre_trained.pth.tar

# Then train the left and right model for dual randomized smoothing, you can also modifiy the pre_trained model path by --ckpt_path
CUDA_VISIBLE_DEVICES=0 python cifar_model_lower_2.py cifar10 cifar_resnet110 --sigma 0.18  --batch 256 --N 1 --lr 0.001 --out_dir models/cifar10/resnet110/  --pos r --scracth c_smoothadv_0.25 --mode org
CUDA_VISIBLE_DEVICES=1 python cifar_model_lower_2.py cifar10 cifar_resnet110 --sigma 0.18  --batch 256 --N 1 --lr 0.001 --out_dir models/cifar10/resnet110/  --pos l --scracth c_smoothadv_0.25 --mode org

### if you want to combine you own method with our DS_RS, we strongly recommeded using the proposed method to train a model on the original cifar10 image first and then fine-tuning DRS based on the pre-trained model

### to train the model on ImageNet, please first download the pre-train model via xxx, which is trained on the original ImageNet, and put this model on model/Imagenet/resnet50/pre_trained.pth.tar

# Then train the left and right model for dual randomized smoothing, you can also modifiy the pre_trained model path by --ckpt_path
CUDA_VISIBLE_DEVICES=0 python cifar_model_lower_2.py imagenet resnet50 --sigma 0.18  --batch 256 --N 1 --lr 0.00001 --out_dir models/cifar10/resnet50/  --pos r --scracth cohen_0.25 --mode org
CUDA_VISIBLE_DEVICES=1 python cifar_model_lower_2.py imagenet resnet110 --sigma 0.18  --batch 256 --N 1 --lr 0.00001 --out_dir models/cifar10/resnet50/  --pos l --scracth cohen_0.25 --mode org
