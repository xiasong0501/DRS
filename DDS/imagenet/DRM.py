import torch
import torch.nn as nn 
import timm
import torch.nn.functional as F
from architectures import get_architecture as get_architecture1
from architectures_lowres import get_architecture
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
def get_res_index():
    shape=(50176)
    false_tensor_A = torch.zeros(shape, dtype=torch.bool)
    false_tensor_B = torch.zeros(shape, dtype=torch.bool)
    count=0
    for i in range(50176):
        if i%2==0:
            if count%2==0:
                false_tensor_A[i]=True
                false_tensor_B[i+1]=True

            else:
                false_tensor_A[i+1]=True
                false_tensor_B[i]=True         
        if i%224==0:
            count=count+1
    false_tensor_A=false_tensor_A.reshape(1,224,224).repeat(3,1,1)
    false_tensor_B=false_tensor_B.reshape(1,224,224).repeat(3,1,1)
    index_tensor=torch.cat((false_tensor_A,false_tensor_B),dim=0).reshape(2,3,224,224)
    return index_tensor

def get_low_res_data(data,index_tensor):
    
    return_data=torch.tensor([]).cuda()
    # for i in range(len(data)):
    temp_data=torch.tensor([]).cuda()
    temp=torch.tensor([]).cuda()
    # for j in range(len(index_tensor)):
    index=index_tensor[0].reshape(1,3,224,224).repeat(len(data),1,1,1)
    data=data.cuda(non_blocking=True)
    index_tensor=index_tensor.cuda(non_blocking=True)
    # temp=data[:][index] 
    temp=data*index.cuda(non_blocking=True)
    temp=F.interpolate(temp.reshape(len(data),3,224,224),size=(224,112), mode='bilinear',align_corners=False)*2 

    ran=len(index_tensor)
    temp_data=temp
    while ran>1:
        ran=ran-1
        temp_data=torch.cat((temp,temp),dim=1)
    return_data=temp_data.reshape(len(index_tensor)*len(data),3,224,112)
    return return_data
# def get_low_res_data(data,index_tensor):
#     return_data=torch.tensor([]).cuda()
#     for i in range(len(data)):
#         temp_data=torch.tensor([]).cuda()
#         for j in range(len(index_tensor)):
#             temp=data[i][index_tensor[j]] 
#             temp_data=torch.cat((temp_data,temp),dim=0)
#         temp_data=temp_data.reshape(len(index_tensor),3,224,112)
#         return_data=torch.cat((return_data,temp_data),dim=0)
#     assert len(return_data)==len(data)*len(index_tensor),print(return_data.shape)
#     return return_data

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier_name="beit"):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("imagenet/256x256_diffusion_uncond.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        base_classifier_l='imagenet/ckpt/checkpoint_1_0.0lkl_rob.pth.tar'
        base_classifier_r='imagenet/ckpt/checkpoint_1_0.0rkl_rob.pth.tar'
        checkpoint_l = torch.load(base_classifier_l)
        checkpoint_r = torch.load(base_classifier_r)
        model_test_l=get_architecture('resnet50', 'imagenet')
        model_test_r=get_architecture('resnet50', 'imagenet')

        model_test_l.load_state_dict(checkpoint_l['state_dict'])
        model_test_r.load_state_dict(checkpoint_r['state_dict'])
        classifier_l=model_test_l
        classifier_r=model_test_r
        # classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
        classifier_l.eval().cuda()
        classifier_r.eval().cuda()

        self.classifier_l = classifier_l
        self.classifier_r = classifier_r
        # # Load the BEiT model

        index_tensor=get_res_index().cuda()
        self.index_tensor_l=index_tensor[0:1]
        self.index_tensor_r=index_tensor[1:]
    def forward(self, x, t,l_r):
        x_in = x * 2 -1
        imgs = self.denoise(x_in, t)
        # imgs=x_in
        imgs=(imgs+1)/2
        imgs = imgs.cuda()
        # imgs = F.interpolate(imgs,size=(224,224), mode='bilinear')

        
        imgs_l=get_low_res_data(imgs,self.index_tensor_l)
        imgs_l = F.interpolate(imgs_l,size=(224,224), mode='nearest')
        imgs_r=get_low_res_data(imgs,self.index_tensor_r)
        imgs_r = F.interpolate(imgs_r,size=(224,224), mode='nearest')    
        # assert 1==0, print(imgs_r.shape) 
        # imgs = torch.nn.functional.interpolate(imgs, (512, 512), mode='bicubic', antialias=True)

        
        with torch.no_grad():
            # imgs_l = F.interpolate(imgs_l,size=(224,224), mode='nearest')
            out_l = self.classifier_l(imgs_l)
            # imgs_r = F.interpolate(imgs_r,size=(224,224), mode='nearest')
            out_r = self.classifier_r(imgs_r)
        return out_l,out_r
    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out