import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

torch.manual_seed(0)
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
    return_data=torch.tensor([])
    for i in range(len(data)):
        temp_data=torch.tensor([])
        for j in range(len(index_tensor)):
            temp=data[i][index_tensor[j]] 
            temp_data=torch.cat((temp_data,temp),dim=0)
        temp_data=temp_data.reshape(len(index_tensor),3,224,112)
        return_data=torch.cat((return_data,temp_data),dim=0)
    assert len(return_data)==len(data)*len(index_tensor),print(return_data.shape)
    return return_data

def get_upper_res_data(data):
    data_len=len(data)
    return_data=torch.zeros([data_len,3,32,32])
    # for i in range(data_len):
    for j in range(16):
        return_data[:,:,:,2*j]=data[:,:,:,j]
        return_data[:,:,:,2*j+1]=data[:,:,:,j]
    return return_data
    """A smoothed classifier g """

def get_noise_batch(batch,index_tensor,this_batch_size,sigma):
    batch=get_low_res_data(batch.cpu(),index_tensor)
    batch = batch.repeat((this_batch_size, 1, 1, 1))
    # noise = torch.randn_like(batch, device='cpu') * 0
    # noised_batch=batch + noise
    # noised_batch = F.interpolate(noised_batch,size=(32,32), mode='bilinear')
    #noised_batch=get_upper_res_data(noised_batch)
    # noised_batch=noised_batch.cuda()
    return batch.cuda()
    """A smoothed classifier g """

    """A smoothed classifier g """
# Adapted from https://github.com/locuslab/smoothing/blob/master/code/core.py
class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, t: int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.t = t 

    def return_radius(self,counts_estimation,cAHat,n,alpha):
        nA = counts_estimation[cAHat].item()
        counts_selection=np.delete(counts_estimation,cAHat)
        cBHat = counts_selection.argmax().item()
        mul_par=int(100000/n)
        # mul_par=1
        pABar = self._lower_confidence_bound(nA*mul_par, n*mul_par, alpha)  #certify这一段计算radius的方法是可以修改的
        nB=counts_estimation[cBHat].item()
        if pABar < 0.5:
            return cAHat, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
        return cAHat, radius
        
    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # self.base_classifier_r.eval()
        # draw samples of f(x+ epsilon)
        counts_selection0,counts_selection1 = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = (counts_selection0+counts_selection1).argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation0,counts_estimation1 = self._sample_noise(x, n, batch_size)
       # assert 1==0,print(counts_selection0,counts_estimation0)
        # use these samples to estimate a lower bound on pA
        cAHat0 , radius0 =self.return_radius(counts_estimation0,cAHat,n,alpha)
        cAHat1 , radius1 =self.return_radius(counts_estimation1,cAHat,n,alpha)
        radius=(radius0+radius1)/(2**0.5)
        # radius=radius0
        # cAHat1 , radius =self.return_radius(counts_estimation1+counts_estimation0,cAHat,n*2,alpha)
        
       # radius=radius0+radius1+radius2+radius3
        # radius=(radius0+radius1)/(2**0.5)
        # if radius< 0.00001:
        #     return -1, radius,radius_org
        # else:
        #     return cAHat, radius,radius_org
        # if radius< 0.00001:
        #     return -1, radius
        # else:
        return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            counts0 = np.zeros(self.num_classes, dtype=int)
            counts1 = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch=x.reshape(1,3,224,224)
                # index_tensor=get_res_index()
                # index_tensor_l=index_tensor[0:1]
                # index_tensor_r=index_tensor[1:]
                # noised_batch_l=get_noise_batch(batch,index_tensor_l,this_batch_size,self.sigma)
                # noised_batch_r=get_noise_batch(batch,index_tensor_r,this_batch_size,self.sigma)
                # assert 1==0, print(imgs_r.shape) 
                batch = x.repeat((this_batch_size, 1, 1, 1))
                predictions1,predictions0 = self.base_classifier(batch, self.t,'l')
                # predictions0=(predictions0+predictions1).argmax(1)
                # predictions1=predictions0
                predictions0=predictions0.argmax(1)
                predictions1=predictions1.argmax(1)
                
                
                # predictions1=predictions0
                
                # predictions1 = self.base_classifier(batch, self.t,'r').argmax(1)
                # predictions0=predictions1
                counts0 += self._count_arr(predictions0.cpu().numpy(), self.num_classes)
                counts1 += self._count_arr(predictions1.cpu().numpy(), self.num_classes)

           # assert 1==0,print(counts0)
            return counts0,counts1
                # batch = x.repeat((this_batch_size, 1, 1, 1))
                # predictions = self.base_classifier(batch, self.t).argmax(1)
                # counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            # return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]