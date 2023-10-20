# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from torch import Tensor
from torch.nn.utils import clip_grad

from mmcv.utils import (IS_NPU_AVAILABLE, TORCH_VERSION, _BatchNorm,
                        digit_version)
from ..dist_utils import allreduce_grads
from ..fp16_utils import LossScaler, wrap_fp16_model
from .hook import HOOKS, Hook
import numpy as np
try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    if IS_NPU_AVAILABLE:
        from torch.npu.amp import GradScaler
    else:
        from torch.cuda.amp import GradScaler
except ImportError:
    pass

def conv_operator(filename, kernel, in_channels=1):
    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, kernel.repeat(1, in_channels, 1, 1), stride=1, padding=1,)
    y = y.squeeze(0).numpy().transpose(1, 2, 0)

    return img, y

@HOOKS.register_module()
class offsetHook(Hook):
    """A hook get the conv offset.
    """

    def __init__(self,
                alpha,
                beta,
                gamma
                 ):
        """
        model: regist model
        label: GT label x,y,h,w
        alpha: scale factor of MIDLOSS
        beta: scale factor of DIV in NON_GT_LOC
        gamma:scale factor of DIV in GT_LOC

        """
        self.offsetMaps = {}
        self.features = {}
        self.offset_name = []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # print(333333333333333333333)
        self.sobel_x = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.pooling = torch.nn.AvgPool2d(2,stride=2)
        self.norm = nn.InstanceNorm2d(num_features= 1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
        self.numOfOutBounding = 0
    def get_stepmap(self,name):
        def hook(module, input, output):
            self.offsetMaps[name]  = output
            self.features[name] = input.detach().data
        return hook

    def getThegap(self,div,targetDiv):
        # print({"div":div})
        # print({'targetdiv':(targetDiv==0).all()})
        metric = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        return metric(div,targetDiv)
    def genMASK(self,labels,shape,h_feature,w_feature,ifWOLF=False):
        scale_factor = max(shape[0]//h_feature,shape[1]//w_feature)
        labels = labels/scale_factor
        labels[:,2:4] = torch.ceil(labels[:,2:4])
        labels = labels.int()
        # print(labels)
        mask = torch.zeros(1,h_feature,w_feature).cuda()
        for label in labels:
            if ifWOLF:
                mask[:,label[1]:label[3]+1,label[0]:label[2]+1]=label[3] + label[2] - label[1] - label[0]
            else:
                mask[:,label[1]:label[3]+1,label[0]:label[2]+1]=1

        return mask
    def getTheTarget(self,feature,c,h,w,factor):
        b,c_feature,h_feature,w_feature = feature.size()
        # targetDiv = torch.zeros(b,c//2,h,w).cuda()
        # GTmask = torch.zeros(b,1,h_feature,w_feature).cuda()
        # print(feature.size())
        # for i,(labels,meta) in enumerate(zip(runner.label,runner.shape)):
        #     # 最直接的一种方式，离GT近就分配大的DIV ，当时想的是参考电子产生电子场的形式(不合适，散度不适合抵消）羊群效应，把资源比较羊，我们想要所有羊进洞（GT），因此没有
        #     # 洞（GT）的地方，散度为0，有洞的地方散度小于0，特征图尺寸越小，散度应该也越小：损失函数里面应该体现这一点,
        #     # 大小表示缩放的程度
        #     # 。而在GT框里面的时候采取梯度的方式？有道理，GT大小不是整数的时候
        #     # print(meta)
        #     # print(labels)
        #     shape = meta['pad_shape']
        #     # for label in labels:
        #     GTmask[i] = self.genMASK(labels,shape,h_feature,w_feature)  
            # print(shape)
            # name = meta['filename'].split('/')[-1].split('.')[0]
        # if runner._iter%2000==0:
        #     # for label in labels:
        #     save_image(GTmask[0,:3,...],f'{runner._iter}_{name}_GTmask.png')
            # sssss
            # GTmask  = torch.zeros_like(feature)
            # 非GT区域乘以_GTmasks
            # _GTmask = 1 - GTmask

            # 距离越大，散度越大？（错）计算资源固定，因此没有GT的地方散度为0
            # p_0_x, p_0_y = torch.meshgrid(
            #         torch.arange(0, h, 1),
            #         torch.arange(0, w, 1))
            # p = torch.cat([p_0_x,p_0_y],1)

            # center=[label[0]+label[2]/2,label[1]+label[3]/2]
            # temp = p-center[:,None,None]
            # d2 = torch.sqrt(temp[0]**2 + temp[1]**2)*_GTmask
            # targetDiv+= temp/d2 *self.beta

        #     targetDiv+=(gradFeaure  * self.gamma * -1  )  #GT所处位置为负源
        # if h!=h_feature and w!=w_feature:
        #     GTmask = self.pooling(GTmask)
        # GTmask = GTmask.repeat(1,c//2,1,1)
        # if h!=h_feature and w!=w_feature:
        #     GTmask = self.pooling(GTmask)
        # return GTmask*self.beta
        # 计算GT框内梯度，此处需要特征图，GT框内散度小于0
        # print(f'feature:{(feature==0).all()}')
        # feature = feature * GTmask
        # print(f'feature:{(feature==0).all()}')
        # return factor*(-1)
        gradxFeature = F.conv2d(feature, self.sobel_x.repeat(c//2, c_feature*2//c, 1, 1), stride=1, padding=1,groups=c//2)
        gradyFeature = F.conv2d(feature, self.sobel_y.repeat(c//2, c_feature*2//c, 1, 1), stride=1, padding=1,groups=c//2)
        gradFeaure = torch.sqrt(gradxFeature**2+gradyFeature**2)
        # print(f'grad:{(gradFeaure==0).all()}')
        gradFeaure = nn.ReLU(inplace=True)(self.norm(gradFeaure))
        # gradFeaure = (torch.sigmoid(gradFeaure)-0.5)*2
        if h!=h_feature and w!=w_feature:
            gradFeaure = self.pooling(gradFeaure)
        gradFeaure = gradFeaure *factor
        #  # return (torch.sigmoid(targetDiv)-0.5)*2
        # return factor.repeat(1,c//2,1,1)*self.beta*(-1)
        return gradFeaure*self.beta * (-1)
    def genGTmask(self,runner,feature,c):
        b,c_feature,h_feature,w_feature = feature.size()
        GTmask = torch.zeros(b,1,h_feature,w_feature).cuda()
        # print(feature.size())
        for i,(labels,meta) in enumerate(zip(runner.label,runner.shape)):
            # 最直接的一种方式，离GT近就分配大的DIV ，当时想的是参考电子产生电子场的形式(不合适，散度不适合抵消）羊群效应，把资源比较羊，我们想要所有羊进洞（GT），因此没有
            # 洞（GT）的地方，散度为0，有洞的地方散度小于0，特征图尺寸越小，散度应该也越小：损失函数里面应该体现这一点,
            # 大小表示缩放的程度
            # 。而在GT框里面的时候采取梯度的方式？有道理，GT大小不是整数的时候
            shape = meta['pad_shape']

            GTmask[i] = self.genMASK(labels,shape,h_feature,w_feature)  
        GTmask = GTmask.repeat(1,c//2,1,1)
        return GTmask     
    def genWOLF(self,runner,feature,c ):
        b,c_feature,h_feature,w_feature = feature.size()
        mask = torch.zeros(b,1,h_feature,w_feature).cuda()
        # print(feature.size())
        for i,(labels,meta) in enumerate(zip(runner.label,runner.shape)):
            # 最直接的一种方式，离GT近就分配大的DIV ，当时想的是参考电子产生电子场的形式(不合适，散度不适合抵消）羊群效应，把资源比较羊，我们想要所有羊进洞（GT），因此没有
            # 洞（GT）的地方，散度为0，有洞的地方散度小于0，特征图尺寸越小，散度应该也越小：损失函数里面应该体现这一点,
            # 大小表示缩放的程度
            # 。而在GT框里面的时候采取梯度的方式？有道理，GT大小不是整数的时候
            shape = meta['pad_shape']
            centers = (labels[:,0:2]+labels[:,2:4])/2
            circle = labels[:,2:4] - labels[:,0:2]
            temp = torch.zeros_like(labels)
            for j in range(1,3):
                circle_ = circle*(torch.rand(1)+j)
                temp[:,0:2] = torch.clip(centers - circle_/2,0,10000)
                temp[:,2] = torch.clip(centers[:,0] + circle_[:,0]/2, 0, shape[1])
                temp[:,3] = torch.clip(centers[:,1] + circle_[:,1]/2, 0, shape[0])
                mask[i] += self.genMASK(temp,shape,h_feature,w_feature,ifWOLF=True)
        maskx = F.conv2d(mask, self.sobel_x, stride=1,  groups=1)
        masky = F.conv2d(mask, self.sobel_y, stride=1, groups=1)
        wolf = torch.cat([maskx,masky],dim=1)/4
        wolf = F.pad(wolf,(1,1,1,1),'constant',0)
        wolf = wolf.repeat(1,c//2,1,1)
        mask[mask>0] =1
        return wolf/4,mask.repeat(1,c,1,1)
    def getTheMap(self, padOffsetX, padOffsetY):
        b,c,h,w = padOffsetX.size()
        divmapx = F.conv2d(padOffsetX, self.sobel_x.repeat(c, 1, 1, 1), stride=1, groups=c)
        divmapy = F.conv2d(padOffsetY, self.sobel_y.repeat(c, 1, 1, 1), stride=1, groups=c)

        # print(divmapx)
        divmap = divmapy+ divmapx
        # print(f"h,w: {(h,w)} divmap:{divmap}")
        return divmap*2/(h+w)
    def getTheNum(self,offsetMaps,area):
        if area == 'label':
            pass
        elif area =='background':
            pass 
        elif area =='matter':
            pass
    def ifatROI(self,p_0,GTmask):
        factor = torch.ones_like(GTmask)
        b,c,h,w = GTmask.size()
        for i in range(b):
            for j in range(c):
                # print((GTmask[i,j,p_0[i,0,:,:].long(),p_0[i,1,:,:].long()]==1).type(torch.bool))
                factor[i,j][(GTmask[i,j,torch.clip(p_0[i,0,:,:].long(),0,h-1),\
                                    torch.clip(p_0[i,1,:,:].long(),0,w-1)]==1).type(torch.bool)]=0    #  GTmask  b 1 h, w,            p_0 b,c,h,w
        return factor
    def genPadOffset(self,offsetMap,iter):
        b,c,h,w = offsetMap.size()
        # 构建一个pad
        if iter<20000:
            h_ = h*iter//80000
            w_ = w*iter//80000
        else:
            h_ = h//4
            w_ = w//4
        padoffsetx = F.pad(offsetMap[:,::2,...],(1,1,0,0),'constant',0)
        padoffsetx = F.pad(padoffsetx,(0,0,1,0),'constant',h_)
        padoffsetx = F.pad(padoffsetx,(0,0,0,1),'constant',-1*h_)

        padoffsety = F.pad(offsetMap[:,1::2,...],(0,0,1,1),'constant',0)
        padoffsety = F.pad(padoffsety,(1,0,0,0),'constant',w_)
        padoffsety = F.pad(padoffsety,(0,1,0,0),'constant',-1*h_) 
        return padoffsetx,padoffsety
    def midloss(self, runner): 
        # condsider the divergence 散度 to metric P-M eqution
        # compute the gradience of the input 
        # loss1  num of kernel in the position of label must be lager
        # loss2  num of kernel besides the position of label must be smaller
        # loss3  define the important area: edge.  the num of kernel in the important area must be lager 
        loss = 0       
        # print(runner.offset_name)
        # print(runner.offsetMaps.keys())
        # 不应该从细粒度到粗粒度

        for i,name in enumerate(runner.offset_name[::-1]):
            offsetMap = runner.offsetMaps[name]
            feature = runner.features[name]
            b, c, h, w = offsetMap.size()
            # targetDiv,GTmask = self.getTheTarget(runner,feature,c,h,w)  #GTmask 尺度不一定对
            if i==0:
                p_0_x, p_0_y = torch.meshgrid(
                        torch.arange(0, h, 1),
                        torch.arange(0, w, 1))
                p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(b,1,1,1)   #(b,2,h,w)
                p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(b,1,1,1)    
                p_0 = torch.cat([p_0_x, p_0_y], 1).repeat(1,c//2,1,1).cuda()
            if p_0.size(3) != offsetMap.size(3):
                # _,_,h,w = p_0.size()
                p_0= 2*p_0.repeat(1,1,2,2).view(b,c,2, h//2, 2, w//2).permute(0,1,3, 2, 5, 4).contiguous(). \
                            reshape(b,c,h, w)   
                p_0[:,:,1::2,1::2] =  p_0[:,:,1::2,1::2] +1        
            p_0 = (p_0 + offsetMap).view(b, c, h, w)
            # print((offsetMap==0).all())
            # get the div of delta P0
            GTmask = self.genGTmask(runner,offsetMap,c) 
            # print(f'GTmask:{GTmask.size()}')
            # print(f'p0:{p_0.size()}')
            _,c_feature,h_feature,w_feature = feature.size()
            # DOING:羊群效应之牧羊犬 ：  草场最外围有一群被驱赶的羊，注意：如果当前位置不在羊圈，那么才开始移动。
            factor = self.ifatROI(p_0,GTmask) # 如果在ROI,targetOffset 归0吗
            # 生成包围圈狼：
            circle_wolf,wolfMASK = self.genWOLF(runner,offsetMap,c)
            # print(f"circle_wolf{circle_wolf.size()},wolfMASK{wolfMASK.size()}factor{factor.size()}")
            offsetMap = offsetMap*wolfMASK+ offsetMap*(1-wolfMASK) + circle_wolf
            wolf = torch.rand(factor.size(),device=offsetMap.device)
            wolf[wolf > 1-self.gamma]=0.5
            wolf[wolf < 1-self.gamma]=0
            # wolf = wolf*(h+w) //4
            # divmap = self.getTheMap(offsetMap[:,::2,...], offsetMap[:,1::2,...])             
            # print(factor.requires_grad)
            padOffsetX, padOffsetY = self.genPadOffset(offsetMap,runner.iter) 
            # print(f'offsetmap{offsetMap.size()}')
            # divmap = self.getTheMap(padOffsetX, padOffsetY)        #不在ROI区域内的卷积  计算其散度
            divmap = self.getTheMap(offsetMap[:,::2,...], offsetMap[:,1::2,...]) 
            divmap= F.pad(divmap,(1,1,1,1),'constant',0)
             #TODO   生成外圈恐慌 !外圈恐慌有些片面，外圈恐慌造成的
            # 结果是羊往中间跑，但是，我们要羊往羊圈跑，这样能保证羊会往最近的羊圈跑，形成的结果是，羊需要尽快跑到羊圈
            # TODO ：怎么安排内圈狼，狼会尽可能分散，待实现
            # TODO: 解决有些点偏移量过大导致出界无梯度的问题
   
       #不在ROI区域内的卷积  计算其散度
            targetDiv = ((wolf*factor)+(1-factor)*(1))
            # print(f'divmap:{divmap.size()}targetDiv,{targetDiv.size()}')

            # targetDiv= self.getTheTarget(feature.data,c,h,w,1-factor)  #GTmask 尺度不一定对
            # print(targetDiv[0].repeat(3,1,1)[:3,...].size())
            # print(f'targetmap:{targetDiv.size()}')
            # if i==len(runner.offset_name):

            # print(f"name:{name}  feature:{torch.any(torch.isnan(feature))} target:{torch.any(torch.isnan(targetDiv))}\
            #        div:{torch.any(torch.isnan(divmap))} offsetmap:{torch.any(torch.isnan(offsetMap))}")
            # print(f"-----------------------\ndivmap:{divmap} \ntarget:{targetDiv}   \n ********************")

            if i==len(runner.offset_name)-1:
                runner.target[offsetMap.device] = ((targetDiv+1)/2)[:,0,:,:].unsqueeze(1)
                count = torch.zeros(1,h,w)
                # print(p_0[0,0,...]<0)
                judge =torch.logical_or(torch.logical_or(p_0[0,0,...]<0 , p_0[0,0,...]>h-1),
                                        torch.logical_or( p_0[0,1,...]<0 , p_0[0,1,...]>w-1) )
                count[0][judge]=1
                self.numOfOutBounding = torch.sum(count)/h/w/c
                # runner.target[offsetMap.device] = GTmask
                # print(((1-factor)==0).all())
            # TODO:羊圈外的羊惩罚也必须有  ：狼是惩罚其一，羊没有去最近的羊圈的意图，所以要惩罚羊的步数，及offset要尽可能的小，
            # 尽可能快的进入羊圈：这个要保证羊圈足够大！即target的散度足够大。
            # TODO : 解决边界问题：
            # total  count of conv operation
            # total = h*w
            # numOfLabel += self.getTheNum(offsetMaps,'label')
            # numOfTrash += self.getTheNum(offsetMaps,'background')
            # numOfMatter += self.getTheNum(offsetMaps,'matter')
            # loss += self.getThegap(total, numOfLabel, numOfTrash, numOfMatter)
            # print(divmap)
            # print(targetDiv)
            # print((targetDiv<=0).all())
            # print(targetDiv-divmap)
            # print(f"divmap:{divmap.size()},factor:{factor.size()},targetDiv:{targetDiv.size()},")
            loss += self.getThegap(divmap,targetDiv)*0.5
            # print(loss)


        # return 0
        return loss
    def after_train_iter(self, runner):
        pass
        # k = self.midloss(runner) * self.alpha
        # print(f'midloss:{k.data} out of edge: {self.numOfOutBounding}')
        # # if runner._iter>20000:
        # runner.outputs['loss'] = runner.outputs['loss'] + k