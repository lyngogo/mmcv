# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import copy
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
from PIL import Image,ImageDraw,ImageFont
from mmcv.image import tensor2imgs



@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            # print('**********************')
            # print(outputs['loss'])
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.label = data_batch['gt_bboxes'].data[0]
            # print(data_batch['img'].data[0].size())
            # a = data_batch['img'].data[0].device

            # print(self.label)
            self.shape = data_batch['img_metas'].data[0]
            self.run_iter(data_batch, train_mode=True, **kwargs)
            # b = self.offsetMaps[list(self.offsetMaps.keys())[0]].device
            # print(f'{a}:___________{b}')
            # print(data_batch)
            self.call_hook('after_train_iter')
            # if self._iter<10 and self._iter%2==0:
            #     self.draw_somthing(data_batch)
            # elif self._iter<100 and self._iter%20==0:
            #     self.draw_somthing(data_batch)
            # elif self._iter<1000 and self._iter%200==0:
            #     self.draw_somthing(data_batch)                
            # elif self._iter%2000==0:
            #     self.draw_somthing(data_batch)
            
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
    @torch.no_grad()
    def draw_somthing(self, data, **kwargs):
        img_metas = data['img_metas'].data[0]
        labels = data['gt_bboxes'].data[0]
        a = torch.tensor(3.).cuda()
        target = self.target[a.device]
        # with torch.no_grad():
        #     result = self.model(return_loss=False, rescale=True, **data)
        batch_size = len(labels)
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        for i,(img_meta,label,img,targ) in enumerate(zip(img_metas,labels,imgs,target)):
            if i==0:
                name_ = img_meta['filename'].split('/')[-1].split('.')[0]
                # img = self.tensor2im(img.detach().data)
                # print(type(img))
                
                for lb in label:
                    # print(lb)
                    lb = lb.int()
                    cv2.rectangle(img, (int(lb[0]),int(lb[1])), (int(lb[2]),int(lb[3])), (250,0,0),5)
                # img_metas = data['img_metas'][0].data[0]
                # bbox = data['gt_bboxes'][0].data[0]
                # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                # assert len(imgs) == len(img_metas)
                # for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h_, w_, _ = img_meta['img_shape']
                img_show = img[:h_, :w_, :]
                PALETTE = None
                feature = self.offsetMaps
                h, w = feature["layer2_0"].size(2), feature["layer2_0"].size(3)
                h_ = h *8
                w_ = w* 8
                img_show = mmcv.imresize(img, (w_, h_))
                p_0_x, p_0_y = torch.meshgrid(
                    torch.arange(0, h, 1),
                    torch.arange(0, w, 1))
                p_0_x = p_0_x.cuda()
                p_0_y = p_0_y.cuda()
                temp = []
                for name in self.offset_name:
                    h, w = feature[name].size(2), feature[name].size(3)
                    img1 = copy.deepcopy(img_show)
                    for i in range(1):
                        offset_0 = feature[name][0,2*i:2*i+2,:,:].unsqueeze(0)
                        # print((offset_0==0).all())
                        stride_h = h_//8//h
                        stride_w = w_//8//w
                        _,_,H, W = offset_0.size()
                        offset_0 = offset_0.repeat(1,1, stride_h, stride_w).view(2,stride_h, H, stride_w, W).permute(0,2, 1, 4, 3).contiguous(). \
                            reshape(2,stride_h * H, stride_w * W)
                        if temp !=[]:
                            p_0_x = temp[0, 0,:,:] + offset_0[0,...]*stride_h
                            p_0_y = temp[0, 1,:,:] + offset_0[1, ...]*stride_w
                        else:
                            p_0_x = p_0_x + offset_0[0,...]*stride_h
                            p_0_y = p_0_y + offset_0[1,...]*stride_w

                        p_0_x = torch.flatten(p_0_x).view(1, 1, h_//8, w_//8)
                        p_0_y = torch.flatten(p_0_y).view(1, 1, h_//8, w_//8)
                        p_0 = torch.cat([p_0_x, p_0_y], 1)

                        for x, y in zip(torch.flatten(p_0[0, 0, :, :]), torch.flatten(p_0[0, 1, :, :])):
                            H =  int(8*x +4)
                            W =  int(8*y+ 4)
                            cv2.circle(img_show, (W, H), 3, (171, 84, 90), -2)
                            # draw.ellipse((W - 3, H - 3, W + 3 ,
                            #               H +  3), outline='red', width=5, fill=250)
                        # original_img.show()
                        # print(p_0)
                        # plot_img = draw_objs(img_show,
                        #                      boxes=predict_boxes,
                        #                      classes=predict_classes,
                        #                      scores=predict_scores,
                        #                      masks=predict_mask,
                        #                      category_index=category_index,
                        #                      line_thickness=3,
                        #                      font='arial.ttf',
                        #                      font_size=20)
                        # plt.subplot(2, 4, i + 1)
                        # print(img_show)
                        # plt.imshow(img_show)
                        img_show = np.array(img_show,dtype= np.uint8)
                        
                        img_show = Image.fromarray(img_show)
                        img_show.save(f"{self._iter}_"+name_+"_"+name+"_"+str(i) +".jpg")
                        img_show = copy.deepcopy(img1)
                        # break
                    temp = p_0
                # t = np.array(targ[:3,...].cpu(),dtype= np.uint8)
                # print((targ==0).all())
                t = Image.fromarray(self.tensor2im(targ.repeat(3,1,1)))
                t.save(f"{self._iter}_"+name_+"_"+str(i) +"target.jpg")
                    # plt.savefig("temp.jpg")
                    # plt.show()
                    # 保存预测的图片结果
                    # img_show.save("test_result.jpg")

    def tensor2im(self,input_image, imtype=np.uint8):
        """"
        Parameters:
            input_image (tensor) --  输入的tensor，维度为CHW，注意这里没有batch size的维度
            imtype (type)        --  转换后的numpy的数据类型
        """
        # print(input_image.size())
        mean = [0.485, 0.456, 0.406] # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
        std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor): # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            # for i in range(len(mean)): # 反标准化，乘以方差，加上均值
            #     image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
            # print(image_numpy)
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
            # print(image_numpy)        
        else:  # 如果传入的是numpy数组,则不做处理
            image_numpy = input_image
        return image_numpy.astype(imtype)
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
