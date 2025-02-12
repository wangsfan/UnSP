# -*- coding: utf-8 -*-
"""
@File  : self_paced_test.py
@Author: 林嚣张
@Date  : 2023/4/6 10:24
@Software  : pycharm
"""
import sys
sys.path.append('./etnet/')
sys.path.append('./mode2v/')
import argparse
import itertools
import os
import os.path as osp
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from torch.nn import L1Loss
import cv2
import kornia.geometry.transform as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader

from cedric_firenet.options.inference_options import set_inference_options
from e2v_utils import LossFn
from spade_e2v import Unet6 as Unet
from etnet.model.eitr.eitr import EITR
import time
from ETloss import perceptual_loss, temporal_consistency_loss

from modelpami import recon_model as model_arch
from modeltcsvt.networks import SSIR
from mode2v.e2v_model import CistaLSTCNet, CistaTCNet
torch.autograd.set_detect_anomaly(True)

class DataSet(data.Dataset):
    def __init__(self, path, train, seq_len, args, abs_e=True, crop_x=256, crop_y=256, img_ch=1, num_samples=7,
                 norm_e=False):
        self.train = train
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.num_bins = 5
        self.abs_e = abs_e
        self.norm_e = norm_e
        self.img_ch = img_ch
        self.w = 240
        self.h = 180
        self.flip_p = 0.5
        self.ang = 10

        self.num_evs = int(self.w * self.h)
        self.t, self.x, self.y, self.p = 2, 3, 4, 1
        self.seq_len = seq_len
        self.evs_len = seq_len * self.num_evs
        self.num_samples = num_samples
        self.args = args

        if train:
            trainpath = Path(path)
            fname = [itm for itm in os.scandir(trainpath) if itm.is_file()]

            files = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.csv']
            self.files = [list(g) for _, g in itertools.groupby(sorted(files), lambda x: x[0:-9])]
            self.imgs = [Path(itm).as_posix() for itm in fname if Path(itm).suffix == '.jpg']

        else:
            self.json_path = Path(path) / "event"
            self.img_path = Path(path) / "img"
            self.fnames = [o.name for o in os.scandir(self.json_path) if o.is_file()]
            self.fnames = [o for o in self.fnames if (self.img_path / (o.split('.')[0] + '.jpg')).is_file()]
            self.fnames = [o.split('.')[0] for o in self.fnames if int(o.split('_')[-1].split('.')[0]) > 2]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):

        fname = self.files[item]
        iname = [l for l in self.imgs if l.startswith(fname[0][:-9])]
        iname.sort()

        randx = np.random.randint(0, self.w - self.crop_x)
        randy = np.random.randint(0, self.h - self.crop_y)
        flipud = np.random.uniform(0, 1) > self.flip_p
        fliprl = np.random.uniform(0, 1) > self.flip_p
        angle = np.random.uniform(-self.ang, self.ang)
        self.args_ = [randx, randy, flipud, fliprl, angle]
        img_t = self.fix_time([int(itm[-16:-4]) for itm in iname])

        # 从csv文件中读取事件数据
        evs_stream = torch.zeros(0, 4).float()
        for i, fn in enumerate(fname):
            ev_ten = pd.read_csv(fn)
            ev_ten = torch.from_numpy(ev_ten.values[:, [self.t, self.x, self.y, self.p]]).float()
            evs_stream = torch.cat([evs_stream, ev_ten], 0)
            if evs_stream.shape[0] > self.evs_len:
                break

        evs_stream[:, 0] = torch.from_numpy(self.fix_time(evs_stream[:, 0].numpy()))
        gussion_stream = self.Gussion_group_sampling(evs_stream, samping_mode=1)

        if self.abs_e:
            evs_stream[:, 3] = 1

        im_idx = np.searchsorted(img_t, evs_stream[-1, 0].item(), side="rigth")
        image_ref = cv2.imread(iname[im_idx - 1], cv2.IMREAD_GRAYSCALE)

        img = image_ref / 255.0
        img = torch.from_numpy(img[None]).float()
        imgs = self.transform_koria(img, *self.args_)

        if self.img_ch == 3:
            imgs = imgs.repeat(self.img_ch, 1, 1)

        return gussion_stream, imgs

    def image_check(self, imgs):
        save_path = './images/check/image/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_len = imgs.shape[1]
        for i in range(image_len):
            now = datetime.now()
            timestr = now.strftime("%d_%H_%M_%S.png")
            image = np.array(imgs[0, i, :].reshape(128, 128, 3))
            plt.imsave(save_path + timestr, image)

    def fix_time(self, vect):
        ref = np.ones(len(vect) - 1)
        y_hat = np.diff(vect) / ref
        starts = np.where(y_hat < 0)[0]
        vect = np.asarray(vect)
        for i in range(len(starts)):
            vect[starts[i] + 1:] += vect[starts[i]]
        return vect

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def Gussion_single_sampling(self, evs_stream):
        events = torch.zeros([self.num_samples, self.seq_len, self.num_bins, self.crop_x, self.crop_y]).float()
        # 这里用self.seq_len来截断evs_stream，插入高斯采样
        evs_streams = self.gaussian_sampling_stream(event_stream=evs_stream, num_samples=self.num_samples)
        # events_sample: [num_samples, self.seq_len, 5, H, W]
        for nums, stream_in_list in enumerate(evs_streams):
            # 使evs_stream的长度能被self.seq_len整除
            stream_in_list = stream_in_list[0:int(stream_in_list.shape[0] / self.seq_len - 1) * self.seq_len, :]
            for i, ev_ten in enumerate(np.split(stream_in_list, self.seq_len)):
                try:
                    assert ev_ten.shape[1] == 4
                    ev_ten = self.ev2grid(ev_ten, num_bins=self.num_bins, width=self.w, height=self.h)
                    events[nums, i] = self.transform_koria(ev_ten, *self.args_)
                    if self.norm_e:
                        events[nums, i] = self.norm(events[nums, i])
                except:
                    print('eve_streams:', evs_streams)
                    print('stream_in_list:', stream_in_list)
                    print('ev_ten:', ev_ten)
        return events

    def Gussion_group_sampling(self, evs_stream, samping_mode = 2):
        # 初始化events
        output_events = torch.zeros([self.num_samples, self.seq_len, self.num_bins, self.crop_x, self.crop_y]).float()

        # 初始化事件体素
        sampled_events = torch.zeros([self.seq_len, 5, 128, 128])

        # 高斯选择15个事件点，通过self paced最终得到self.num_samples个事件帧
        # 高斯采样初始化，设置均值为0.35，标准差为0.02
        num_min = int(0.2 * self.w * self.h)
        num_max = int(0.5 * self.w * self.h)
        if samping_mode == 1:
            nums = np.random.choice(np.arange(num_min, num_max + 1), size=self.num_samples, replace=False) * 15
        elif samping_mode == 2:
            nums = np.linspace(num_min, num_max + 1, self.num_samples, dtype=int) * 15
        for sample_id, sample_num in enumerate(nums):
            # Take first n events from events
            sample = evs_stream[-nums[sample_id]:]
            for idx, evs in enumerate(np.split(sample, self.seq_len)):
                # Call ev2grid to aggregate events into voxels
                voxel_grid = self.ev2grid(evs, num_bins=self.num_bins, width=self.w, height=self.h)

                # transform for the events
                event_transform = self.transform_koria(voxel_grid, *self.args_)

                # Add voxel_grid to sampled_events
                if self.norm_e:
                    norm_events = self.norm(event_transform)
                    sampled_events[idx, :] = norm_events
                else:
                    sampled_events[idx, :] = event_transform

            output_events[sample_id, :] = sampled_events
            # output_events1 = output_events[:,0,:]

        return output_events

    def interval_samping(self, evs_stream):
        # 初始化events
        output_events = torch.zeros([self.num_samples, self.seq_len, self.num_bins, self.crop_x, self.crop_y]).float()

        # 初始化事件体素
        sampled_events = torch.zeros([self.seq_len, 5, 128, 128])

        num_min = int(0.2 * self.w * self.h)
        num_mid = int(0.4 * self.w * self.h)
        num_max = int(0.6 * self.w * self.h)

        nums_forward = np.random.choice(np.arange(0, num_min), size=self.num_samples, replace=False) * 15
        nums_backward = np.random.choice(np.arange(num_mid, num_max), size=self.num_samples, replace=False) * 15

        for sample_id in range(self.num_samples):
            sample = evs_stream[nums_forward[sample_id]: nums_backward[sample_id]]
            if sample.shape[0] % 15 != 0:
                add_num = sample.shape[0] % 15
                sample = sample[0:sample.shape[0] - add_num,:]
            for idx, evs in enumerate(np.split(sample, 15)):
                # Call ev2grid to aggregate events into voxels
                voxel_grid = self.ev2grid(evs, num_bins=self.num_bins, width=self.w, height=self.h)

                # transform for the events
                event_transform = self.transform_koria(voxel_grid, *self.args_)

                # Add voxel_grid to sampled_events
                if self.norm_e:
                    norm_events = self.norm(event_transform)
                    sampled_events[idx, :] = norm_events
                else:
                    sampled_events[idx, :] = event_transform

            output_events[sample_id, :] = sampled_events

        return output_events


    def gaussian_sampling_stream(self, event_stream, num_samples):
        # Set range of sampled event numbers
        num_min = int(self.down_bound * self.w * self.h * self.seq_len)
        num_max = int(self.up_bound * self.w * self.h * self.seq_len)

        # Create an array to save events_sample
        sampled_events = []

        # 定义区间
        left = 0
        right = int(self.w * self.h * self.seq_len * 0.35)

        for sample_id in range(num_samples):
            # Take first n events from events
            sample = event_stream[left:right, :]
            sampled_events.append(sample)

            # Sample number of events from Gaussian distribution
            n = np.random.randint(num_min, num_max)

            # 更新区间
            left = left + n
            right = right + n

        return sampled_events

    def gaussian_sampling(self, events, num_samples):
        """
        Apply Gaussian sampling to the events.
        Args:
            events (np.ndarray): The input events with shape (num_events, 4), including t, x, y, p.
            num_samples (int): The number of samples.
        Returns:
            sampled_events (list): A list of sampled event sets.
        """
        # Calculate number of events
        num_events = events.shape[0]

        # Set range of sampled event numbers
        num_min = int(0.3 * self.w * self.h)
        num_max = int(0.5 * self.w * self.h)

        # Create an array to save events_sample
        sampled_events = torch.zeros([num_samples, 5, 128, 128])

        for sample_id in range(num_samples):
            # Sample number of events from Gaussian distribution
            n = np.random.randint(num_min, num_max)

            # Take first n events from events
            sample = events[:n, :]

            # Call ev2grid to aggregate events into voxels
            voxel_grid = self.ev2grid(sample, num_bins=self.num_bins, width=self.w, height=self.h)

            # transform for the events
            event_transform = self.transform_koria(voxel_grid, *self.args_)

            # Add voxel_grid to sampled_events
            sampled_events[sample_id, :] = event_transform

        if self.norm_e:
            sampled_events = self.norm(sampled_events)

        return sampled_events

    def ev2grid(self, events, num_bins, width, height):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.

        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        assert (events.shape[1] == 4)
        assert (num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events[-1, 0]
            first_stamp = events[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
            ts = events[:, 0]
            xs = events[:, 1].long()
            ys = events[:, 2].long()
            pols = events[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                        * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                        + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(num_bins, height, width)

        return voxel_grid

    def transform(self, evs, img, randx, randy, flipud, fliprl, angle):
        img = img.transpose([1, 2, 0])  # channels last
        evs = evs.transpose([1, 2, 0])  # channels last
        if fliprl:
            evs = cv2.flip(evs, 1)
            img = cv2.flip(img, 1)
        if flipud:
            evs = cv2.flip(evs, 0)
            img = cv2.flip(img, 0)

        center = (img.shape[0] // 2, img.shape[1] // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        evs = cv2.warpAffine(evs, M, (evs.shape[1], evs.shape[0]))

        evs = evs.transpose([2, 0, 1])  # channels first
        img = img.transpose([2, 0, 1])  # channels first

        evs = evs[:, randy: randy + self.crop_size, randx: randx + self.crop_size]
        img = img[:, randy: randy + self.crop_size, randx: randx + self.crop_size]

        return evs, img

    def transform_koria(self, tensor, randx, randy, flipud, fliprl, angle):

        if flipud:
            tensor = torch.flip(tensor, dims=(0, 1))
        if fliprl:
            tensor = torch.flip(tensor, dims=(0, 2))

        # tensor = kornia.rotate(tensor, angle=angle, center=(tensor.shape[3], tensor.shape[2])
        center = torch.ones(1, 2)
        center[..., 0] = tensor.shape[2] / 2  # x
        center[..., 1] = tensor.shape[1] / 2  # y
        scale = torch.ones((1, 2))
        angle = torch.ones(1) * angle

        # M = KG.rotation(center, angle, scale)
        M = K.get_rotation_matrix2d(center, angle, scale)
        tensor = K.warp_affine(tensor[None], M, dsize=(self.h, self.w))[0]

        tensor = tensor[:, randy: randy + self.crop_y, randx: randx + self.crop_x]

        return tensor

    def norm(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)

        return events


def visualize_feature_map(feature_map, save_path=None):
    # 将 tensor 转为 numpy 数组
    np_feature_map = feature_map.cpu().detach().numpy()
    np_feature_map = np.transpose(np_feature_map, (1, 2, 0))

    # 进行归一化操作
    min_val = np_feature_map.min()
    max_val = np_feature_map.max()
    np_feature_map = (np_feature_map - min_val) / (max_val - min_val) * 255

    # 将归一化后的 feature map 转换为灰度图
    gray_img = np.uint8(np_feature_map[:, :, 0])

    # 使用cv2.applyColorMap将灰度图转换为热力图
    heatmap = cv2.applyColorMap(gray_img, cv2.COLORMAP_VIRIDIS)

    # 保存图像文件
    cv2.imwrite(save_path, heatmap)

def visualize_feature_map(train_pred, save_path=None):
    # 将 tensor 转为 numpy 数组
    np_train_pred = train_pred.cpu().detach().numpy()
    np_image = np.transpose(np_train_pred, (1, 2, 0))

    # 进行归一化操作
    min_val = np_image.min()
    max_val = np_image.max()
    np_feature_map = (np_image - min_val) / (max_val - min_val) * 255

    # 将归一化后的 feature map 转换为灰度图
    gray_img = np.uint8(np_feature_map[:, :, 0])

    # 保存图像文件
    cv2.imwrite(save_path, gray_img)

def dataset(args):
    trainpath = osp.join(args.root_dir)
    tr = DataSet(trainpath, train=True, seq_len=args.seq_len, args=args, abs_e=args.abs_e, num_samples=args.sample_nums,
                 norm_e=args.norm_e, crop_x=128, crop_y=128, img_ch=1)

    tr_loder = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=0)

    return tr_loder

def load_model(checkpoint, device):
    config = checkpoint['config']
    print(config)
    state_dict = checkpoint['state_dict']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

def main(args):
    loss_list = []
    # 参数初始化
    device = 'cuda:0'
    tr = dataset(args)
    lossETp = perceptual_loss()
    loss1 = L1Loss()
    # lossETt = temporal_consistency_loss()
    lossfn = LossFn(as_loss=True, to_cuda=device)
    # eitr_kwargs = {'num_bins':5, 'norm':0}
    # netG = EITR(eitr_kwargs).cuda()
    # netG = Unet().cuda() 
    # netG = SSIR()
    # network_data = torch.load("/home/thc/SSIR-main/ckpt/SSIR_e80.pth")
    # print('=> using pretrained model {:s}'.format(args.pretrained))
    # netG = torch.nn.DataParallel(netG).cuda()
    # model = model.cuda()
    # netG.load_state_dict(network_data)
    netG = CistaLSTCNet(image_dim=[128,128], base_channels=64, depth=5, num_bins=5)
    checkpoint = torch.load('/home/thc/V2E2V-main/pretrained/RecNet_cista-lstc.pth.tar', map_location=device)
    netG.load_state_dict(checkpoint['state_dict'], strict=True)
    # checkpoint = torch.load("/home/thc/EventHDR-main/EventHDR-main/model.pth") 
    # checkpoint = torch.load("/media/thc/Elements/unsp/etnet.pth") 
    # state_dict = checkpoint['state_dict']
    # netG.load_state_dict(state_dict)
 
    # netG = load_model(checkpoint, device)
    # netG.load_state_dict(torch.load(osp.join('model/SPADE_E2VID.pth'), map_location=device))
    netG = netG.to(device)
    netG.train()
    tr_param = netG.parameters()
    optimizerG = torch.optim.Adam(tr_param, args.lr)

    # training
    sample_nums = args.sample_nums
    seq_len = args.seq_len
    loss_for_train = []
    k = 0
    elapsed0 = 0
    elapsed1 = 0
    elapsed2 = 0
    start_time0 = time.time()
    for e in range(args.epochs):
        for i, (train_events, train_image) in enumerate(tr):
            # train_pred = 0
            # feat0 = 0
            # pred0 = 0
            with torch.no_grad():
            # #     # pred_tensors
                pred_tensors = torch.zeros([1, sample_nums, 1, 128, 128])
            # #     for idx in range(sample_nums):
            # #         # pred = torch.mean(train_events[0, idx, :], dim=[0, 1]).repeat(1, 3, 1, 1)
            # #         input_event = train_events[:, idx, 0, :3].detach()
            # #         pred_tensors[:, idx, :] = input_event

            # #     pred_tensors = pred_tensors.to(device)
            # #     train_image = train_image.to(device)
                pred_from_net = torch.zeros(pred_tensors.shape).to(device)
            train_events = train_events.to(device)
            train_image = train_image.to(device)
            for sample_idx in range(sample_nums):
                # 训练的输入
                train_pred = pred_tensors[:, sample_idx, :]
                optimizerG.zero_grad()
                stats = None

                # 输入训练数据，得到输出和state
                for seq_idx in range(15):
                    if seq_idx == 0:
                        prev_img = torch.zeros_like(train_image)  
                        state = None       
                    output, state = netG(train_events[:, sample_idx, seq_idx], prev_img, state)
                    prev_img = output.clone()
            # torch.cuda.synchronize()
            # start_time1 = time.time()
                    # if seq_idx % 2 == 0:
                    #     with torch.no_grad():
                    #          pred0 = netG(train_events[:, seq_idx, :])
                    # else:
            # start_time2 = time.time()
            
                    # pred0 = netG(train_events[:, sample_idx, seq_idx],  )
                pred_from_net[:, sample_idx, :] = output
            # imgs = imgs.repeat(img_ch, 1, 1)
            # torch.cuda.synchronize()
            # elapsed0 += time.time() - start_time2
            
            # feat0, pred0 = netG(train_events[:, 0, 7])
            # feat1, pred1 = netG(train_events[:, 1, 7]) 
            # feat2, pred2 = netG(train_events[:, 2, 7]) 
            # feat3, pred3 = netG(train_events[:, 3, 7]) 
            # feat4, pred4 = netG(train_events[:, 4, 7]) 
                        # if seq_idx == 0:
                        #     train_pred = pred0
                        # else :
                        #     train_pred = torch.cat((train_pred,pred0),dim = 0)
            # torch.cuda.synchronize()
            
            # elapsed1 += time.time() - start_time1  

            
            # loss_all = loss1(pred0, train_image)
            # loss_all = lossfn.improved_loss(train_pred, train_image[0, :].repeat(sample_nums, 1, 1, 1), k)

            # ssim_loss, mse_loss, lpips_loss, k, loss_all = lossfn.improved_loss(pred_from_net[0, :],
                                                                                # train_image[0, :].repeat(sample_nums, 1,
                                                                                                        #  1, 1), k)
            ssim_loss, mse_loss, lpips_loss, k, loss_all = lossfn.improved_loss(pred_from_net[0, :],train_image[0, :].repeat(sample_nums, 1, 1, 1), k)
            # sum_loss = loss_all

            # with torch.no_grad():
                # loss_for_train.append([sum_loss.item(), ssim_loss.item(),
                                    #    mse_loss.item(), lpips_loss.item()])

            print(
                f'all:{loss_all.item():.3f}'
                f'epoch:{e}')
            
            
            
            optimizerG.zero_grad()
            # loss_all.requires_grad_(True)  # 加入此句就行了 

            loss_all.backward()
            # loss_all.backward()
            # loss123.backward()
            # print('1')
                  

            optimizerG.step()
            # optimizerG.zero_grad()    # 清空梯度
            # loss_list.append(loss_all.item())


            # netG.eval()
            # torch.cuda.empty_cache()
            
            # torch.cuda.synchronize()
        elapsed0 = time.time() - start_time0
        print(elapsed0)
        

        if e == 40:
            # elapsed2 = time.time() - start_time0
            print('time/img', elapsed0, 'time/batch', elapsed1, 'time/epoch', elapsed2 )

            break
        # if e >= 40 and e % 10 == 0:
        #     path = os.path.join('./saved_models', 'interval_sampling')
        #     if os.path.exists(path) is None:
        #         os.makedirs(path)
        #     save_path = os.path.join(path, 'nums_' + str(sample_nums) + '_eps_' + str(e) + '.pth')
        #     torch.save(deepcopy(netG.state_dict()), save_path)
    torch.save(netG.state_dict(), osp.join('/media/thc/Elements/unsp/self_paced2_1124.pth'))
    print('Finish')

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        type=str,
                        default='/media/thc/Elements/unsp/evs_2',
                        help='Path to dir')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=15, help='Sequence length')
    parser.add_argument('--abs_e', type=bool, default=False, help='Use non-polarity format')
    parser.add_argument('--norm_e', type=bool, default=True, help='Normalize events')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--sample_nums', type=int, default=5, help='Number of sampling')
    set_inference_options(parser)
    args = parser.parse_args()
    main(args)
    # main(args)