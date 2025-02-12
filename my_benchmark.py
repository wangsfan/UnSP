# -*- coding: utf-8 -*-
"""
@File  : my_benchmark.py
@Author: 林嚣张
@Date  : 2023/5/28 16:38
@Software  : pycharm
"""
import sys
sys.path.append('./etnet/')
sys.path.append('./mode2v/')
import argparse
import glob
import os
import os.path as osp
import random

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from thop import profile
from modelpami import recon_model as model_arch
from mode2v.e2v_model import CistaLSTCNet, CistaTCNet


from e2v_utils import LossFn, IntensityRescaler


class testset:
    def __init__(self, root_dir, ev_rate, norm_e, e_abs, num_freams):

        self.num_frm = num_freams
        self.norm_e = norm_e
        self.e_abs = e_abs
        self.bs = 1
        self.h = 180
        self.w = 240
        self.img_num = 0
        self.bins = 5
        self.root_dir = root_dir + '/'
        self.events_file = 'events.txt'
        self.img_file = 'images.txt'
        if ev_rate is None:
            self.num_events = int(random.uniform(0.2, 0.5) * self.h * self.w)
        else:
            self.num_events = int(0.35 * self.h * self.w)
        self.args = [self.bins, self.w, self.h]  # num_bins, width, height
        self.iterator = pd.read_csv(self.root_dir + self.events_file, header=None,
                                    delimiter=' ',
                                    names=['t', 'x', 'y', 'p'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16},
                                    engine='c',
                                    index_col=False)

        self.img_metadata = pd.read_csv(self.root_dir + self.img_file,
                                        delimiter=' ',
                                        header=None, names=['t', 'fname'],
                                        index_col=False)
        self.time_stamps = pd.read_csv(self.root_dir + 'time_stamp.csv')

        max_time_stamp = self.time_stamps.iloc[num_freams + 1][1]
        self.iterator = self.iterator[:max_time_stamp + 1]

    def getitem(self, item):

        first_time_stamp = self.time_stamps.iloc[item][1]
        new_time_stamp = self.time_stamps.iloc[item + 1][1]
        event_tensor = self.iterator.values[first_time_stamp:new_time_stamp]
        num_evs = event_tensor.shape[0] // self.num_events
        if num_evs == 0:
            num_evs = 1
        event_tensor = np.array_split(event_tensor, num_evs, axis=0)
        evs = torch.zeros(num_evs, self.bins, self.h, self.w)
        img_name = self.root_dir + self.img_metadata.fname[item]

        with torch.no_grad():
            for i in range(num_evs):
                ev_ten = torch.from_numpy(event_tensor[i])
                if self.e_abs:
                    ev_ten[:, 3][ev_ten[:, 3] == -1] = 1
                    ev_ten[:, 3][ev_ten[:, 3] == 0] = 1
                evs[i] = self.events_to_voxel_grid_pytorch(ev_ten, *self.args)
                if self.norm_e:
                    evs[i] = self.norm(evs[i])
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) / 255.0
            img = torch.from_numpy(img).float()

        return evs, img

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

    def events_to_voxel_grid_pytorch(self, events, num_bins, width, height):
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
            events_torch = events

            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height,
                source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(
                dim=0,
                index=xs[valid_indices] + ys[valid_indices] * width + (tis_long[valid_indices] + 1) * width * height,
                source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(num_bins, height, width)

        return voxel_grid

E2VID_dict = {'num_bins': 5,
              'skip_type': 'sum',
              'recurrent_block_type': 'convlstm',
              'num_encoders': 3,
              'base_num_channels': 32,
              'num_residual_blocks': 2,
              'use_upsample_conv': True,
              'norm': 'none'}


def arch(model_path, n):
    global netG
    device = 'cuda:0'
    if n == 0:
        from org_e2vid.model import E2VIDRecurrent
        # netG = E2VIDRecurrent(E2VID_dict)
        # netG.load_state_dict(torch.load(model_path, map_location=device))
        raw_model = torch.load(model_path)
        netG = E2VIDRecurrent(raw_model['model'])
        netG.load_state_dict(raw_model['state_dict'])

    elif n == 1:
        from org_e2vid.model import E2VIDRecurrent
        raw_model = torch.load(model_path)
        netG = E2VIDRecurrent(raw_model['model'])
        netG.load_state_dict(raw_model['state_dict'])

    elif n == 2:
        from spade_e2v import Unet6 as Unet
        netG = Unet()
        netG.load_state_dict(torch.load(model_path, map_location=device))

    elif n == 3:
        from cedric_firenet.utils.loading_utils import load_model
        netG = load_model(model_path)

    elif n == 4:
        from spade_e2v import Unet6 as Unet
        netG = Unet()
        netG.load_state_dict(torch.load(model_path, map_location=device))

    elif n == 5:
        from spade_e2v import Unet6 as Unet
        netG = Unet()
        netG.load_state_dict(torch.load(model_path, map_location=device))

    elif n == 6:
        from etnet.model.eitr.eitr import EITR
        eitr_kwargs = {'num_bins':5, 'norm':0}
        netG = EITR(eitr_kwargs)
        state_dict = torch.load(model_path, map_location=device) 
        netG.load_state_dict(state_dict)
    elif n == 7:
        from etnet.model.eitr.eitr import EITR
        eitr_kwargs = {'num_bins':5, 'norm':0}
        netG = EITR(eitr_kwargs)
        checkpoint = torch.load(model_path, map_location=device) 
        state_dict = checkpoint['state_dict']
        netG.load_state_dict(state_dict)
    elif n==8:
        checkpoint = torch.load("/home/thc/EventHDR-main/EventHDR-main/model.pth") 
        ck1 = torch.load(model_path) 
        netG = load_model1(checkpoint, ck1, device)
    elif n==9:
        netG = CistaLSTCNet(image_dim=[128,128], base_channels=64, depth=5, num_bins=5)
        checkpoint = torch.load(model_path, map_location=device)
        netG.load_state_dict(checkpoint['state_dict'], strict=True)
    elif n==10:
        netG = CistaLSTCNet(image_dim=[128,128], base_channels=64, depth=5, num_bins=5)
        checkpoint = torch.load(model_path, map_location=device)
        netG.load_state_dict(checkpoint, strict=True)


    else:
        print('error')

    return netG.eval().cuda()

def load_model1(checkpoint, ck1, device):
            config = checkpoint['config']
            # state_dict = torch.load(ck1['state_dict'])
            logger = config.get_logger('test')

            # build model architecture
            model = config.init_obj('arch', model_arch)
            logger.info(model)
            if config['n_gpu'] > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(ck1)

            model = model.to(device)
            model.eval()
            return model


def create_dataframe(data_dir, models_list):
    models = [dir.split('/')[-1].split('.')[0] for dir, n in models_list]
    models = [val for i, val in enumerate(models) for _ in (0, 1)]
    datasets = [dir.split('/')[-1] for dir, n in data_dir]

    for i in range(len(models)):
        if i % 2 == 0:
            models[i] = models[i] + '_norm'

    df_full = {}
    for i in ['mse', 'ssim', 'lpips', 'tc']:
        exec(f'{i} = pd.DataFrame(columns=models + datasets)')
        df_full[f'{i}'] = eval(f'{i}')

    df_10 = {}
    for i in ['mse', 'ssim', 'lpips', 'tc']:
        exec(f'{i} = pd.DataFrame(index=datasets, columns=models)')
        df_10[f'{i}'] = eval(f'{i}')

    return df_full, df_10


def create_dataframe_all(data_dir, models_list):
    models = [dir.split('/')[-1].split('.')[0] for dir, n in models_list]
    models = [val for i, val in enumerate(models) for _ in (0, 1)]
    datasets = [dir.split('/')[-1] for dir, n in data_dir]

    for i in range(len(models)):
        if i % 2 == 0:
            models[i] = models[i] + '_norm'

    df_full = {}
    for i in ['mse', 'ssim', 'lpips', 'tc']:
        exec(f'{i} = pd.DataFrame(index=datasets, columns=models)')
        df_full[f'{i}'] = eval(f'{i}')

    return df_full


def main(arg, model_path, output_path):
    torch.cuda.empty_cache()
    rescale = IntensityRescaler()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lossfn = LossFn(as_loss=False, to_cuda='cuda:0')
    ev_rate = 0.35
    # ev_rate = random.uniform(0.2, 0.5)
    root_dir = arg.root_dir

    models_list = list()

    # models_list.append([osp.join(root_dir, 'model/E2VID.pth.tar'), 0])
    # models_list.append([osp.join(root_dir, 'model/E2VID_lightweight.pth.tar'), 1])
    # models_list.append([osp.join(root_dir, 'model/SPADE_E2VID.pth'), 2])
    # models_list.append([osp.join(root_dir, model_path), 3])
    # models_list.append([osp.join(root_dir, 'my_test/model/SPADE_With_Uncertainty_full.pth'), 4])
    # models_list.append([osp.join(root_dir, "/media/thc/Elements/unsp/etnet.pth"), 7])
    # models_list.append([osp.join(root_dir, model_path), 6])
    models_list.append([osp.join(root_dir, '/home/thc/V2E2V-main/pretrained/RecNet_cista-lstc.pth.tar'), 9])
    models_list.append([osp.join(root_dir, '/media/thc/Elements/unsp/self_paced2_1124.pth'), 10])

    data_dir = list()

    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/dynamic_6dof'), 550])
    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/boxes_6dof'), 550])
    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/poster_6dof'), 550])
    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/office_zigzag'), 247])
    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/slider_depth'), 84])
    data_dir.append([osp.join(root_dir, '/media/thc/Elements/unsp/dvs_datasets/calibration'), 550])

    df_full = create_dataframe_all(data_dir, models_list)

    vals = range(len(models_list))
    vals = [val for val in vals for _ in (0, 1)]
    norm = True
    data_dict = {}
    for i in vals:
        arch_path, arch_n = models_list[i]
        netG = arch(arch_path, arch_n)
        norm = not norm
        for d, frm in data_dir:
            te = testset(d, ev_rate, norm_e=norm, e_abs=False, num_freams=frm)
            lpips_e, tc_loss, tc_gt_loss, ssim_e, mse_e = [], [], [], [], []
            stats = None
            path = './res/' + d.split('/')[-1] + '-' + arch_path.split('/')[-1].split('.')[0]
            for f in tqdm.tqdm(range(frm)):
                with torch.no_grad():
                    x, y = te.getitem(f)
                    x = x[:, :, :176].cuda()
                    y = y[None, None, :176].cuda()

                    if f == 0:
                        pred = torch.zeros_like(y)  

                    for ii in range(x.shape[0]): 

                        xx = x[ii][None]
                        # xx = xx.unsqueeze(1)
                        # xx = torch.concat((xx,xx,xx),1)
                        # pred = netG(xx)
                        # if arch_n in [0, 1, 3]:
                        #     pred, stats = netG(xx, stats)
                        # elif arch_n == 6 or arch_n == 7:
                        #     _, pred = netG(xx)
                        # else:
                        #     _, pred, stats = netG(xx, stats, pred)
                        pred, stats = netG(xx, pred, stats)

                        # flops, params = profile(netG, inputs=(xx, pred, stats, ))
                    # print("flops"+ str(flops/1000**3) + 'G')
                    img1 = y.repeat(1, 3, 1, 1)
                    if arch_n in [0, 1, 3]:
                        pred1 = pred.repeat(1, 3, 1, 1)
                    else:
                        pred1 = pred

                    if f > 0:
                        tc = lossfn.tem_loss(pred0, pred1, img0, img1)
                        tc_loss.append(tc.item())
                        tc_gt = lossfn.tem_loss(img0, img1, img0, img1)
                        tc_gt_loss.append(tc_gt.item())

                    ssim_loss, mse_loss, lpips = lossfn.test_loss(pred1, y)

                    ssim_e.append(ssim_loss.item())
                    mse_e.append(mse_loss.item())
                    lpips_e.append(lpips.item())

                    pred0 = pred1.detach()
                    img0 = img1.detach()

                    p = rescale(pred1)
                    y = rescale(img1)

                    p = p[0].detach().cpu().numpy().mean(0)
                    y = y[0].detach().cpu().numpy().mean(0)

                    p = np.uint8(cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX))
                    y = np.uint8(cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX))

                    y = clahe.apply(y)
                    p = clahe.apply(p)

                    if not os.path.exists(path):
                        os.makedirs(path)

                    cv2.imwrite(osp.join(path, 'pred_' + str(f) + '.png'), p)

            if norm:
                col = arch_path.split('/')[-1].split('.')[0] + '_norm'
            else:
                col = arch_path.split('/')[-1].split('.')[0]

            idx = d.split('/')[-1]

            '''
            得到每张图像的loss，而不只是平均值
            column = col + '_' + idx
            df_full['mse'].loc[0:frm, column] = mse_e
            df_full['ssim'].loc[0:frm, column] = ssim_e
            df_full['lpips'].loc[0:frm, column] = lpips_e
            df_full['tc'].loc[0:frm, column] = np.pad(tc_gt_loss, (0, frm - len(tc_gt_loss)), 'mean')

            df_full['mse'].loc['mean', column] = np.mean(mse_e[:frm])
            df_full['ssim'].loc['mean', column] = np.mean(ssim_e[:frm])
            df_full['lpips'].loc['mean', column] = np.mean(lpips_e[:frm])
            df_full['tc'].loc['mean', column] = np.mean(tc_gt_loss[:frm])
            '''

            df_full['mse']._set_value(idx, col, np.mean(mse_e[:frm]))
            df_full['ssim']._set_value(idx, col, np.mean(ssim_e[:frm]))
            df_full['lpips']._set_value(idx, col, np.mean(lpips_e[:frm]))
            df_full['tc']._set_value(idx, col, np.mean(tc_loss[:frm]))
        # df_full['tc']._set_value(idx, 'tc_gt', np.mean(tc_gt_loss[:frm]))

        # df_full['tc']._set_value(idx, 'tc_gt', np.mean(tc_gt_loss[:frm]))

        # n = 'norm' if norm else 'no_norm'
        # arch_name = arch_path.split('/')[-1][:-4] + '_' + d.split('/')[-1] + '_' + n
        # data_dict[arch_name] = {'MSE_data': mse_e[:frm],
        #                         'SSIM_data': ssim_e[:frm],
        #                         'lpips_data': lpips_e[:frm],
        #                         'tc_data': tc_loss[:frm],
        #                         'tc_gt_data': tc_gt_loss[:frm]}
            print('-' * 30)
            print(arch_path.split('/')[-1][:-4], f'Norm: {norm}')
            print(d.split('/')[-1])

        re_path = osp.join(output_path.split('/')[0], output_path.split('/')[1])
        save_path = osp.join(root_dir, re_path, output_path.split('/')[-1])
        if osp.exists(save_path) is not True:
            os.makedirs(save_path)

        df_full['mse'].to_csv(osp.join(save_path, 'mse.csv'))
        df_full['ssim'].to_csv(osp.join(save_path, 'ssim.csv'))
        df_full['lpips'].to_csv(osp.join(save_path, 'lpips.csv'))
        df_full['tc'].to_csv(osp.join(save_path, 'tc.csv'))

    print('finish')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/thc/unsp/')
    args = parser.parse_args()

    # main(args, model_path='my_test/saved_models/self_paced.pth', output_path='my_test/saved_output/self_paced/')
    models = glob.glob(osp.join(args.root_dir, '/home/thc/unsp/model/E2VID.pth.tar'))
    for i in range(len(models)):
        output_path = 'saved_output/fixed/' + models[i].split('/')[-1].split('.')[0]
        # if osp.exists(os.path.join(args.root_dir, output_path)):
        #     continue
        # else:
        main(args, models[i], output_path)
