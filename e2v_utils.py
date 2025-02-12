import math
from collections import deque

import numpy as np
import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import t

from spynet import run as spynet


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class UncertaintyLoss(nn.Module):
    def __init__(self, a=1, b=0.5, c=0.1):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.ssim_module = pyiqa.create_metric('ssim', device="cuda")
        self.lpips = pyiqa.create_metric('lpips-vgg', device='cuda')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, y, x):
        # pred, y与x的尺寸相同, [n, c, h, w]

        pred_mean = pred.mean(dim=0)
        x_mean = x.mean(dim=[0, 1])
        x_var = x_mean.var(dim=[0, 1])

        error1 = self.mse(pred, y)
        error2 = self.c * self.mse(pred_mean, y)

        error = error1 + error2

        ratio = x_var / x_mean
        a = self.a * torch.log(1 + ratio)

        loss = error + a

        return loss.mean(dim=[0, 2, 3])


class LossFn:
    def __init__(self, as_loss, to_cuda):
        self.vgg = VGG19().to(to_cuda).eval()
        self.criterion = nn.L1Loss(reduction='mean')
        self.criterionMSE = nn.MSELoss(reduction='none')
        # self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
        self.ssim_module = pyiqa.create_metric('ssim', loss_reduction='none', as_loss=as_loss, device=to_cuda)
        self.lpips = pyiqa.create_metric('lpips-vgg', loss_reduction='none', as_loss=as_loss, device=to_cuda)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.flownet = spynet.Network().to(to_cuda).eval()
        self.alpha = 50

    def tem_loss(self, pred0, pred1, img0, img1):
        with torch.no_grad():
            flow = self.flownet(img1.detach(), img0.detach())  # Backward optical flow
            img0_warp = spynet.backwarp(img0.detach(), flow)
            pred0_warp = spynet.backwarp(pred0, flow)
            noc_mask = torch.exp(-self.alpha * torch.sum(img1.detach() - img0_warp, dim=1).pow(2)).unsqueeze(1)

        temp_loss = self.criterion(pred1 * noc_mask, pred0_warp * noc_mask)

        return temp_loss

    def only_self_paced(self, pred, y, k):
        batch_size = pred.shape[0]
        # 计算序列损失
        ssim = self.ssim_module(pred, y)
        mse = self.criterionMSE(pred, y).mean(dim=[1, 2, 3])
        lpips = self.lpips(pred, y)

        # 设置参数
        alpha_1, alpha_2, alpha_3 = 0.2, 0.3, 0.5

        paced_rank = alpha_1*mse + alpha_2*ssim + alpha_3*lpips

        # 计算 self learning
        if k == 0:
            paced_rank_hot = (paced_rank < paced_rank.mean()).float().cuda()
            k = 1 / paced_rank.mean()
        else:
            paced_rank_hot = (paced_rank.cuda() < 1 / (k + 0.5)).float().cuda()
            if paced_rank_hot.sum() == 0:
                paced_rank_hot = torch.randint(2, [1, batch_size]).cuda()


        # 计算各损失，与one hot相乘，只保留筛选后的损失项
        ssim_loss = (1 - ssim) * paced_rank_hot
        mse_loss = mse * paced_rank_hot
        lpips_loss = lpips * paced_rank_hot

        # 计算总损失的数值, k上一步损失函数
        sum_loss = ssim_loss.sum() + mse_loss.sum() + lpips_loss.sum()
        final_loss = abs(sum_loss - paced_rank_hot.sum() / k)
        if math.isnan(final_loss):
            print(sum_loss, ssim_loss, mse_loss, lpips_loss, final_loss, k, paced_rank_hot)
        k = paced_rank_hot.sum() / sum_loss
        # pixel_loss = self.criterion(pred, y)

        # loss_all = pixel_loss + ssim_loss + mse_loss + lpips_loss

        return ssim_loss.sum(), mse_loss.sum(), lpips_loss.sum(), k, final_loss



    def improved_loss(self, pred, y, k):
        batch_size = pred.shape[0]
        # 计算序列损失
        ssim = self.ssim_module(pred, y)
        mse = self.criterionMSE(pred, y).mean(dim=[1, 2, 3])
        lpips = self.lpips(pred, y)

        # 计算归一化后的损失
        ssim_norm = self.normalize_loss(1 - ssim)
        mse_norm = self.normalize_loss(mse)
        lpips_norm = self.normalize_loss(lpips)

        # 计算各损失的方差
        # ssim_var = torch.var(ssim)
        # mse_var = torch.var(mse)
        # lpips_var = torch.var(lpips)

        # # 归一化方差,作为损失权重
        # total_var = F.normalize(torch.tensor((ssim_var, mse_var, lpips_var)), dim=0, p=1).cuda()

        # # 动态权重乘归一化后的结果，用于自步学习的选择
        # paced_rank = torch.zeros(batch_size)
        # norm_loss = torch.stack([ssim_norm, mse_norm, lpips_norm.squeeze()], dim=1)
        # for idx in range(batch_size):
        #     paced_rank[idx] = (total_var * norm_loss[idx]).sum()

        # # 计算
        # if k == 0:
        #     paced_rank_hot = (paced_rank < paced_rank.mean()).float().cuda()
        #     k = 1 / paced_rank.mean()
        # else:
        #     paced_rank_hot = (paced_rank.cuda() < 1 / (k + 0.5)).float().cuda()
        #     if paced_rank_hot.sum() == 0:
        #         paced_rank_hot = torch.randint(2, [1, batch_size]).cuda()

        # 计算各损失，与one hot相乘，只保留筛选后的损失项
        ssim_loss = (1 - ssim) * 1
        mse_loss = mse * 1
        lpips_loss = lpips * 1

        # 计算总损失的数值, k上一步损失函数
        sum_loss = ssim_loss.sum() + mse_loss.sum() + lpips_loss.sum()
        # final_loss = torch.abs(sum_loss - paced_rank_hot.sum() / k)
        # if math.isnan(final_loss):
        #     print(sum_loss, ssim_loss, mse_loss, lpips_loss, final_loss, k, paced_rank_hot)
        # k = paced_rank_hot.sum() / sum_loss
        # # pixel_loss = self.criterion(pred, y)

        # loss_all = pixel_loss + ssim_loss + mse_loss + lpips_loss

        return ssim_loss.sum(), mse_loss.sum(), lpips_loss.sum(), k, sum_loss

    @staticmethod
    def normalize_loss(loss, small_value=1e-6):

        min_loss = torch.min(loss)
        max_loss = torch.max(loss)

        # 避免除零错误
        if max_loss - min_loss < small_value:
            return loss

        normalized_loss = (loss - min_loss) / (max_loss - min_loss)

        return normalized_loss

    def loss(self, pred, y):
        # 这里可以利用不确定性计算各loss的权重
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss
        # -------SSIM loss------
        ssim_loss = 1 - self.ssim_module(pred, y).mean()
        # -------pixel wise loss-------
        pixel_loss = self.criterion(pred, y)
        # -------MSE loss-------
        mse_loss = self.criterionMSE(pred, y).mean()
        # -------features and style loss------
        lpips_loss = self.lpips(pred, y).mean()

        return ssim_loss, mse_loss, lpips_loss, pixel_loss + ssim_loss + mse_loss + lpips_loss

    def test_loss(self, pred, y):
        # 这里可以利用不确定性计算各loss的权重
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss
        # -------SSIM loss------
        ssim_loss = self.ssim_module(pred, y)
        # -------MSE loss-------
        mse_loss = self.criterionMSE(pred, y).mean()
        # -------features and style loss------
        lpips_loss = self.lpips(pred, y)

        return ssim_loss, mse_loss, lpips_loss

    def uncertainty_loss(self, pred, y, x):
        # -------Uncertaincy loss -----------
        '''
        pred 网络预测出来的图像，与每次高斯采样相对应，pred -> [15, 3, 128, 128]
        y 表示真实图像，是pred需要对比的ground truth，y -> [15, 3, 128, 128]
        x 高斯采样的事件帧，用于偶然不确定性的计算 x -> [15, 5, 128, 128]
        '''
        mse = nn.MSELoss(reduction='none')

        epi_mean = x.mean(dim=1)
        epi_pred = pred.mean(dim=1)
        epi_var = epi_pred.T * epi_pred - epi_mean.T * epi_mean

        error = mse(pred.mean(dim=0), y)
        loss = torch.exp(-epi_var + 1e-6) * error + epi_var
        loss /= 2
        n_dims = len(y.shape)
        for d in range(n_dims):
            loss = torch.mean(loss, axis=-1)

        return loss

    def improved_uncertainty_loss(self, pred, y):

        L2 = self.criterionMSE(pred, y)
        sigmar = torch.var(pred, dim=0)

        loss1 = torch.log(sigmar)
        loss2 = L2.mean(dim=0) * torch.exp(-loss1)

        loss = loss1 + loss2
        loss /= 2

        return loss.mean()

    def uncertainty_loss_robust(self, pred, y):
        # -------Uncertaincy loss -----------
        '''
        pred 网络预测出来的图像，与每次高斯采样相对应，pred -> [15, 3, 128, 128]
        y 表示真实图像，是pred需要对比的ground truth，y -> [15, 3, 128, 128]
        x 高斯采样的事件帧，用于偶然不确定性的计算 x -> [15, 5, 128, 128]
        '''
        mse = nn.MSELoss(reduction='none')

        epi_pred_mean = pred.mean(dim=0)
        epi_pred_var = pred.var(dim=0)

        error = mse(epi_pred_mean, y)
        loss = error / epi_pred_var + torch.log(epi_pred_var)
        loss /= 2

        n_dims = len(y.shape)
        for d in range(n_dims):
            loss = torch.mean(loss, axis=-1)

        return loss

    def joint_loss(self, pred, y, x):
        # Calculate multiple losses
        ssim_loss = 1 - self.ssim_module(pred, y).mean()
        mse_loss = self.criterionMSE(pred, y)
        lpips_loss = self.lpips(pred, y).mean()

        # Calculate uncertainty
        epi_var = x.mean(dim=[0, 1])  # 根据x计算事件方差
        entropy = - (pred * torch.log(pred)).sum(dim=1)
        # kl_div =  # 根据pred和y计算KL散度

        # Dynamically adjust weights
        alpha = torch.sigmoid(epi_var)
        beta = torch.sigmoid(entropy.mean())
        gamma = torch.sigmoid(kl_div.mean())

        # Final loss
        loss = alpha * ssim_loss + (1 - alpha) * mse_loss + gamma * lpips_loss
        loss += beta * entropy.mean() + kl_div.mean()

        return loss

    def uncertainty_loss_numpy(self, pred, y, x):
        # -------Uncertaincy loss -----------
        num_samples = pred.shape[0]
        loss_all = np.zeros(num_samples)

        for i in range(num_samples):
            epi_mean = x[i, :].mean(axis=0)
            epi_pred = pred[i, :].mean(axis=0)
            epi_var = epi_pred.T * epi_pred - epi_mean.T * epi_mean
            loss = np.power(y[i, :] - pred[i, :], 2)
            loss = np.exp(-epi_var) * loss + epi_var
            loss /= 2
            loss_all[i] = loss.mean()

        return torch.tensor(np.linalg.norm(loss_all), requires_grad=True)

    def loss2(self, pred, y):
        # the loss function contain
        # pixel wise loss, reg loss, features loss, style loss
        with torch.no_grad():
            y = (y * 2) - 1
        # -------features and style loss------
        features_loss, style_loss = self.featStyleLoss(pred, y.detach())
        # -------SSIM loss------
        y = (y + 1) / 2
        pred = (pred + 1) / 2
        ssim_loss = 1 - self.ssim_module(pred, y)
        # -------pixel wise loss-------
        pixel_loss = self.criterion(pred, y)

        return pixel_loss + ssim_loss + features_loss + style_loss, features_loss

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (h * w)
        return gram

    def featStyleLoss(self, pred, y):
        x_vgg, y_vgg = self.vgg(pred), self.vgg(y.detach())
        f_loss = 0
        s_loss = 0
        for i in range(len(x_vgg)):
            f_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            s_loss += self.weights[i] * self.criterion(self.gram_matrix(x_vgg[i]),
                                                       self.gram_matrix(y_vgg[i].detach()))
        return f_loss, s_loss

    def pixelwise(self, pred, y):
        bs = pred.shape[0]
        pixel_loss = torch.pow(pred.view(bs, -1) - y.view(bs, -1), 2)
        pixel_loss = pixel_loss.mean(1)
        # pixel_loss = torch.sqrt(pixel_loss)
        pixel_loss = pixel_loss.mean()
        return pixel_loss

    def warping(self, x, flo):

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flo

        ## scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = torch.nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.ones(x.size()).cuda()
        mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=False)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


def events_to_voxel_grid_pytorch(events, num_bins, width, height):
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
                              index=xs[valid_indices] + ys[valid_indices] * width + tis_long[
                                  valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=torch.clamp(xs[valid_indices] + ys[valid_indices] * width + (
                                      tis_long[valid_indices] + 1) * width * height, min=0,
                                                max=voxel_grid.shape[0] - 1),
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def hotpix_torch(evs):
    nonzero_ev = (evs != 0)
    num_nonzeros = nonzero_ev.sum()
    mean = evs.sum() / num_nonzeros
    stddev = torch.sqrt((evs ** 2).sum() / num_nonzeros - mean ** 2)
    h = stddev * t.ppf((1 + 0.99) / 2, num_nonzeros.item() - 1)
    clip_v = mean + h
    clip_mat = evs >= clip_v
    clip_mat = clip_mat.float()
    evs -= clip_mat * evs
    return evs


def norm(events):
    with torch.no_grad():
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            events = mask * (events - mean) / (stddev + 1e-8)

    return events


def detach_tensor(h):
    """detach tensors from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self):
        self.auto_hdr = True
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = 10
        self.Imin = 0
        self.Imax = 1

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:

            Imin = torch.min(img).item()
            Imax = torch.max(img).item()
            # ensure that the range is at least 0.1
            Imin = np.clip(Imin, 0.0, 0.45)
            Imax = np.clip(Imax, 0.55, 1.0)
            # adjust image dynamic range (i.e. its contrast)
            if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                self.intensity_bounds.popleft()
            self.intensity_bounds.append((Imin, Imax))
            self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
            self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img.byte()  # convert to 8-bit tensor
        return img


def lr_schedule2(max_v, min_v, len_v, cicle=1, invert=False):
    flen = len_v
    len_v /= cicle

    lr_sch0 = np.cos(np.arange(np.pi, 2 * np.pi, np.pi / (len_v * 0.1)))
    lr_sch1 = np.cos(np.arange(0, np.pi, np.pi / (len_v * 0.9)))

    lr_sch0 = (lr_sch0 + 1) / 2
    lr_sch0 = (max_v - min_v) * lr_sch0
    lr_sch0 += min_v

    lr_sch1 = (lr_sch1 + 1) / 2
    lr_sch1 = lr_sch1 * max_v

    lr_sch = np.concatenate([lr_sch0, lr_sch1])

    if invert:
        lr_sch = np.cos(lr_sch)

    lr_sch = np.tile(lr_sch, [cicle])
    return lr_sch[:flen]


def lr_schedule(max_v, min_v, len_v, cicle=1, invert=False):
    flen = len_v
    len_v /= cicle
    if invert:
        lr_sch = np.cos(np.arange(np.pi, 2 * np.pi, np.pi / len_v))
    else:
        lr_sch = np.cos(np.arange(0, np.pi, np.pi / len_v))
    lr_sch = (lr_sch + 1) / 2
    lr_sch = (max_v - min_v) * lr_sch
    lr_sch += min_v
    lr_sch = np.tile(lr_sch, [cicle])
    return lr_sch[:flen]


def lr_finder(tr, model, criterion):
    lr_init = 1e-8
    lr_fin = 1
    max_step = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)
    loss_sig = []
    lr_sch = lr_schedule(max_v=lr_fin, min_v=lr_init, len_v=max_step, invert=True)
    mo_sch = lr_schedule(max_v=0.9, min_v=0.1, len_v=max_step, invert=False)
    plt.ion()
    _, ax = plt.subplots(1)
    for i, data in enumerate(tr):
        x = data[0].cuda()
        y = data[1].cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion.loss(y_hat, y)
        loss.backward()
        loss_sig.append(loss.item())
        optimizer.step()

        ax.cla()
        ax.plot(loss_sig)
        plt.pause(0.001)
        plt.show()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_sch[i]
            param_group['momentum'] = mo_sch[i]
        if i == max_step - 1:
            break
