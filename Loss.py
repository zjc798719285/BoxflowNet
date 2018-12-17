import torch
from log.logger import Logger
import numpy as np
logger = Logger('./log/log.mat')

def loss(pre, tar, roi):
    roi = np.array(roi)
    pre_np = pre.detach().cpu().numpy()
    tar_np = tar.detach().cpu().numpy()
    # tt = tar_np[2:6]
    tarabs = sum(sum(map(abs, tar_np[:, 3:5])))
    preabs = sum(sum(map(abs, pre_np[:, 0:2])))

    xt, yt, wt, ht = roi[:, 3] + tar_np[:, 3], roi[:, 4] + tar_np[:, 4], \
                     roi[:, 5] * np.exp(tar_np[:, 5]), roi[:, 6] * np.exp(tar_np[:, 6])
    xmint, xmaxt, ymint, ymaxt = xt - wt/2, xt + wt/2, yt - ht/2, yt + ht/2

    xp, yp, wp, hp = roi[:, 3] + pre_np[:, 0], roi[:, 4] + pre_np[:, 1], \
                     roi[:, 5] * np.exp(pre_np[:, 2]), roi[:, 6] * np.exp(pre_np[:, 3])

    xs, ys, ws, hs = roi[:, 3], roi[:, 4], roi[:, 5], roi[:, 6]
    # xp, yp, wp, hp = roi[2] + pre_np[2], roi[3] + pre_np[3], \
    #                  roi[4] * np.exp(pre_np[4]), roi[5] * np.exp(pre_np[5])

    xminp, xmaxp, yminp, ymaxp = xp - wp / 2, xp + wp / 2, yp - hp / 2, yp + hp / 2
    xmins, xmaxs, ymins, ymaxs = xs - ws / 2, xs + ws / 2, ys - hs / 2, ys + hs / 2


    ioup = mIOU([xmint, xmaxt, ymint, ymaxt], [xminp, xmaxp, yminp, ymaxp])
    ioub = mIOU([xmint, xmaxt, ymint, ymaxt], [xmins, xmaxs, ymins, ymaxs])
    lossx = SmoothL1_loss(pre[:, 0], tar[:, 3])
    lossy = SmoothL1_loss(pre[:, 1], tar[:, 4])
    lossw = SmoothL1_loss(pre[:, 2], tar[:, 5])
    lossh = SmoothL1_loss(pre[:, 3], tar[:, 6])
    loss = lossx + lossy + lossw + lossh

    return float(lossx), float(lossy), float(lossw), float(lossh), ioub, ioup, loss, tarabs, preabs


def SmoothL1_loss(pre, tar):
    # loss = torch.mean((pre - tar)**2)
    diffabs = torch.abs(pre - tar)
    loss = torch.where(diffabs < 1, 0.5 * (pre - tar)**2, diffabs - 0.5 * torch.ones_like(diffabs))
    loss = torch.mean(loss)
    return loss

def mIOU(b1, b2):
    eps = 1e-10
    xmin, xmax, ymin, ymax = np.maximum(b1[0], b2[0]), np.minimum(b1[1], b2[1]), \
                             np.maximum(b1[2], b2[2]), np.minimum(b1[3], b2[3])

    intra = (xmax - xmin) * (ymax - ymin)
    mask = np.where(xmax - xmin <= 0, 0, 1) * np.where(ymax - ymin <= 0, 0, 1)
    union = (b1[1] - b1[0]) * (b1[3] - b1[2]) + (b2[1] - b2[0]) * (b2[3] - b2[2]) - intra
    iou = intra / (union + eps) * mask
    iou_m = np.mean(iou)
    return iou_m
