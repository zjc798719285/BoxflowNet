import torch
import math
import numpy as np
def ROIPooling(rois, features, target):
    # tnp = target.numpy()
    size = features.shape[2]
    ROIfs, ROIs, Tars = [], [], []
    for bi, roi in enumerate(rois):
        fi = features[bi, ...]
        for i in range(roi.shape[0]):
            xmin, xmax, ymin, ymax = min(max(int((roi[i, 3] - roi[i, 5]/2) * size - 1), 0), size - 1), \
                                     int(min(max(np.ceil((roi[i, 3] + roi[i, 5]/2) * size + 1), 0), size - 1)),\
                                     min(max(int((roi[i, 4] - roi[i, 6]/2) * size - 1), 0), size - 1), \
                                     int(min(max(np.ceil((roi[i, 4] + roi[i, 6]/2) * size + 1), 0), size - 1))
            if abs(xmin) + abs(xmax) + abs(ymin) + abs(ymax) >= (size-1) * 4:
                break
            roif = fi[:, ymin:ymax, xmin:xmax].detach().resize_(1, 256, 10, 10)

            # roif = fi[:, xmin:xmax, ymin:ymax].detach().resize_(1, 256, 8, 8)
            ROIfs.append(roif)
            Tars.append(torch.unsqueeze(target[bi, i, ...], dim=0))
            ROIs.append(roi[i, :])

    ROIfs = torch.cat((ROIfs), dim=0)
    Tars = torch.cat((Tars), dim=0)

    return ROIfs, Tars, ROIs

