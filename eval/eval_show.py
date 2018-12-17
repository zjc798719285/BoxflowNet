from models.model1 import resnet18
from models.model1 import SubNet3 as SubNet
import torch
from DataLoader_imagenet import dataset
from ROIPooling import ROIPooling
import numpy as np
import cv2


trainlist = '../trainlist/trainlist_imagenet2_20.txt'
trainlist_all = '../trainlist/trainlist_imagenet2_all_20.txt'


resnet = resnet18().eval().to('cuda')
resnet.load_state_dict(torch.load('../checkpoint/pretrain/resnet18.pth'))
subnet = SubNet().eval().to('cuda')
subnet.load_state_dict(torch.load('../checkpoint/subnet2.pt'))

train_loader = torch.utils.data.DataLoader(dataset(trainlist=trainlist, trainlist_all=trainlist_all,
                                                    split_len=(9, 10), delta=(0.8, 0.8)), batch_size=1,
                                                    num_workers=2, shuffle=False)
def main():
    for ind, (img, mhi, rois, tars) in enumerate(train_loader):
        size = 512
        image = img[0, ...].numpy()
        outs = resnet(mhi.to('cuda'))
        tt = outs[3]
        ROIfs, Tars, ROIs = ROIPooling(rois=rois.numpy(), features=tt, target=tars)
        Otars = subnet(ROIfs)
        # for i, (roi, tar) in enumerate(zip(ROIfs, Tars)):
        #     pre_tars = subnet(roi)
        #     Otars.append(pre_tars.detach().cpu().numpy())

        for idx, (roi, otari, tari) in enumerate(zip(ROIs, Otars.detach().cpu().numpy(), Tars.numpy())):

            ox, oy, ow, oh = roi[3] + otari[0], roi[4] + otari[1], \
                             roi[5] * np.exp(otari[2]), roi[6] * np.exp(otari[3])
            x, y, w, h = roi[3] + tari[3], roi[4] + tari[4], \
                             roi[5] * np.exp(tari[5]), roi[6] * np.exp(tari[6])
            xo, yo, wo, ho = roi[3], roi[4], roi[5], roi[6]

            oxmin, oxmax, oymin, oymax = int((ox - ow/2)*size), int((ox + ow/2)*size), \
                                     int((oy - oh/2)*size), int((oy + oh/2)*size)

            xmin, xmax, ymin, ymax = int((x - w / 2) * size), int((x + w / 2) * size), \
                                         int((y - h / 2) * size), int((y + h / 2) * size)
            xmino, xmaxo, ymino, ymaxo = int((xo - wo / 2) * size), int((xo + wo / 2) * size), \
                                     int((yo - ho / 2) * size), int((yo + ho / 2) * size)

            cv2.rectangle(image, (oxmin, oymin), (oxmax, oymax), [0, 255, 0], 2) #  绿色预测位置
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [0, 0, 255], 1)     #  红色实际位置
            cv2.rectangle(image, (xmino, ymino), (xmaxo, ymaxo), [255, 0, 0], 1)  #  蓝色上一帧位置
        print(idx)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            while True:
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break


if __name__ =='__main__':
    main()