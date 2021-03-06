from torch.utils.data import Dataset
from utils import parse_imagenet
from utils import getMHI_2frame3 as getMHI
import cv2
import torch
import numpy as np


class dataset(Dataset):

    def __init__(self, trainlist, trainlist_all, split_len, delta):
        self.trainlist = open(trainlist, 'r').readlines()
        self.trainlist_all = open(trainlist_all, 'r').readlines()
        self.split_len = split_len
        self.delta = delta


    def __len__(self):
        return len(self.trainlist)

    def __getitem__(self, index):
        idxlist = [int(i) for i in self.trainlist[index].split(' ', len(self.trainlist[index])) if not i == '\n']
        # idxlistend = np.random.randint(low=self.split_len[0], high=self.split_len[1], size=1)
        # idxlist = idxlist[0:idxlistend[0]]
        idxsets = [[0, 2, 3, 4, 5], [0, 5, 9, 13, 17]]
        randset = np.random.randint(low=0, high=2, size=1)
        idxlist = [idxlist[i] for i in idxsets[randset[0]]]
        label_star = parse_imagenet(self.trainlist_all[idxlist[0]][0:-1])
        # print('label_star', label_star)
        label_end = parse_imagenet(self.trainlist_all[idxlist[-1]][0:-1])
        # print(label_end)
        label_diff = []; label_star_ = []
        # print('star', idxlist[0], 'end', idxlist[-1])
        for lab_s in label_star:
            lab_e = [i for i in label_end if int(i[0]) == int(lab_s[0])] #
            if len(lab_e) == 0:
                    continue
            dx, dy, dw, dh = lab_e[0][3] - lab_s[3], lab_e[0][4] - lab_s[4], \
                             np.log(lab_e[0][5] / lab_s[5]), np.log(lab_e[0][6] / lab_s[6])

            diff = np.array([lab_e[0][0], lab_e[0][1], lab_e[0][2], dx, dy, dw, dh])
            label_diff.append(diff)
            label_star_.append(lab_s)
        label_star_np = np.array(label_star_)
        label_diff = np.array(label_diff)
        DIM_res = 10 - label_star_np.shape[0]
        res = np.ones(shape=(DIM_res, 7)) * 1e6
        # print(label_star_.shape)
        if len(label_star_) > 0:
            # print('lab', label_star_.shape, 'res', res.shape)
            label_star_np = np.concatenate((label_star_np, res), 0)
            label_diff = np.concatenate((label_diff, res), 0)
        else:
            label_star_np = res
            label_diff = res
        imgs = []
        for idx in idxlist:
            imgpath = self.trainlist_all[idx]
            # print(imgpath)
            img = cv2.resize(cv2.imread(imgpath[0:-1]), (512, 512))
            imgs.append(img)

        # for offset in range(self.split_len):
        #     imgpath = self.trainlist_all[star + offset]
        #     # print(imgpath)
        #     img = cv2.resize(cv2.imread(imgpath[0:-1]), (512, 512))
        #     imgs.append(img)
        mhi1 = getMHI(imgs, delta=self.delta)
        mhi1 = np.repeat(mhi1, repeats=3, axis=0)/255
        # mhi2 = getMHI2(imgs, delta=self.delta)
        # mhi2 = np.repeat(mhi2, repeats=3, axis=0) / 255
        return imgs[-1], mhi1, label_star_np, label_diff.astype(np.float32)









if __name__ == '__main__':
    trainlist = './trainlist/train_imagenet_1_500_20.txt'
    trainlist_all = './trainlist/train_imagenet_1_500_all_20.txt'
    size = 512
    train_loader = torch.utils.data.DataLoader(dataset(trainlist=trainlist, trainlist_all=trainlist_all,
                                                       split_len=(5, 10), delta=(0.5, 0.5)
                                                       ), batch_size=1, num_workers=2)
    for ind, (img, mhi, target, end) in enumerate(train_loader):
        img_np = mhi.numpy()
        # img_np2 = mhi[1].numpy()
        target_np = target.numpy()
        end_np = end.numpy()
        for i in range(img_np.shape[0]):
            mhi = np.transpose(img_np[i, ...], [1, 2, 0])
            # mhi2 = np.transpose(img_np2[i, ...], [1, 2, 0])
            img_show = (mhi * 255).astype(np.uint8).copy()
            # img_show2 = (mhi2 * 255).astype(np.uint8).copy()
            for ti, td in zip(target_np[i, ...], end_np[i, ...]):

                xmin, ymin, xmax, ymax = int((ti[3]-ti[5]/2)*size), int((ti[4]-ti[6]/2)*size), \
                                         int((ti[3] + ti[5]/2)*size), int((ti[4] + ti[6]/2)*size)
                if xmin + xmax + ymin + ymax > 1e6:
                    break
                cv2.rectangle(img_show, (xmin, ymin), (xmax, ymax), [255, 0, 0], 1)
                x, y, w, h = ti[3] + td[3], ti[4] + td[4], ti[5] * np.exp(td[5]), ti[6]*np.exp(td[6])
                # x, y, w, h = td[2],  td[3], td[4], td[5]

                xmin, xmax, ymin, ymax = int((x - w/2)*size), int((x+w/2)*size), int((y-h/2)*size), int((y+h/2)*size)

                cv2.rectangle(img_show, (xmin, ymin), (xmax, ymax), [0, 0, 255], 1)

            cv2.imshow('frame', img_show)
            # cv2.imshow('frame2', img_show2)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                while True:
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                       break














