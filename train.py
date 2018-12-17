from models.model1 import resnet18
from models.model1 import SubNet3 as SubNet
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from DataLoader_imagenet import dataset
# from DataLoader import dataset
from ROIPooling import ROIPooling
import time
import Loss
from log.logger import Logger
logger = Logger('./log/log.mat')

batch_size = 8
trainlist = './trainlist/train_imagenet_1_500_20.txt'
trainlist_all = './trainlist/train_imagenet_1_500_all_20.txt'


subnet = SubNet().to('cuda')
# subnet.load_state_dict(torch.load('./checkpoint/subnet61.pt'))
basemodel = resnet18().eval().to('cuda')
basemodel.load_state_dict(torch.load('./checkpoint/pretrain/resnet18.pth'))
train_loader = torch.utils.data.DataLoader(dataset(trainlist=trainlist, trainlist_all=trainlist_all,
                                                    split_len=(8, 10), delta=(0.5, 0.5)), batch_size=batch_size,
                                                    num_workers=2, shuffle=True)

optimizer = optim.SGD(subnet.parameters(), lr=2e-4)
def train(epoch):
    for epo in range(epoch):
        mioub, mioup = [], []
        for ind, (_, mhi, rois, tars) in enumerate(train_loader):
            t1 = time.time()
            outs = basemodel(mhi.to('cuda'))
            tt = outs[3]
            try:
                ROIfs, Tars, ROIs = ROIPooling(rois=rois.numpy(), features=tt, target=tars)
                sum_loss, slx, sly, slw, slh, ioubs, ioups, tarabss, preabss = [], [], [], [], [], [], [], [], []
                pre_tars = subnet(ROIfs)
                lx, ly, lw, lh, ioub, ioup, loss, tarabs, preabs = Loss.loss(pre_tars, Tars.to('cuda'), ROIs)
                sum_loss.append(loss)
                slx.append(lx)
                sly.append(ly)
                slw.append(lw)
                slh.append(lh)
                ioubs.append(ioub)
                ioups.append(ioup)
                tarabss.append(tarabs)
                preabss.append(preabs)
                sum(sum_loss).backward()
                torch.nn.utils.clip_grad_norm_(subnet.parameters(), 100)
                mlx, mly, mlw, mlh, ioub, ioup, tarabs, preabs = \
                   sum(slx)/len(slx), sum(sly)/len(sly), \
                    sum(slw)/len(slw), sum(slh)/len(slh), sum(ioubs)/len(ioubs), sum(ioups) / len(ioups),\
                    sum(tarabss)/len(tarabss), sum(preabss)/len(preabss)#  sum(map(float, sum_loss))/len(sum_loss),
                optimizer.step()
            except IOError:
                print(IOError)
                print('train step have bugs, goto next step')

                # continue
            t2 = time.time()
            mioub.append(ioub)
            mioup.append(ioup)
            timep = (t2 - t1)
            print('epoch:', epo, 'step:', ind, 'lx:{:.4f}'.format(mlx), 'ly:{:.4f}'.format(mly),
                                               'lw:{:.4f}'.format(mlw), 'lh:{:.4f}'.format(mlh),
                                               'ioub:{:.4f}'.format(ioub), 'ioup:{:.4f}'.format(ioup),
                                               'mioub:{:.4f}'.format(sum(mioub)/len(mioub)),
                                               'mioup:{:.4f}'.format(sum(mioup)/len(mioup)),
                                               'tarabs:{:.4f}'.format(tarabs), 'preabs:{:.4f}'.format(preabs),
                                               'time:{:.4f}'.format(timep))
            logger.write(['lx', 'ly', 'lw', 'lh', 'diff'], [mlx, mly, mlw, mlh, tarabs - preabs])
            if ind % 10 == 0:
                logger.savetomat()
        torch.save(subnet.state_dict(), './checkpoint/subnet{}.pt'.format(epo))




if __name__ == '__main__':
    train(10000)
