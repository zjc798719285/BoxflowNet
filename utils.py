import os
import cv2
import numpy as np
import xml.etree.cElementTree as ET

def load_ann_per_img(imgpath):
    img = cv2.imread(imgpath)
    height = img.shape[0]
    width = img.shape[1]
    labpath = imgpath.replace('sequences', 'annotations')[0:-12] + '.txt'
    imgid = str(int(imgpath.split('\\', len(imgpath))[-1].replace('.jpg', '')))
    ann = open(labpath, 'r').readlines()
    boxes = [i.split(',', len(i)) for i in ann if i.split(',', len(i))[0] == imgid]
    boxes = np.array([list(map(float, b)) for b in boxes])
    boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5] = \
       boxes[:, 2]/width + boxes[:, 4]/width/2, boxes[:, 3]/height + \
       boxes[:, 5]/height/2, boxes[:, 4]/width, boxes[:, 5]/height
    return boxes


def parse_imagenet(imagepath):
    imgpath = imagepath
    labdir = imgpath.replace('Data', 'Annotations').replace('JPEG', 'xml')
    lab_xml = open(labdir).read()
    root = ET.fromstring(lab_xml)
    objs = root.findall('object')
    labs = []
    for obj in objs:
        xmin = int(obj.find('bndbox').find('xmin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        id = int(obj.find('trackid').text)
        # bbox = [(xmin + xmax)/2/width, (ymin + ymax)/2/height, (xmax - xmin)/width, (ymax - ymin)/height]
        # size = [width, height]
        labs.append(np.array([id, width, height, (xmin + xmax)/2/width, (ymin + ymax)/2/height,
                                                   (xmax - xmin)/width, (ymax - ymin)/height]))
    if labs == []:
        labs.append(np.ones(shape=(7)))
    return labs

# def getlist_imagenet(imgdir, split_len):
#     listseq = [os.path.join(os.path.join(imgdir, i), j) for i in os.listdir(imgdir) for j in os.listdir(os.path.join(imgdir, i))]
#     imgs = [os.path.join(i, j) for i in listseq for j in os.listdir(i)]
#     f1 = open('trainlist_imagenet_all_10.txt', 'w')
#     f2 = open('trainlist_imagenet_10.txt', 'w')
#
#     for img, lab in imgs:
#         f1.write(img)
#         f1.write('\n')
#     indx = 0
#     for idx, seq in enumerate(listseq):
#         print(idx)
#         length = len(os.listdir(os.path.join(imgdir, seq))) - split_len
#         n_length = len(os.listdir(os.path.join(imgdir, seq)))
#         for i in range(length):
#             for j in range(split_len):
#                 f2.write(str(indx + i + j + 1))
#                 f2.write(' ')
#             f2.write('\n')
#         indx += n_length
#     print('trainlist done!')
#     f2.close()

def getlist_imagenet(imgdir, split_len):
    seqs = [os.path.join(imgdir, os.path.join(i, j)) for i in os.listdir(imgdir) for j in os.listdir(os.path.join(imgdir, i))]
    seqs = seqs[1:500]
    outimgs=[]
    for idx, seq in enumerate(seqs):
        print(idx)
        outseqs = []
        imgs = [os.path.join(seq, i) for i in os.listdir(seq)]
        labs = [i.replace('Data', 'Annotations').replace('JPEG', 'xml') for i in imgs]
        for img, lab in zip(imgs, labs):
            lab_xml = open(lab).read()
            root = ET.fromstring(lab_xml)
            objs = root.findall('object')
            if len(objs) > 0:
                outseqs.append(img)
        outimgs.append(outseqs)
    print('process labels is done!')
    f1 = open('train_imagenet_1_500_all_20.txt', 'w')
    f2 = open('train_imagenet_1_500_20.txt', 'w')
    idx = 0
    for outseq in outimgs:
        for outimg in outseq:
            f1.write(outimg)
            f1.write('\n')
            print()
        seqlen = len(outseq)
        seqlen_ = len(outseq) - split_len
        for i in range(seqlen_):
            for j in range(split_len):
                f2.write(str(idx + i + j + 1))
                f2.write(' ')
            f2.write('\n')
        idx += seqlen
    f1.close()
    f2.close()
    print('finish!')



def getlist(imgdir, split_len):
    f1 = open('vallist_all_5.txt', 'w')
    f2 = open('vallist_5.txt', 'w')
    for seq in os.listdir(imgdir):
        for img in os.listdir(os.path.join(imgdir, seq)):
            imgp = os.path.join(os.path.join(imgdir, seq), img)
            f1.write(imgp)
            f1.write('\n')
    f1.close()
    print('trainlist_all  done!')
    indx = 0
    for idx, seq in enumerate(os.listdir(imgdir)):
        print(idx)
        length = len(os.listdir(os.path.join(imgdir, seq))) - split_len
        n_length = len(os.listdir(os.path.join(imgdir, seq)))
        for i in range(length):
            for j in range(split_len):
                f2.write(str(indx+i + j + 1))
                f2.write(' ')
            f2.write('\n')
        indx += n_length
    print('trainlist done!')
    f2.close()



def getMHI_2frame(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        else:
            diff = cv2.absdiff(img, imgs[i-1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            update = np.where(mhi * 0.5 < 0, 0, mhi * 0.5)
            mhi = gray_mask + update
            mhi = np.where(mhi > 255, 255, mhi)
    return mhi.astype(np.float32)

def getMHI_2frame1(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        else:
            diff = cv2.absdiff(img, imgs[i-1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            update = np.where(mhi * 0.5 < 0, 0, mhi * 0.5)
            mhi = gray_mask + update
            # mhi = np.where(mhi > 255, 255, mhi)
    return mhi.astype(np.float32)

def getMHI_2frame2(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        else:
            diff = cv2.absdiff(img, imgs[i-1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            update = np.where(mhi - 50 < 0, 0, mhi - 50)
            mhi = gray_mask + update
            # mhi = np.where(mhi > 255, 255, mhi)
    return mhi.astype(np.float32)


def getMHI_2frame3(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        else:
            diff = cv2.absdiff(img, imgs[i-1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            update = np.where(mhi - 50 < 0, 0, mhi - 50)
            mhi = gray_mask + update
            mhi = np.where(mhi >= 255, 255, mhi)
    return mhi.astype(np.float32)



def getMHI_3frame(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs[0:-1]):
        if i == 0:
            diff = cv2.absdiff(imgs[i + 1], imgs[i])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        else:
            diff1 = cv2.absdiff(imgs[i + 1], imgs[i])
            gray_diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)

            _, gray_mask1 = cv2.threshold(gray_diff1, 20, 255, cv2.THRESH_BINARY)
            diff2 = cv2.absdiff(imgs[i], imgs[i - 1])
            gray_diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
            _, gray_mask2 = cv2.threshold(gray_diff2, 20, 255, cv2.THRESH_BINARY)
            gray_mask = cv2.bitwise_or(gray_mask1, gray_mask2)

        # gray_mask = cv2.dilate(gray_mask, None, iterations=1)
        # gray_mask = cv2.erode(gray_mask, None, iterations=1)
        update = np.where(mhi - 50 < 0, 0, mhi-50)
        mhi = gray_mask + update
        mhi = np.where(mhi > 255, 255, mhi)


    return mhi.astype(np.float32)


def getMHI_org(imgs, delta):
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    num = len(imgs)
    for i, img in enumerate(imgs):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        index = (i + 1) / ((1 + num) * num/2)
        mhi = mhi + gray_img * index



    return mhi.astype(np.float32)


def getMHI_3bk(imgs, delta):
    imgbs = [i.astype(np.float32) for i in imgs]
    bg = (sum(imgbs) / len(imgbs)).astype(np.uint8)
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs[0:-1]):
        if i == 0:
            diff = cv2.absdiff(imgs[i + 1], imgs[i])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        else:
            diff1 = cv2.absdiff(imgs[i + 1], imgs[i])
            gray_diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)

            _, gray_mask1 = cv2.threshold(gray_diff1, 20, 255, cv2.THRESH_BINARY)
            diff2 = cv2.absdiff(imgs[i], imgs[i - 1])
            gray_diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
            _, gray_mask2 = cv2.threshold(gray_diff2, 20, 255, cv2.THRESH_BINARY)
            gray_mask = cv2.bitwise_or(gray_mask1, gray_mask2)

        gray_mask = cv2.dilate(gray_mask, None, iterations=1)
        gray_mask = cv2.erode(gray_mask, None, iterations=1)
        mhi = gray_mask + mhi * delta[0]

    return mhi.astype(np.float32)



def getMHI_bkavg(imgs, delta):
    imgbs = [i.astype(np.float32) for i in imgs]
    bg = (sum(imgbs)/len(imgbs)).astype(np.uint8)
    mhi = np.zeros(shape=(1, imgs[0].shape[0], imgs[1].shape[1]))
    for i, img in enumerate(imgs):
        if i == 0:
            continue
        else:
            diff = cv2.absdiff(img, bg)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, gray_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            gray_mask = cv2.dilate(gray_mask, None, iterations=1)
            gray_mask = cv2.erode(gray_mask, None, iterations=1)

        update = np.where(mhi - 25 < 0, 0, mhi - 25)
        mhi = gray_mask + update


    return mhi.astype(np.float32)



def test_mhi():
    imgdir = 'E:\Person_detection\Dataset\VID\VisDrone2018-VID-val\sequences'
    imglist = []
    for i in os.listdir(imgdir)[0:10]:
        imgpath = os.path.join(imgdir, i)
        imglist.append(cv2.imread(imgpath))
    mhi = getMHI_2frame(imglist, (0.9, 0.8))
    while True:
        cv2.imshow('frame', mhi)
        cv2.waitKey(10)



if __name__ == '__main__':
    imgdir = 'G:\ILSVRC2017_VID\ILSVRC\Data\VID\\train'
    getlist_imagenet(imgdir, 20)

    # imgdir = 'G:\ILSVRC2017_VID\ILSVRC\Data\VID\\train' \
    #          '\ILSVRC2015_VID_train_0001\ILSVRC2015_train_00143001\\000068.JPEG'
    # labdir = 'G:\ILSVRC2017_VID\ILSVRC\Annotations\VID\\tra' \
    #          'in\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00000000\\000000.xml'
    # boxes = parse_imagenet(imgdir)



    # imgdir = 'E:\Person_detection\Dataset\VID\VisDrone2018-VID-val\sequences'
    # outdir = 'E:\Person_detection\BoxFlowNet'
    # l = getlist(imgdir, 5)

    # test_mhi()

    # sequence = 'E:\Person_detection\Dataset\VID\VisDrone2018-VID-train\sequences\\uav0000013_00000_v'
    # for i in os.listdir(sequence):
    #     imgp = os.path.join(sequence, i)
    #     img, lab = load_ann_per_img(imgp)
    #     for idx, box in enumerate(lab):
    #         x, y, w, h, cls = box[2], box[3], box[4], box[5], int(box[7])
    #         xmin = int(x*512)
    #         xmax = int((x + w)*512)
    #         ymin = int(y*512)
    #         ymax = int((y + h)*512)
    #
    #         color = [[250, 50, 50], [200, 0, 50], [150, 100, 50], [100, 100, 50], [50, 150, 0], [0, 200, 100],
    #                 [0, 250, 50], [0, 50, 150], [0, 100, 200], [50, 0, 100], [50, 0, 200], [200, 0, 200]]
    #
    #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color[cls], 1)
    #     cv2.imshow('frame', img)
    #     while True:
    #         if cv2.waitKey(1) & 0xFF == ord(' '):
    #             break
