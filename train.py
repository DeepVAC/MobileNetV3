import sys
import random
from PIL import Image
from deepvac import Executor, FileLineCvStrDataset, LOG, DeepvacTrain, is_ddp
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from deepvac.syszux_mobilenet import MobileNetV3

### customized dataset begin
class ClsDataset(FileLineCvStrDataset):
    def __init__(self, config):
        super(ClsDataset, self).__init__(config)
        self.img_size = config.image_size

    def __getitem__(self, idx):
        sample, target = super(ClsDataset, self).__getitem__(idx)


        sample = cv2.resize(sample,(self.img_size[0], self.img_size[1]))

        sample = sample.astype(np.float32)
        sample = sample/127.5 - 1
        sample = sample.transpose([2, 0, 1])
        return sample, int(target)

### customized dataset end

class DeepvacCls(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacCls,self).__init__(deepvac_config)

    def initNetWithCode(self):
        # to initial self.net
        self.net = MobileNetV3(class_num=self.conf.num_classes)
        self.net.to(self.conf.device)

    def initModelPath(self):
        pass

    def initOptimizer(self):
        self.initAdamOptimizer()

    def initCriterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def initTrainLoader(self):
        self.train_dataset = ClsDataset(self.conf.train)
        if is_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.train.batch_size,
            shuffle=False if is_ddp else self.conf.train.shuffle,
            num_workers=self.conf.workers,
            pin_memory=self.conf.pin_memory,
            sampler=self.train_sampler if is_ddp else None
        )

    def initValLoader(self):
        self.val_dataset = ClsDataset(self.conf.val)
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.conf.val.batch_size,
            shuffle=self.conf.val.shuffle,
            num_workers=self.conf.workers,
            pin_memory=self.conf.pin_memory,
        )
        LOG.logI("val loader len: {}".format(len(self.val_loader)))

    def preIter(self):
        self.labels = self.target
        self.target = torch.LongTensor(self.target)

    def doForward(self):
        # inference
        self.prediction = self.net(self.sample)

    def doLoss(self):
        #cls
        self.loss = self.criterion(self.prediction, self.target)
    
    def postIter(self):
        if self.is_train:
            return

        self.prediction = np.argmax(self.prediction.cpu().data,axis=1)
        for pred, target in zip(self.prediction, self.labels):
            print(pred,target)
            if pred == target:
                self.n_correct += 1

    def preEpoch(self):
        if self.is_train:
            return
        self.n_correct = 0

    def postEpoch(self):
        if self.is_train:
            return
        self.accuracy = self.n_correct / float(self.conf.val.num * self.conf.val.batch_size)
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))


if __name__ == '__main__':
    from config import config as deepvac_config
    cls = DeepvacCls(deepvac_config)
    cls()
