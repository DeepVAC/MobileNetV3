import sys
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_report import OcrReport, ClassifierReport
from deepvac.syszux_loader import FileLineCvStrDataset
from deepvac.syszux_log import LOG
import torch
import torch.utils.data as data
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

        img_h, img_w = sample.shape[:2]
        in_w = int(img_w/img_h * self.img_size[1])
        sample = cv2.resize(sample,(in_w, self.img_size[1]))

        if in_w >= self.img_size[0]:
            sample = sample[0:self.img_size[1], 0:self.img_size[0]]
        else:
            sample = cv2.copyMakeBorder(sample, 0, 0, 0, self.img_size[0] - in_w, cv2.BORDER_CONSTANT, value = [0, 0, 0])

        sample = sample.astype(np.float32)
        sample = sample/127.5 - 1
        sample = sample.transpose([2, 0, 1])
        return sample, int(target)

class DeepvacClsTest(Deepvac):
    def __init__(self, deepvac_config):
        super(DeepvacClsTest,self).__init__(deepvac_config)
        self.initTestLoader()
    
    def initNetWithCode(self):
        #to initial self.net
        self.net = MobileNetV3(class_num=self.conf.num_classes)
        self.net.to(self.device)

    def inference(self):
        with torch.no_grad(): 
            for i, (inp, self.labels) in enumerate(self.test_loader):
                inp = inp.to(self.device)
                self.prediction = self.net(inp).cpu()
                self.prediction = np.argmax(self.prediction.data, axis=1)
                LOG.logI('pred:{}'.format(self.prediction))

    def report(self):
        cls_report = ClassifierReport(ds_name="cls_dataset",total_num=self.test_loader.__len__(), cls_num=self.conf.num_classes, threshold=0.99)
        with torch.no_grad():
            for i, (inp, self.labels) in enumerate(self.test_loader):
                #self.exportTorchViaTrace(inp)
                inp = inp.to(self.device)
                self.prediction = self.net(inp).cpu()
                self.prediction = np.argmax(self.prediction.data, axis=1)
                for gt, pred in zip(np.array(self.labels), np.array(self.prediction)):
                    cls_report.add(gt,pred)
                if i%100 ==0:
                    print(i)
        cls_report()

    def process(self):
        if self.conf.test.use_fileline:
            self.report()
        else:
            self.inference()

    def initTestLoader(self):
        if self.conf.test.use_fileline:
            self.test_dataset = ClsDataset(self.conf.test)
            self.test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.conf.test.batch_size,
                shuffle=self.conf.test.shuffle,
                num_workers=self.conf.workers,
                pin_memory=self.conf.pin_memory
            )
        else:
            img = cv2.imread(self.conf.test.image_path)
            img = cv2.resize(img,(self.conf.test.image_size[0],self.conf.test.image_size[1]))

            img = img.astype(np.float32)
            img = img/127.5 - 1

            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
            self.test_loader = [(img.unsqueeze(0), None)]

if __name__ == '__main__':
    from config import config as deepvac_config
    cls = DeepvacClsTest(deepvac_config)
    cls()
