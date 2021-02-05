import sys
from deepvac import Deepvac, ClassifierReport, FileLineCvStrDataset, LOG 
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from train import ClsDataset
from deepvac.syszux_mobilenet import MobileNetV3

class DeepvacClsTest(Deepvac):
    def __init__(self, deepvac_config):
        super(DeepvacClsTest,self).__init__(deepvac_config)
        self.initTestLoader()
    
    def initNetWithCode(self):
        self.net = MobileNetV3(class_num=self.conf.num_classes)

    def process(self):
        cls_report = ClassifierReport(ds_name="cls_dataset",total_num=self.test_loader.__len__(), cls_num=self.conf.num_classes, threshold=0.99)
        for i, (inp, self.labels) in enumerate(self.test_loader):
            inp = inp.to(self.device)
            self.prediction = self.net(inp).cpu()
            self.prediction = np.argmax(self.prediction.data, axis=1)
            for gt, pred in zip(np.array(self.labels), np.array(self.prediction)):
                cls_report.add(gt,pred)
            if i%100 ==0:
                print(i)
        cls_report()

    def initTestLoader(self):
        self.test_dataset = ClsDataset(self.conf.test)
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.conf.test.batch_size,
            shuffle=self.conf.test.shuffle,
            num_workers=self.conf.workers,
            pin_memory=self.conf.pin_memory
        )

if __name__ == '__main__':
    from config import config as deepvac_config
    cls = DeepvacClsTest(deepvac_config)
    cls(torch.rand([1, 3, 48, 192])
)
