import sys
sys.path.extend(["/opt/public/airlock/lihang/pyclipper/", "/opt/public/airlock/lihang/shapely/", "/opt/public/airlock/lihang/deepvac/"])
from deepvac import LOG, DeepvacTrain
import numpy as np

class DeepvacCls(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacCls,self).__init__(deepvac_config)
    
    def postIter(self):
        if self.config.is_train:
            return

        self.prediction = np.argmax(self.config.output.cpu().data,axis=1)
        for pred, target in zip(self.prediction, self.config.target.cpu()):
            print(pred,target)
            if pred == target:
                self.n_correct += 1

    def preEpoch(self):
        if self.config.is_train:
            return
        self.n_correct = 0

    def postEpoch(self):
        if self.config.is_train:
            return
        self.accuracy = self.n_correct / self.config.val_dataset.__len__()
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

if __name__ == '__main__':
    from config import config as deepvac_config
    cls = DeepvacCls(deepvac_config)
    cls()
