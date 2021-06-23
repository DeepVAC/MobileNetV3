import numpy as np
import cv2

from deepvac.datasets import FileLineCvStrDataset
from deepvac.utils import addUserConfig

### customized dataset begin
class ClsDataset(FileLineCvStrDataset):
    def __getitem__(self, idx):
        sample, target = super(ClsDataset, self).__getitem__(idx)
        sample = cv2.resize(sample,(self.config.img_size[0], self.config.img_size[1]))
        sample = sample.astype(np.float32)
        sample = sample/127.5 - 1
        sample = sample.transpose([2, 0, 1])
        return sample, int(target)
