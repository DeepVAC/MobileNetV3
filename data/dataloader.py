from deepvac.datasets import FileLineCvStrDataset
from deepvac.utils import addUserConfig
import numpy as np
import cv2

### customized dataset begin
class ClsDataset(FileLineCvStrDataset):
    def __init__(self, deepvac_config, fileline_path, delimiter, sample_path_prefix):
        super(ClsDataset, self).__init__(deepvac_config, fileline_path, delimiter, sample_path_prefix)
        self.img_size = addUserConfig("img_size", self.config.img_size, [192, 48, 3])

    def __getitem__(self, idx):
        sample, target = super(ClsDataset, self).__getitem__(idx)
        sample = cv2.resize(sample,(self.img_size[0], self.img_size[1]))
        sample = sample.astype(np.float32)
        sample = sample/127.5 - 1
        sample = sample.transpose([2, 0, 1])
        return sample, int(target)
