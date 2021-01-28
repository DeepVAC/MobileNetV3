# MobileNetV3
DeepVAC-compliant MobileNetV3 implementation

### Requires

deepvac, pytorch, cv2, numpy

### Prepare your dataset

config.train.fileline_data_path_prefix = 'train images path'
config.train.fileline_path = 'data/train_cls.txt'

config.val.fileline_data_path_prefix = 'val images path'
config.val.fileline_path = 'val ground-truth path'

config.test.fileline_data_path_prefix = 'test images path'
config.test.fileline_path = 'test ground-truth path'

### cls num

config.num_classes = 4

### train with single gpu

python3 train.py

### train with ddp

python train.py --rank 0 --gpu 0

python train.py --rank 1 --gpu 1

### other details 

https://github.com/DeepVAC/deepvac
