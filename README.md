# MobileNetV3
DeepVAC-compliant MobileNetV3 implementation

### 项目依赖

deepvac, pytorch, opencv-python, numpy

### 配置文件

** 准备数据 **

修改config.py文件，指定训练集、验证集、测试集的图片目录和对应的标注txt文件

```
config.train.fileline_data_path_prefix = 'train images path'
config.train.fileline_path = 'data/train_cls.txt'

config.val.fileline_data_path_prefix = 'val images path'
config.val.fileline_path = 'data/val_cls.txt'

config.test.fileline_data_path_prefix = 'test images path'
config.test.fileline_path = 'data/test_cls.txt'
```

** 修改分类数 **

修改config.py

```
config.num_classes = 4
```

### 训练

** 单卡训练 **

```
python3 train.py
```

** ddp训练 **

修改word_size

```
config.world_size = 2

```
ddp 训练

```
python train_ddp.py --rank 0 --gpu 0
python train_ddp.py --rank 1 --gpu 1
...
```
这里启动的训练命令数与word_size对应，0为主进程

### 测试

** 指定模型路径 **

修改config.py指定模型路径

```
config.model_path = 'model path'
```

** 运行测试脚本 **

```
python3 test.py
```

### 项目参考

有关配置文件和ddp的更多细节，请参考：https://github.com/DeepVAC/deepvac
