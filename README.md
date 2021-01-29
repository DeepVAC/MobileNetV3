# MobileNetV3
DeepVAC-compliant MobileNetV3 implementation

# 简介
本项目实现了符合DeepVAC规范的MobileNet V3。

**项目依赖**

- deepvac
- pytorch
- opencv-python
- numpy


# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象。

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

## 3. 准备数据集
自行准备。

## 4. 修改配置文件

修改config.py文件。主要修改内容：
- 指定训练集、验证集、测试集的图片目录和对应的标注txt文件

```python
config.train.fileline_data_path_prefix = 'train images path'
config.train.fileline_path = 'data/train_cls.txt'

config.val.fileline_data_path_prefix = 'val images path'
config.val.fileline_path = 'data/val_cls.txt'

config.test.fileline_data_path_prefix = 'test images path'
config.test.fileline_path = 'data/test_cls.txt'
```

- 修改分类数
```
config.num_classes = 4
```

## 5. 训练

### 5.1 单卡训练
执行命令：
```bash
python3 train.py
```

### 5.2 分布式训练

在config.py中修改如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```


## 6. 测试

指定要测试模型的路径，在config.py指定待测模型路径：

```python
config.model_path = 'your_model_path'
```

然后运行测试脚本：

```bash
python3 test.py
```

## 7， 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)。
