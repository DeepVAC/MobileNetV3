# MobileNetV3
DeepVAC-compliant MobileNetV3 implementation

# 简介
本项目实现了符合DeepVAC规范的MobileNet V3

**项目依赖**

- deepvac >= 0.5.6
- pytorch >= 1.8.0
- opencv-python
- numpy

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象。

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)

## 3. 准备数据集
自行准备。

## 4. 修改配置文件

修改config.py文件。主要修改内容：
- 指定训练集、验证集、测试集的图片目录前缀、对应的标注txt文件和分隔符

```python
# line 42-44
fileline_path = 'data/train_cls.txt'
delimiter = ' ' 
sample_path_prefix = <your sample_path_prefix>

# line 58
fileline_path = 'data/val_cls.txt'

# line 71
fileline_path = 'data/test_cls.txt'
```

- 修改分类数
```
config.core.cls_num = 4
```

## 5. 训练

执行命令：
```bash
python3 train.py
```

## 6. 测试

指定要测试模型和测试数据的路径，在config.py指定待测模型和数据路径：

```python
fileline_path = 'data/test_cls.txt'
config.core.model_path = "your test model dir / pretrained weights"
```

然后运行测试脚本：

```bash
python3 test.py
```

## 7， 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- checkpoint加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)。
