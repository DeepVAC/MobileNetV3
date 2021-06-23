import torch
import torch.optim as optim

from deepvac import AttrDict, new
from deepvac.backbones.mobilenet import MobileNetV3

from data.dataloader import ClsDataset

config = new('DeepvacCls')
## ------------------ common ------------------
config.core.DeepvacCls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.DeepvacCls.output_dir = 'output'
config.core.DeepvacCls.log_every = 100
config.core.DeepvacCls.disable_git = True
config.core.DeepvacCls.model_reinterpret_cast = True
config.core.DeepvacCls.cast_state_dict_strict = True
#config.core.DeepvacCls.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.DeepvacCls.epoch_num = 200
config.core.DeepvacCls.save_num = 1

## -------------------- tensorboard ------------------
# config.core.DeepvacCls.tensorboard_port = "6007"
# config.core.DeepvacCls.tensorboard_ip = None

## -------------------- script and quantize ------------------
#config.cast.script_model_dir = "./output/script.pt"

## -------------------- net and criterion ------------------
config.core.DeepvacCls.cls_num = 4
config.core.DeepvacCls.net = MobileNetV3(class_num=config.core.DeepvacCls.cls_num)
config.core.DeepvacCls.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
config.core.DeepvacCls.optimizer = torch.optim.Adam(config.core.DeepvacCls.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.DeepvacCls.epoch_num) ** 0.9, 8)
config.core.DeepvacCls.scheduler = optim.lr_scheduler.LambdaLR(config.core.DeepvacCls.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
fileline_path = 'data/train_cls.txt'
delimiter = ' ' 
sample_path_prefix = '<your sample path prefix>'
config.datasets.ClsDataset = AttrDict()
config.datasets.ClsDataset.img_size = [192, 48, 3]

config.core.DeepvacCls.train_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.DeepvacCls.batch_size = 2
config.core.DeepvacCls.num_workers = 4
config.core.DeepvacCls.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.DeepvacCls.train_dataset,
    batch_size = config.core.DeepvacCls.batch_size,
    shuffle = True,
    num_workers = config.core.DeepvacCls.num_workers,
    pin_memory = True,
)

## ------------------------- DDP ------------------
config.core.DeepvacCls.dist_url = 'tcp://localhost:27030'
config.core.DeepvacCls.world_size = 2

## -------------------- val ------------------
fileline_path = 'data/val_cls.txt'
config.core.DeepvacCls.val_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.DeepvacCls.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.DeepvacCls.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
config.core.DeepvacClsTest = config.core.DeepvacCls.clone()

config.core.DeepvacClsTest.model_path = '<your test model dir / pretrained weights>'
fileline_path = 'data/test_cls.txt'
config.core.DeepvacClsTest.test_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.DeepvacClsTest.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.DeepvacClsTest.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)