import torch
import torch.optim as optim

from deepvac import config, AttrDict
from deepvac.backbones import MobileNetV3

from data.dataloader import ClsDataset

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 100
config.core.disable_git = True
config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = True
#config.core.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.epoch_num = 200
config.core.save_num = 1

## -------------------- tensorboard ------------------
#config.core.tensorboard_port = "6007"
#config.core.tensorboard_ip = None

## -------------------- script and quantize ------------------
#config.cast.script_model_dir = "./output/script.pt"

## -------------------- net and criterion ------------------
config.core.cls_num = 4
config.core.net = MobileNetV3(class_num=config.core.cls_num)
config.core.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.epoch_num) ** 0.9, 8)
config.core.scheduler = optim.lr_scheduler.LambdaLR(config.core.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
fileline_path = 'data/train_cls.txt'
delimiter = ' ' 
sample_path_prefix = '/gemfield/hostpv2/lihang/cls_minidata/'
config.datasets.ClsDataset = AttrDict()
config.datasets.ClsDataset.img_size = [192, 48, 3]
config.core.train_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.train_dataset,
    batch_size = 12,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,
    sampler = None
)

## -------------------- val ------------------
fileline_path = 'data/val_cls.txt'
config.core.val_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
#config.core.model_path = "your test model dir / pretrained weights"
config.core.model_path = "output/main/model__2021-05-19-21-13__acc_0__epoch_6__step_0__lr_0.000484222075.pth"
fileline_path = 'data/test_cls.txt'
config.core.test_dataset = ClsDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## ------------------------- DDP ------------------
config.dist_url = 'tcp://172.16.90.55:27030'
config.world_size = 1
