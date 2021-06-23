import torch
import torch.optim as optim
import torchvision.transforms as trans

from deepvac import AttrDict, new
from deepvac.backbones.mobilenet import MobileNetV3
from deepvac.datasets import FileLineDataset

config = new('MobileNetv3Train')
## ------------------ common ------------------
config.core.MobileNetv3Train.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.MobileNetv3Train.output_dir = 'output'
config.core.MobileNetv3Train.log_every = 100
config.core.MobileNetv3Train.disable_git = True
config.core.MobileNetv3Train.model_reinterpret_cast = True
config.core.MobileNetv3Train.cast_state_dict_strict = True
#config.core.MobileNetv3Train.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.MobileNetv3Train.epoch_num = 200
config.core.MobileNetv3Train.save_num = 1

## -------------------- tensorboard ------------------
# config.core.MobileNetv3Train.tensorboard_port = "6007"
# config.core.MobileNetv3Train.tensorboard_ip = None

## -------------------- script and quantize ------------------
#config.cast.script_model_dir = "./output/script.pt"

## -------------------- net and criterion ------------------
config.core.MobileNetv3Train.cls_num = 4
config.core.MobileNetv3Train.net = MobileNetV3(class_num=config.core.MobileNetv3Train.cls_num)
config.core.MobileNetv3Train.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
config.core.MobileNetv3Train.optimizer = torch.optim.Adam(config.core.MobileNetv3Train.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.MobileNetv3Train.epoch_num) ** 0.9, 8)
config.core.MobileNetv3Train.scheduler = optim.lr_scheduler.LambdaLR(config.core.MobileNetv3Train.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
fileline_path = 'data/train_cls.txt'
delimiter = ' ' 
sample_path_prefix = '<your sample path prefix>'
config.datasets.FileLineDataset = AttrDict()
config.datasets.FileLineDataset.composer = trans.Compose([
        trans.Resize([192, 48]),
        trans.ToTensor(),
        trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

config.core.MobileNetv3Train.train_dataset = FileLineDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.MobileNetv3Train.batch_size = 2
config.core.MobileNetv3Train.num_workers = 4
config.core.MobileNetv3Train.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.MobileNetv3Train.train_dataset,
    batch_size = config.core.MobileNetv3Train.batch_size,
    shuffle = True,
    num_workers = config.core.MobileNetv3Train.num_workers,
    pin_memory = True,
)

## ------------------------- DDP ------------------
config.core.MobileNetv3Train.dist_url = 'tcp://localhost:27030'
config.core.MobileNetv3Train.world_size = 2

## -------------------- val ------------------
fileline_path = 'data/val_cls.txt'
config.core.MobileNetv3Train.val_dataset = FileLineDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.MobileNetv3Train.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.MobileNetv3Train.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
config.core.MobileNetv3Test = config.core.MobileNetv3Train.clone()
config.core.MobileNetv3Test.model_path = '<your test model dir / pretrained weights>'
fileline_path = 'data/test_cls.txt'
config.core.MobileNetv3Test.test_dataset = FileLineDataset(config, fileline_path, delimiter, sample_path_prefix)
config.core.MobileNetv3Test.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.MobileNetv3Test.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)