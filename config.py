from deepvac.syszux_config import *
from torchvision import transforms as trans
config.aug = AttrDict()
config.synthesis = AttrDict()
# DDP
config.dist_url = 'tcp://localhost:27030'
config.world_size = 2
config.disable_git = True
# general
config.num_classes = 2
config.workers = 3
config.epoch_num = 120
config.device = 'cuda'
config.lr = 0.001
config.lr_step = [10,20,40]
config.lr_factor = 0.1
config.save_num = 1
config.log_every = 10
config.num_workers = 3
config.momentum = 0.9
config.nesterov = False
config.weight_decay = 5e-4
config.checkpoint_suffix = None


# output
config.output_dir = 'output'
config.script_model_dir = 'script.pt'
config.trace_model_dir = 'trace.pt'

# train
config.train.fileline_data_path_prefix = 'train images path'
config.train.fileline_path = 'train ground-truth path'
config.train.batch_size = 128
config.train.shuffle = True
config.train.image_size = [192, 48, 3]

#val
config.val.fileline_data_path_prefix = 'val images path'
config.val.fileline_path = 'val ground-truth path'
config.val.batch_size = 1
config.val.num = 800
config.val.num_disp = 10
config.val.shuffle = True
config.val.image_size = [192, 48, 3]

#test
config.model_path = 'model path'
config.test.fileline_data_path_prefix = 'test images path'
config.test.fileline_path = 'test ground-truth path'
config.test.use_fileline = True
config.test.batch_size = 1
config.test.shuffle = False
config.test.image_size = [192, 48, 3]

