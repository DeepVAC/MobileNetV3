import sys
sys.path.insert(0,'/gemfield/hostpv/gemfield/github/deepvac')
from deepvac import config
# DDP
config.dist_url = 'tcp://localhost:27030'
config.world_size = 2
config.disable_git = True
# general
config.num_classes = 4
config.workers = 3
config.epoch_num = 120
config.device = 'cpu'
config.lr = 0.001
config.lr_step = [10,20,40]
config.lr_factor = 0.1
config.save_num = 1
config.log_every = 10
config.num_workers = 3
config.momentum = 0.9
config.nesterov = False
config.weight_decay = 5e-4
#config.model_path = "/gemfield/hostpv/gemfield/github/MobileNetV3/output/disable_git/model__2021-02-02-16-07__acc_0.00125__epoch_3__step_0__lr_0.001.pth"
config.jit_model_path = "/gemfield/hostpv/gemfield/github/MobileNetV3/output/disable_git/script__2021-02-02-18-08__acc_0.00125__epoch_0__step_0__lr_0.001.pt"
#config.model_path = "/gemfield/hostpv/gemfield/github/MobileNetV3/output/disable_git/qat__2021-02-02-16-35__acc_0.00125__epoch_7__step_0__lr_0.001.pt"
#config.checkpoint_suffix = "2021-02-02-15-52__acc_0.00125__epoch_2__step_0__lr_0.001.pth"

config.qat_dir = 'test.qat'
# output
config.output_dir = 'output'
config.script_model_dir = 'script.pt'
#config.trace_model_dir = 'trace.pt'

# train
config.train.fileline_data_path_prefix ='/gemfield/hostpv/gemfield/github/MobileNetV3/data/images'
config.train.fileline_path = 'data/train_cls.txt'
config.train.batch_size = 128
config.train.shuffle = True
config.train.image_size = [192, 48, 3]

#val
config.val.fileline_data_path_prefix = '/gemfield/hostpv/gemfield/github/MobileNetV3/data/images'
config.val.fileline_path = 'data/val_cls.txt'
config.val.batch_size = 1
config.val.num = 800
config.val.shuffle = True
config.val.image_size = [192, 48, 3]

#test
config.test.fileline_data_path_prefix = '/gemfield/hostpv/gemfield/github/MobileNetV3/data/images'
config.test.fileline_path = 'data/val_cls.txt'
config.test.batch_size = 1
config.test.image_size = [192, 48, 3]
