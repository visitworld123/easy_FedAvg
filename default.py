from yacs.config import CfgNode as CN


_C=CN()
#------------------------federated general setting------------------------#
_C.global_round = 1000
_C.client_num = 10
_C.client_num_per_round = 5
_C.task = 'classification'
_C.fedprox = False



#------------------------trainging parameter-------------------------------#
_C.bs = 64
_C.seed = 0
_C.augmentation = 'default'
_C.dataloader_workers = 0
_C.data_sampler = "Random"  # 'Random', 'imbalance' or 'decay_imb'
_C.image_resize = 32
_C.TwoCropTransform = False

#-----------------------data partition setting-----------------------------#
_C.dataset = 'cifar10'  # 'cifar10', 'cifar100', 'fmnist' or 'SVHN'
_C.data_dir = '/home/zqy2139219/zqy/data'
_C.partition_alpha = 0.1
_C.partition_method = 'hetero'


#-------------------------------model setting---------------------------------#
_C.model = 'resnet18'
_C.num_classes = 10 
_C.output_dim = 10
_C.input_channels = 3
_C.model_out_feature = True
_C.model_out_feature_layer = 'last'
_C.opti = 'SGD' # 'SGD', 'Adam'
_C.lr = 0.01
_C.momentum = 0.9
_C.wd = 0.0001
_C.nesterov = False

#------------------------------label noise setting-----------------------------# 
_C.level_n_system = 0.6
_C.level_n_lowerb = 0.5


#------------------------semi-supervised learning setting---------------------------# 
_C.FSSL_label_ratio = 0.05


#------------------------------GPU setting---------------------------------# 
_C.gpu_index = 0

_C.log_level = 'INFO'  # 'INFO' or 'DEBUG'
_C.record_tool = 'wandb' # 'wandb' or 'tensorboard'
_C.wandb_record = True
_C.wandb_project = 'testtest'
_C.tensorboard_dir = '/output/tensorboard/logs'


def get_cfg_defaults():
    return _C.clone() #局部变量使用形式


cfg = _C