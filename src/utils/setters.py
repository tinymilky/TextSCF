import os
import torch
import random

import numpy as np

def setGPU(opt):

    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    if not torch.cuda.is_available():
        raise Exception("No GPU found")
    print("----->>>> GPU %s is set up ..." % opt['gpu_id'])

def setFoldersLoggers(opt, split_id = None):

    opt['data_path'] = os.path.join(opt['datasets_path'], opt['dataset'])
    opt['log'] = os.path.join(opt['logs_path'], opt['dataset'], opt['model'])

    os.makedirs(opt['log'], exist_ok = True)

    print("----->>>> Log path: %s" % opt['log'])
    print("----->>>> Data set path: %s" % opt['data_path'])

def setSeed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True