import re
import os
import glob
import torch
import numpy as np

from torch.utils.data import DataLoader

from models import getModel
from loaders.oasis_pkl_loader import oasis_pkl_loader
from utils.functions import modelSaver, convert_state_dict

def loadDataset(opt, split = 'train'):

    dataset_name = opt['dataset']
    data_path = opt['data_path']

    if dataset_name == 'oasis_pkl':
        loader = oasis_pkl_loader(root_dir = data_path, split = split)
    else:
        raise ValueError('Unkown datasets: please define proper dataset name')

    print("----->>>> %s dataset is loaded ..." % dataset_name)

    return loader

def getDataLoader(opt, split='train'):

    if split == 'train':
        data_shuffle = True
        batch_size = opt['batch_size']
    else:
        data_shuffle = False
        batch_size = 1

    num_workers = opt['num_workers']
    print("----->>>> Loading %s dataset ..." % (split))
    dataset = loadDataset(opt, split)
    loader = DataLoader(dataset = dataset,
                        num_workers = num_workers,
                        batch_size = batch_size,
                        pin_memory = True,
                        shuffle = data_shuffle)
    print("----->>>> %s batch size: %d, # of %s iterations per epoch: %d" %  (split, batch_size, split, int(len(dataset) / batch_size)))

    return loader

def getModelSaver(opt):

    model_saver = modelSaver(opt['log'], opt['save_freq'], opt['n_checkpoints'])

    return model_saver

def findLastCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("net_epoch_(.*)_score_.*.pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        init_epoch = max(epochs_exist)
    else:
        init_epoch = 0

    score = None
    if init_epoch > 0:
        for file_ in file_list:
            file_name = "net_epoch_" + str(init_epoch) + "_score_(.*).pth.*"
            result = re.findall(file_name, file_)
            if result:
                score = result[0]
                break

    return_name = None
    if init_epoch > 0:
        return_name =  "net_epoch_" + str(init_epoch) + "_score_" + score + ".pth"

    return init_epoch, score, return_name

def findBestCheckpoint(save_path):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("best_score_(.*)_net_epoch_.*.pth.*", file_)
            if result:
                epochs_exist.append(result[0])
        score = max(epochs_exist)

        for file_ in file_list:
            file_name = "best_score_" + str(score) + "_net_epoch_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return_name = result[0]
                file_name = "best_score_" + str(score) + "_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                epoch = result[0]
                return epoch, score, return_name

    raise ValueError("can't find checkpoints")

def findCheckpointByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "net_epoch_" + str(epoch) + "_score_.*.pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def findBestDiceByEpoch(save_path, epoch):

    file_list = glob.glob(os.path.join(save_path, '*epoch*.pth'))
    if file_list:
        for file_ in file_list:
            file_name = "best_score_.*_net_epoch_" + str(epoch) + ".pth.*"
            result = re.findall(file_name, file_)
            if result:
                return result[0]

    raise ValueError("can't find checkpoints")

def getTrainModelWithCheckpoints(opt):

    model = getModel(opt)
    init_epoch, score, file_name = findLastCheckpoint(opt['log'])

    if init_epoch > 0:
        print("----->>>> Resuming model by loading epoch %s with dice %s" % (init_epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)))
        model.load_state_dict(states)

    return model, init_epoch

def getTestModelWithCheckpoints(opt):

    model = getModel(opt)
    file_name = 'unknown'
    epoch = '0'
    score = '0'
    which_model = 'unknown'

    if opt['load_ckpt'] == 'best':
        epoch, score, file_name = findBestCheckpoint(opt['log'])
        which_model = 'best'
    elif opt['load_ckpt'] == 'last':
        epoch, score, file_name = findLastCheckpoint(opt['log'])
        which_model = 'last'
    elif "epoch" in opt['load_ckpt']:
        epoch = opt['load_ckpt'].split('_')[1]
        file_name = findCheckpointByEpoch(opt['log'], epoch)
        which_model = str(epoch) + 'th'
    elif opt['load_ckpt'] == 'none':
        print("----->>>> No model is loaded")
    else:
        raise ValueError("Not either best, last or epoch")

    if file_name != 'unknown':
        print("----->>>> Resuming the %s model by loading epoch %s with dice %s" % (which_model, epoch, score))
        states = convert_state_dict(torch.load(os.path.join(opt['log'], file_name)), is_multi=opt['use_multi_gpus'])
        model.load_state_dict(states)

    info = {
        "file_name": file_name,
        "epoch": int(epoch),
        "score": float(score),
    }

    return model, info
