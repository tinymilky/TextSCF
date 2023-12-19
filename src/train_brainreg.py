import os
import sys
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, dice_eval
from utils.loss import DiceLoss, Grad3d
from utils import getters, setters


def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt)
    model_saver = getters.getModelSaver(opt)

    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)

    criterion_sim = nn.MSELoss()
    criterion_reg = Grad3d(penalty='l2')
    criterion_dsc = DiceLoss(num_class=36)
    best_dsc = 0

    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        loss_all = AverageMeter()
        loss_sim_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_dsc_all = AverageMeter()
        for idx, data in enumerate(train_loader):
            model.train()
            data = [Variable(t.cuda()) for t in data[:4]]
            x, x_seg = data[0], data[1]
            y, y_seg = data[2], data[3]

            (y_pred, preint_flow), pos_flow = model(x, y, y_seg)

            # similarity loss
            sim_loss = criterion_sim(y_pred, y) * opt['loss_ws'][0]
            loss_sim_all.update(sim_loss.item(), y.numel())

            # regularization loss
            reg_loss = criterion_reg(preint_flow, y) * opt['loss_ws'][1]
            loss_reg_all.update(reg_loss.item(), y.numel())

            # dice loss
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            def_seg = model.transformer(x_seg_oh.float(), pos_flow.float())
            dsc_loss = criterion_dsc(def_seg, y_seg) * opt['loss_ws'][2]
            loss_dsc_all.update(dsc_loss.item(), y.numel())

            loss = sim_loss + reg_loss + dsc_loss
            loss_all.update(loss.item(), y.numel())

            adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Deformable Sim: {:.6f}, Reg: {:.6f}, DSC: {:.6f}'.format(idx+1, len(train_loader), loss.item(), sim_loss.item(), reg_loss.item(), dsc_loss.item()), end='\r', flush=True)

        print('Epoch {} train loss {:.4f}, Deformable Sim: {:.6f}, Reg: {:.6f}, DSC: {:.6f}'.format(epoch+1, loss_all.avg, loss_sim_all.avg, loss_reg_all.avg, loss_dsc_all.avg))

        '''
        To reproduce the results on the leaderboard, you can simply run training for 500 epochs and use the last checkpoint.
        '''

        train_dsc = 1-loss_dsc_all.avg
        best_dsc = max(train_dsc, best_dsc)
        model_saver.saveModel(model, epoch, train_dsc)
        print('Epoch {} train dice {:.6f}, best dice {:.6f}'.format(epoch+1, train_dsc, best_dsc))

        '''
        Or you can replace the above code with your custom validation code modified from the following snippet.
        '''

        #### replace with your own validation code here ####
        # eval_dsc = AverageMeter()
        # with torch.no_grad():
        #     for data in val_loader:
        #         model.eval()
        #         data = [Variable(t.cuda())  for t in data[:4]]
        #         x, x_seg = data[0], data[1]
        #         y, y_seg = data[2], data[3]

        #         _, pos_flow = model(x,y,y_seg,registration=True)
        #         def_out = reg_model(x_seg.cuda().float(), pos_flow)
        #         dsc = dice_eval(def_out.long(), y_seg.long(), 36)
        #         eval_dsc.update(dsc.item(), x.size(0))

        # best_dsc = max(eval_dsc.avg, best_dsc)
        # model_saver.saveModel(model, epoch, eval_dsc.avg)
        # print('Epoch {} validation dice {:.6f}, best dice {:.6f}'.format(epoch+1, eval_dsc.avg, best_dsc))
        #### replace with your own validation code here ####

if __name__ == '__main__':

    opt = {
        'img_size': (160, 192, 224), # input image size
        'loss_ws': [1., 0.1, 1.],    # sim, reg, dsc
        'logs_path': './logs',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 20,          # number of checkpoints to keep
        'power': 0.9,                # decay power
        'num_workers': 4,            # number of workers for data loading
    }

    '''
    The dataset path is the root folder of the dataset, where the 'datasets_path' will be joined with the 'dataset'. This is for the convenience of managing multiple datasets and neural networks.
    '''
    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'brainTextSCFComplex')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'oasis_pkl')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--epochs", type = int, default = 500)
    parser.add_argument("--reg_w", type = float, default = 0.1)
    parser.add_argument("--lr", type = float, default = 1e-4)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}

    opt['loss_ws'][1] = opt['reg_w']
    print("----->>>> loss weights: sim: {}, reg: {}, dsc: {}".format(opt['loss_ws'][0], opt['loss_ws'][1], opt['loss_ws'][2]))

    run(opt)

    '''
    Example command:
    python train_brainreg.py -d oasis_pkl -m brainTextSCFComplex -bs 1 --epochs 501 --reg_w 0.1 start_channel=64 scp_dim=2048 diff_int=0 clip_backbone=vit
    '''