import os
import torch
import interpol
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, dice_binary, dice_eval, convert_pytorch_grid2scipy
from scipy.ndimage.interpolation import map_coordinates, zoom


def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    test_loader = getters.getDataLoader(opt, split='test')
    model, _ = getters.getTestModelWithCheckpoints(opt)

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_dsc_full_scipy = AverageMeter()
    eval_dsc_half_scipy = AverageMeter()
    eval_dsc_half_torch = AverageMeter()
    with torch.no_grad():
        for data_ori in test_loader:
            model.eval()
            data = [t.cuda() for t in data_ori[:4]]

            sample_idx = data_ori[4]
            _, sv_file_name = os.path.split(test_loader.dataset.total_list[sample_idx])
            sub_idxs = sv_file_name.split('.')[0][2:]
            sub1_idx, sub2_idx = sub_idxs.split('_')

            x, x_seg = data[0], data[1]
            y, y_seg = data[2], data[3]

            _, pos_flow = model(x,y,y_seg,registration=True)
            def_out, pytorch_grid = model.transformer(x_seg.cuda().float(), pos_flow.cuda(),is_grid_out=True,mode='nearest')
            dsc_trans = dice_eval(def_out.long(), y_seg.long(), 36)
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            dsc_raw = dice_eval(x_seg.long(), y_seg.long(), 36)
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            print("----->>>> Running on subject {} and {} <<<<-----".format(sub1_idx, sub2_idx))
            print("----->>>> Full resolution torch disp dice: %.4f" % dsc_trans)

            if opt['is_submit']:

                pytorch_grid = pytorch_grid.squeeze(0).permute(3,0,1,2)
                scipy_disp = convert_pytorch_grid2scipy(pytorch_grid.data.cpu().numpy())

                _,_,xx,yy,zz = x_seg.shape
                identity = np.meshgrid(np.arange(xx), np.arange(yy), np.arange(zz), indexing='ij')
                moving_warped = map_coordinates(x_seg.cpu().squeeze(0).squeeze(0).data.numpy(), identity + scipy_disp, order=0)

                dice_bs=[]
                for i in range(1,36,1):
                    dice_bs.append(dice_binary(moving_warped.copy(),y_seg.cpu().squeeze(0).squeeze(0).data.numpy().copy(), k = i))
                dsc_scipy_full = np.mean(dice_bs)
                print("----->>>> Full resolution scipy disp dice: %.4f" % dsc_scipy_full)

                eval_dsc_full_scipy.update(dsc_scipy_full, x.size(0))

                downsample_scipy_disp = np.array([zoom(scipy_disp[i], 0.5, order=2) for i in range(3)])

                flow_fp = os.path.join(opt['log'], 'task_03')
                os.makedirs(flow_fp, exist_ok=True)
                flow_fp = os.path.join(flow_fp, 'disp_'+sv_file_name.split('.')[0][2:] + '.npz')
                np.savez(flow_fp, np.array(downsample_scipy_disp).astype(np.float16))
                print("----->>>> Saved flow field to %s" % flow_fp)

                disp_field = np.load(flow_fp)['arr_0'].astype('float32')
                print("----->>>> Loaded flow field shape", disp_field.shape)
                torch_disp_field = torch.from_numpy(disp_field).unsqueeze(0).cuda()
                torch_disp_field = torch.cat([torch_disp_field[0:1,0:1],torch_disp_field[0:1,1:2],torch_disp_field[0:1,2:3]], dim=1)
                print("----->>>> Torch flow field shape", torch_disp_field.shape)

                ppram = dict(shape=[160, 192, 224], anchor='edge', bound='zero', interpolation=2)
                torch_disp_field = [interpol.resize(x, **ppram).unsqueeze(0) for x in torch_disp_field[0]]
                torch_disp_field = torch.cat(torch_disp_field, dim=0).unsqueeze(0)
                def_out = model.transformer(x_seg.cuda().float(), torch_disp_field.cuda(),mode='nearest')
                dsc_half_torch = dice_eval(def_out.long(), y_seg.long(), 36)
                eval_dsc_half_torch.update(dsc_half_torch.item(), x.size(0))
                print("----->>>> Half resolution torch disp dice: %.4f" % dsc_half_torch)

                disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
                moving_warped = map_coordinates(x_seg.cpu().squeeze(0).squeeze(0).data.numpy(), identity + disp_field, order=0)

                dice_bs=[]
                for i in range(1,36,1):
                    dice_bs.append(dice_binary(moving_warped.copy(),y_seg.cpu().squeeze(0).squeeze(0).data.numpy().copy(), k = i))
                dsc_scipy_half = np.mean(dice_bs)
                eval_dsc_half_scipy.update(dsc_scipy_half, x.size(0))
                print("----->>>> Half resolution scipy disp dice: %.4f" % dsc_scipy_half)

        print("----->>>> Avg Reg DSC: %.4f+-%.4f, Before Reg DSC: %.4f+-%.4f, Scipy Full DSC: %.4f+-%.4f, Scipy Half DSC: %.4f+-%.4f, Torch Half DSC: %.4f+-%.4f" % (eval_dsc_def.avg, eval_dsc_def.std, eval_dsc_raw.avg, eval_dsc_raw.std, eval_dsc_full_scipy.avg, eval_dsc_full_scipy.std, eval_dsc_half_scipy.avg, eval_dsc_half_scipy.std, eval_dsc_half_torch.avg, eval_dsc_half_torch.std))

        # save csv
        fp = os.path.join('logs', opt['dataset'], opt['model'], 'results.csv')
        df = pd.DataFrame({'avg_dice': eval_dsc_def.vals})
        df.to_csv(fp, index=False)


if __name__ == '__main__':

    opt = {
        'img_size': (160, 192, 224), # input image size
        'logs_path': './logs',       # path to saved logs
        'num_workers': 4,            # number of workers for data loading
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'VxmDense')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'OASIS')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--load_ckpt", type = str, default = "last") # best, last or epoch
    parser.add_argument("--is_submit", type = int, default = 0) # whether to save the flow field for learn2reg challenge submission

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}

    run(opt)

    '''
    Example command:
    python test_brainreg.py -d oasis_pkl -m brainTextSCFComplex -bs 1 --is_submit 1 start_channel=64 scp_dim=2048 diff_int=0 clip_backbone=vit --load_ckpt last
    '''