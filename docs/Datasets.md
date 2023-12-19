## [OASIS dataset](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)
The dataset was provided by [Andrew Hoopes](https://www.nmr.mgh.harvard.edu/user/3935749) and [Adrian V. Dalca](http://www.mit.edu/~adalca/) in support of the HyperMorph research. If you utilize this dataset, please acknowledge the original work by citing the associated HyperMorph paper and adhere to the [OASIS Data Use Agreement](http://oasis-brains.org/#access).


 - [HyperMorph: Amortized Hyperparameter Learning for Image Registration](https://arxiv.org/abs/2101.01035).  
   Hoopes A, Hoffmann M, Fischl B, Guttag J, Dalca AV.
   IPMI 2021.

 - Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.  
    Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL.  
    Journal of Cognitive Neuroscience, 19, 1498-1507.

Inspired by [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), this dataset has been converted to `.pkl` format and is available for download via this Dropbox link:  [oasis_pkl](https://www.dropbox.com/scl/fo/ve4wancuxty69kulxmn10/h?rlkey=kygv9b16p70fh3gocj6c6l8l0&dl=0). To facilitate the use of our code with this dataset, a complete project setup is included in the shared folder. Once the necessary Python packages are installed, you can directly execute the project using the following command:

```
python train_brainreg.py -d oasis_pkl -m brainTextSCFComplex -bs 1 --epochs 501 --reg_w 0.1 start_channel=64 scp_dim=2048 diff_int=0 clip_backbone=vit
```

To use the pretrained mode, download the [complete project setup](https://www.dropbox.com/scl/fo/ve4wancuxty69kulxmn10/h?rlkey=kygv9b16p70fh3gocj6c6l8l0&dl=0), run the script with the following command in folder `./src` to get the npz files:
```
python test_brainreg.py -d oasis_pkl -m brainTextSCFComplex -bs 1 --is_submit 1 --load_ckpt ./../../../checkpoint/oasis_9002_64_2048_0_vit.pth start_channel=64 scp_dim=2048 diff_int=0 clip_backbone=vit
```
- `--is_submit`: Whether to create npz files for submission to the challenge.
- `--load_ckpt`: The type of the checkpoint to load, 'last' is from the latest checkpoint, 'best' is from the checkpoint with highest validation score, and a path such as './../../../checkpoint/oasis_9002_64_2048_0_vit.pth' directing to the checkpoint.

The npz files will be saved at `./textSCF/src/logs/oasis_pkl/brainTextSCFComplex/` where `textSCF` is the root of the code repository.

## Abdomen CT

Todo

## Cardiac Cine-MRI

Todo