import clip
import math
import numpy as np
import pandas as pd
from collections import OrderedDict

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils.functions import SpatialTransformer, VecInt

class LK_encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(LK_encoder, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Conv3d(in_cs, out_cs, 3, 1, 1)
        self.large = nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding)
        self.one = nn.Conv3d(in_cs, out_cs, 1, 1, 0)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        x = x1 + x2 + x3 + x

        return self.prelu(x)

class encoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=3, stride=1, padding=1):
        super(encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class decoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_cs, out_cs, kernel_size, stride, padding, output_padding),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)

class MLP(nn.Module):

    def __init__(self, init_dim, in_cs, out_cs):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(init_dim, in_cs),
            nn.ReLU(inplace=True),
            nn.Linear(in_cs, in_cs*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_cs*2, out_cs),
        )

    def forward(self, x):
        return self.mlp(x)

class clipSCPFlow3D(nn.Module):

    label2text_dict = OrderedDict([
        # [0, 'background'], # 0: background added later after text description
        [1, 'Left-Cerebral-White-Matter'],
        [2, 'Left-Cerebral-Cortex'],
        [3, 'Left-Lateral-Ventricle'],
        [4, 'Left-Inf-Lat-Ventricle'],
        [5, 'Left-Cerebellum-White-Matter'],
        [6, 'Left-Cerebellum-Cortex'],
        [7, 'Left-Thalamus'],
        [8, 'Left-Caudate'],
        [9, 'Left-Putamen'],
        [10, 'Left-Pallidum'],
        [11, '3rd-Ventricle'],
        [12, '4th-Ventricle'],
        [13, 'Brain-Stem'],
        [14, 'Left-Hippocampus'],
        [15, 'Left-Amygdala'],
        [16, 'Left-Accumbens'],
        [17, 'Left-Ventral-DC'],
        [18, 'Left-Vessel'],
        [19, 'Left-Choroid-Plexus'],
        [20, 'Right-Cerebral-White-Matter'],
        [21, 'Right-Cerebral-Cortex'],
        [22, 'Right-Lateral-Ventricle'],
        [23, 'Right-Inf-Lat-Ventricle'],
        [24, 'Right-Cerebellum-White-Matter'],
        [25, 'Right-Cerebellum-Cortex'],
        [26, 'Right-Thalamus'],
        [27, 'Right-Caudate'],
        [28, 'Right-Putamen'],
        [29, 'Right-Pallidum'],
        [30, 'Right-Hippocampus'],
        [31, 'Right-Amygdala'],
        [32, 'Right-Accumbens'],
        [33, 'Right-Ventral-DC'],
        [34, 'Right-Vessel'],
        [35, 'Right-Choroid-Plexus'],
    ])

    abbr2full_dict = OrderedDict([
        ['vit', 'ViT-L/14@336px'],
        ['res', 'RN50x64'],
    ])

    def __init__(self, in_cs, out_cs, clip_backbone='vit'):

        super(clipSCPFlow3D, self).__init__()

        self.clip_backbone = self.abbr2full_dict[clip_backbone]
        self.init_text_embeddings()

        self.generate_flow_x = MLP(self.init_dim, in_cs, out_cs)
        self.generate_flow_y = MLP(self.init_dim, in_cs, out_cs)
        self.generate_flow_z = MLP(self.init_dim, in_cs, out_cs)

        self.flow = nn.Conv3d(out_cs,3,3,1,1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def init_text_embeddings(self):

        label2text_dict = self.label2text_dict
        transformed_dict = {key: value.replace('-', ' ').lower() for key, value in label2text_dict.items()}

        clip_dict = {key: f"A magnetic resonance imaging of a {value} in human brain." for key, value in transformed_dict.items()}

        text_segments = clip.tokenize(list(clip_dict.values())).to('cuda')
        model, _ = clip.load(self.clip_backbone, 'cuda')
        model.eval()
        model.requires_grad_(False)
        text_features = model.encode_text(text_segments).float() # 35 x 512
        _, _, V = torch.linalg.svd(text_features)
        background_vector = V[:, -1].unsqueeze(0)
        self.text_features = torch.cat([background_vector, text_features], dim=0) # 36 x 512
        self.text_features.requires_grad_(False)
        self.text_features = self.text_features.float()
        print("Successfully initialized *CLIP* embedded anatomical features for %d classes using %s backbone." % (self.text_features.shape[0], self.clip_backbone))

        self.init_dim = self.text_features.shape[1]

    def forward(self, x, y_seg=None):

        base_x = self.generate_flow_x(self.text_features) # 36 x out_cs
        base_y = self.generate_flow_y(self.text_features) # 36 x out_cs
        base_z = self.generate_flow_z(self.text_features) # 36 x out_cs

        y_seg = nn.functional.interpolate(y_seg.float(),scale_factor=0.5,mode='nearest',recompute_scale_factor=0) # bs x 1 x h x w x d

        flow_filter_x = F.embedding(y_seg.long().squeeze(dim=1), base_x).permute(0, 4, 1, 2, 3).contiguous() # bs x out_cs x h x w x d
        flow_filter_y = F.embedding(y_seg.long().squeeze(dim=1), base_y).permute(0, 4, 1, 2, 3).contiguous() # bs x out_cs x h x w x d
        flow_filter_z = F.embedding(y_seg.long().squeeze(dim=1), base_z).permute(0, 4, 1, 2, 3).contiguous() # bs x out_cs x h x w x d

        flow_x = (flow_filter_x * x).sum(dim=1, keepdim=True)
        flow_y = (flow_filter_y * x).sum(dim=1, keepdim=True)
        flow_z = (flow_filter_z * x).sum(dim=1, keepdim=True)
        scp_flow = torch.cat([flow_x, flow_y, flow_z], dim=1)

        x = scp_flow + self.flow(x)

        return x

class brainTextSCFComplex(nn.Module):

    def __init__(self,
            start_channel = '64',  # N_s in the paper
            scp_dim = '2048',      # C_{\phi} in the paper
            num_classes = '36',    # Number of anatomical regions including background
            diff_int = '0',        # whether to use diffeomorphic integration
            clip_backbone = 'vit', # CLIP backbone, vit or res, denotes ViT-L/14@336px and RN50x64 respectively
            lk_size = '5',         # kernel size of LK encoder
            img_size = '(160,192,224)', # input image size
        ):

        super(brainTextSCFComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.scp_dim = int(scp_dim)
        self.num_classes = int(num_classes)
        self.diff_int = int(diff_int)
        self.clip_backbone = clip_backbone
        self.lk_size = int(lk_size)
        self.img_size = eval(img_size)

        print("start_channel: %d, scp_dim: %d, num_classes: %d, diff_int: %d, clip_backbone: %s, lk_size: %d, img_size: %s" % (self.start_channel, self.scp_dim, self.num_classes, self.diff_int, self.clip_backbone, self.lk_size, self.img_size))

        N_s = self.start_channel
        K_s = self.lk_size

        self.flow = clipSCPFlow3D(self.scp_dim, N_s*2, self.clip_backbone)

        self.eninput = encoder(2, N_s)
        self.ec1 = encoder(N_s, N_s)
        self.ec2 = encoder(N_s, N_s * 2, 3, 2, 1) # stride=2
        self.ec3 = LK_encoder(N_s * 2, N_s * 2, K_s, 1, K_s//2) # LK encoder
        self.ec4 = encoder(N_s * 2, N_s * 4, 3, 2, 1) # stride=2
        self.ec5 = LK_encoder(N_s * 4, N_s * 4, K_s, 1, K_s//2) # LK encoder
        self.ec6 = encoder(N_s * 4, N_s * 8, 3, 2, 1) # stride=2
        self.ec7 = LK_encoder(N_s * 8, N_s * 8, K_s, 1, K_s//2) # LK encoder
        self.ec8 = encoder(N_s * 8, N_s * 8, 3, 2, 1) # stride=2
        self.ec9 = LK_encoder(N_s * 8, N_s * 8, K_s, 1, K_s//2) # LK encoder

        self.dc1 = encoder(N_s * 16, N_s * 8, kernel_size=3, stride=1)
        self.dc2 = encoder(N_s * 8,  N_s * 4, kernel_size=3, stride=1)
        self.dc3 = encoder(N_s * 8,  N_s * 4, kernel_size=3, stride=1)
        self.dc4 = encoder(N_s * 4,  N_s * 2, kernel_size=3, stride=1)
        self.dc5 = encoder(N_s * 4,  N_s * 4, kernel_size=3, stride=1)
        self.dc6 = encoder(N_s * 4,  N_s * 2, kernel_size=3, stride=1)

        self.up1 = decoder(N_s * 8, N_s * 8)
        self.up2 = decoder(N_s * 4, N_s * 4)
        self.up3 = decoder(N_s * 2, N_s * 2)
        self.up4 = decoder(N_s * 2, N_s * 2)

        self.transformer = SpatialTransformer(self.img_size)
        self.integrate = VecInt((s//2 for s in self.img_size), 7)

    def forward(self, x, y, y_seg, registration=False):

        source, target = x, y
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        flow_field = self.flow(d2, y_seg)

        if self.diff_int==1:
            preint_flow = flow_field / 2.
            flow_field = self.integrate(preint_flow) * 2.
        else:
            preint_flow = flow_field
            flow_field = flow_field

        flow_field = torch.nn.functional.interpolate(flow_field, size=[160, 192, 224], mode='trilinear', align_corners=False)
        y_source = self.transformer(source, flow_field)

        if not registration:
            return (y_source, preint_flow), flow_field
        else:
            return y_source, flow_field