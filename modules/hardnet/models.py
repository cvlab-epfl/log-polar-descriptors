# Copyright 2019 EPFL, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.getcwd().split("/modules/hardnet")[0])
from modules.hardnet.utils import str2bool

import argparse
parser = argparse.ArgumentParser(description='PyTorch HardNet')

parser.add_argument('--hard-augm',
                    type=str2bool,
                    default=False,
                    help='turns on flip and 90deg rotation augmentation')

args, _ = parser.parse_known_args()

from modules.ptn.pytorch.models import Transformer
'''
Class with models definition
'''


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class HardNet(nn.Module):
    def __init__(self,
                 transform,
                 coords,
                 patch_size,
                 scale,
                 is_desc256,
                 orientCorrect=True):
        super(HardNet, self).__init__()

        self.transform = transform
        self.transform_layer = Transformer(transform=transform,
                                           coords=coords,
                                           resolution=patch_size,
                                           SIFTscale=scale)

        self.orientCorrect = orientCorrect

        # model processing patches of size [32 x 32] and giving description vectors of length 2**7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # initialize weights
        self.features.apply(weights_init)
        return

    def input_norm(self, x):

        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / \
               sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    # function to forward-propagate inputs through the network
    def forward(self, img, theta=None, imgIDs=None):

        if theta is None:  # suppose patches are directly given (as e.g. for external test data)
            patches = img
        else:  # extract keypoints from the whole image
            patches = self.transform_layer([img, theta, imgIDs])

        batchSize = patches.shape[0]

        if args.hard_augm:
            bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))

            if self.transform == "STN":
                # transpose to switch dimensions (only if STN)
                transpose = bernoulli.sample(torch.Size([batchSize]))
                patches = torch.cat([
                    torch.transpose(patch, 1, 2) if transpose[pdx] else patch
                    for pdx, patch in enumerate(patches)
                ]).unsqueeze(1)

            # flip the patches' first dimension
            mirrorDim1 = bernoulli.sample(torch.Size([batchSize]))
            patches = torch.cat([
                torch.flip(patch, [1]) if mirrorDim1[pdx] else patch
                for pdx, patch in enumerate(patches)
            ]).unsqueeze(1)

        x_features = self.features(self.input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x), patches


def weights_init(m):
    '''
    Conv2d module weight initialization method
    '''

    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
