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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,
                 transform="PTN",
                 coords="log",
                 resolution=32,
                 SIFTscale=12.0,
                 onGPU=True,
                 squareSize=1500.0):
        super(Transformer, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and onGPU else "cpu")
        if torch.cuda.is_available() and onGPU: torch.cuda.set_device(0)

        # handle passed arguments
        self.transform = transform  # "STN" or "PTN"
        self.coords = coords  # "linear" or "log"
        self.resolution = resolution  # e.g. 12.0
        self.SIFTscale = SIFTscale  # constant scale multiplier for all keypoints, e.g. 12 for opencv SIFT scale

        # declare parameters correcting for differences in width or height:
        # these values may need to be changed if the input images are no longer
        # square [1500 x 1500] px² shape (may also need to change sampling procedure!)
        self.maxHeight = self.maxWidth = squareSize
        self.rescaleSTNtoSIFT = 1 / self.maxWidth
        self.rescalePTNtoSIFT = 0.5
        self.pi = torch.Tensor([np.pi]).to(self.device)

    # Spatial transformer network forward function
    def stn(self, img, theta, imgIDs=None):

        kpLoc = theta[0]
        scaling = theta[1]
        rotation = theta[2]
        scaling = self.rescaleSTNtoSIFT * self.SIFTscale * scaling

        # get [batchSize x 2 x 3] affine transformation matrix
        affMat = torch.empty(len(scaling), 2, 3)
        affMat[:, 0, 0] = torch.cos(rotation) * scaling
        affMat[:, 0, 1] = -scaling * torch.sin(rotation)
        affMat[:, 0, 2] = kpLoc[:, 0]
        affMat[:, 1, 0] = scaling * torch.sin(rotation)
        affMat[:, 1, 1] = torch.cos(rotation) * scaling
        affMat[:, 1, 2] = kpLoc[:, 1]

        if self.dictProcess and imgIDs is not None:
            # allocate space for extracted patches, take care to preserve the patch ordering
            patches = torch.empty(self.batchSize, 1, self.resolution,
                                  self.resolution).to(self.device)

            for key, val in img.items():
                val = val.unsqueeze_(0).to(self.device)
                # get indices of all keypoints in batch placed on the current image
                indices = [
                    idx for idx, idStr in enumerate(imgIDs) if idStr == key
                ]

                root = np.sqrt(len(indices))
                if root % 1 == 0:
                    # work-around to missing broadcasting / batch-processing of multiple keypoints on a common image
                    grid = F.affine_grid(affMat[indices, :, :],
                                         self.getGridSize(val, len(indices)))
                    xyPln = int(
                        self.resolution *
                        root)  # this line requires to check for root%1==0
                    grid = grid.view(
                        1, xyPln, xyPln, 2
                    )  # F.affine_grid can't broadcast a single image onto a batch of grids so we stack in the x-y plane
                    transforms = F.grid_sample(val.float(),
                                               grid.to(self.device).float())
                    patches[indices, :, :, :] = transforms.view(
                        len(indices), 1, self.resolution, self.resolution
                    )  # reshape to [100 x 1 x self.resolution x self.resolution]
                else:
                    # loop over each of the image's keypoints
                    for idx in indices:  # iterate over current image's keypoints (can't vectorize, would require duplicating image x batch dimension)
                        grid = F.affine_grid(
                            affMat[idx, :, :].unsqueeze_(0),
                            self.getGridSize(val)
                        )  # get [-1 x 1]² grids and apply affine transformations
                        patches[idx, 0, :, :] = F.grid_sample(
                            val.float(),
                            grid.to(self.device).float()
                        )  # do bilinear interpolation to sample values on the grid

        else:
            grid = F.affine_grid(affMat.float(), self.getGridSize(
                img))  # get [-1 x 1]² grids and apply affine transformations
            patches = F.grid_sample(
                img.float(),
                grid.to(self.device).float(
                ))  # do bilinear interpolation to sample values on the grid

        return patches

    # Polar transformer network forward function
    def ptn(self, img, theta, imgIDs=None):

        kpLoc = theta[0]
        scaling = theta[1]

        radius_factor = self.rescalePTNtoSIFT * self.SIFTscale * scaling

        W = self.maxWidth  # get width
        maxR = radius_factor  # rescale W

        # get grid resolution, assumes identical resolutions across dictionary if processing dictionary
        if self.dictProcess:
            gridSize = self.getGridSize(
                list(img.values())[0].unsqueeze(1), self.batchSize
            )  #self.getGridSize(list(img.values())[0].unsqueeze(-1).expand(self.batchSize,-1,-1,1)) #self.getGridSize(list(img.values())[0])
        else:
            gridSize = self.getGridSize(img)

        # get [self.batchSize x self.resolution x self.resolution x 2] grids with values in [-1,1],
        # define grids or call torch function and apply unit transform
        ident = torch.from_numpy(
            np.array(self.batchSize * [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        grid = F.affine_grid(ident, gridSize)
        grid_y = grid[:, :, :, 0].view(self.batchSize, -1)
        grid_x = grid[:, :, :, 1].view(self.batchSize, -1)

        grid_y = grid_y.float().to(self.device)
        grid_x = grid_x.float().to(self.device)
        maxR = torch.unsqueeze(maxR, -1).expand(-1, grid_y.shape[-1])

        # get radius of polar grid with values in [1, maxR]
        normGrid = (grid_y + 1) / 2
        if self.coords == "log": r_s_ = torch.exp(normGrid * torch.log(maxR))
        if self.coords == "linear": r_s_ = 1 + normGrid * (maxR - 1)

        # convert radius values to [0, 2maxR/W] range
        r_s = (r_s_ - 1) / (maxR - 1) * (
            W /
            self.maxWidth) * 2 * maxR / W  # r_s equals r^{x^t/W} in eq (9-10)

        # y is from -1 to 1; theta is from 0 to 2pi
        t_s = (
            grid_x + 1
        ) * self.pi  # tmin_threshold_distance_s equals \frac{2 pi y^t}{H} in eq (9-10)

        # use + kpLoc to deal with origin, i.e. (kpLoc[:, 0],kpLoc[:, 1]) denotes the origin (x_0,y_0) in eq (9-10)
        xLoc = torch.unsqueeze(kpLoc[:, 0], -1).expand(-1, grid_x.shape[-1])
        yLoc = torch.unsqueeze(kpLoc[:, 1], -1).expand(-1, grid_y.shape[-1])
        x_s = r_s * torch.cos(
            t_s
        ) + xLoc  # see eq (9) : theta[:,0] shifts each batch entry by the kp's x-coords
        y_s = r_s * torch.sin(
            t_s
        ) + yLoc  # see eq (10): theta[:,1] shifts each batch entry by the kp's y-coords

        # tensorflow grid is of shape [self.batchSize x 3 x self.resolution**2],
        # pytorch grid is of shape [self.batchSize x self.resolution x self.resolution x 2]
        # x_s and y_s are of shapes [1 x self.resolution**2]
        # bilinear interpolation in tensorflow takes _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

        # reshape polar coordinates to square tensors and append to obtain [self.batchSize x self.resolution x self.resolution x 2] grid
        polargrid = torch.cat(
            (x_s.view(self.batchSize, self.resolution, self.resolution, 1),
             y_s.view(self.batchSize, self.resolution, self.resolution, 1)),
            -1)

        if self.dictProcess and imgIDs is not None:
            # allocate space for extracted patches, take care to preserve the patch ordering
            patches = torch.empty(self.batchSize, 1, self.resolution,
                                  self.resolution).to(self.device)

            for key, val in img.items():
                val = val.unsqueeze_(0).to(self.device).float()

                # get indices of all keypoints in batch placed on the current image
                indices = [
                    idx for idx, idStr in enumerate(imgIDs) if idStr == key
                ]

                root = np.sqrt(len(indices))
                if root % 1 == 0:
                    # work-around to missing broadcasting / batch-processing of multiple keypoints on a common image
                    grid = polargrid[
                        indices, :, :, :]  # choose those grids in the batch that correspond to the current image
                    xyPln = int(
                        self.resolution *
                        root)  # this line requires to check for root%1==0
                    grid = grid.view(
                        1, xyPln, xyPln, 2
                    )  # F.affine_grid can't broadcast a single image onto a batch of grids so we stack in the x-y plane
                    transforms = F.grid_sample(val.float(),
                                               grid.to(self.device).float())
                    patches[indices, :, :, :] = transforms.view(
                        len(indices), 1, self.resolution, self.resolution
                    )  # reshape to [100 x 1 x self.resolution x self.resolution]
                else:
                    # loop over each of the image's keypoints
                    for idx in indices:  # iterate over current image's keypoints (can't vectorize, would require duplicating image x batch dimension)
                        patches[idx, 0, :, :] = F.grid_sample(
                            val, polargrid[idx, :, :].unsqueeze(0).to(
                                self.device).float()
                        )  # do bilinear interpolation to sample values on the grid
        else:
            patches = F.grid_sample(
                img, polargrid
            )  # do bilinear interpolation to sample values on the grid

        return patches

    def getGridSize(self, x, batchSize=1):
        gridSize = [*x.shape]
        gridSize[-2:] = 2 * [self.resolution]
        gridSize[0] = batchSize
        gridSize = torch.Size(gridSize)
        return gridSize

    def rad2deg(self, rad):  # pyTorch implementation of np.rad2deg(rad)
        deg = 180.0 * rad / self.pi
        return deg

    def roll(
            self, x, n
    ):  # pyTorch implementation of np.roll(x, torch.tensor([n]), axis=1)
        # roll along the image's height dimension (dim=1)
        n = n % self.resolution
        x = torch.cat((x[:, -n:, :], x[:, :-n, :]), dim=1)
        return x

    # correct PTN patches via rolling across angle-axis, assumes orient is given in radians
    def ptnOrient(self, patch, orient):

        degPerPx = 360 / self.resolution
        # map angle difference (clockwise turns, in rad) to pixel-shift
        # (equivalent to angle2Px = np.round(np.rad2deg(orient)/degPerPx).astype(np.int32))
        angle2Px = torch.round(self.rad2deg(orient) / degPerPx).int()
        # rotation in cartesian coordinates are upwards-shifts on the angle-axis in polar coordinates
        patch = torch.stack([
            self.roll(patch[adx, :, :, :], angle)
            for adx, angle in enumerate(angle2Px)
        ])
        return patch

    def forward(self, input):
        img, theta = input[0], input[1]
        imgIDs = None if len(input) < 3 else input[2]

        self.dictProcess = isinstance(
            img, dict
        )  # if received a dictionary then look up images for memory-efficient processing
        self.batchSize = theta[0].shape[
            0]  # get batch size, i.e. number of keypoints to process

        if self.transform == "STN":
            x = self.stn(img, theta, imgIDs)  # transform the input via STN
            x = x / 255.0  # standardize pixel values in [0,1]

        if self.transform == "PTN":

            x = self.ptn(img, theta, imgIDs)  # transform the input via PTN
            x = self.ptnOrient(x, theta[2])  # and apply orientation correction
            x = x / 255.0  # standardize pixel values in [0,1]

        return x
