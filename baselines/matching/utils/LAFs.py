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

import numpy as np
import torch
from copy import deepcopy

def convertLAFs_to_A23format(LAFs):
    # https://github.com/ducha-aiki/affnet/blob/master/LAF.py
    sh = LAFs.shape
    if (len(sh) == 3) and (sh[1] == 2) and (sh[2] == 3):  # n x 2 x 3 classical [A, (x;y)] matrix
        work_LAFs = deepcopy(LAFs)
    elif (len(sh) == 2) and (sh[1] == 7):  # flat format, x y scale a11 a12 a21 a22
        work_LAFs = np.zeros((sh[0], 2, 3))
        work_LAFs[:, 0, 2] = LAFs[:, 0]
        work_LAFs[:, 1, 2] = LAFs[:, 1]
        work_LAFs[:, 0, 0] = LAFs[:, 2] * LAFs[:, 3]
        work_LAFs[:, 0, 1] = LAFs[:, 2] * LAFs[:, 4]
        work_LAFs[:, 1, 0] = LAFs[:, 2] * LAFs[:, 5]
        work_LAFs[:, 1, 1] = LAFs[:, 2] * LAFs[:, 6]
    elif (len(sh) == 2) and (sh[1] == 6):  # flat format, x y s*a11 s*a12 s*a21 s*a22
        work_LAFs = np.zeros((sh[0], 2, 3))
        work_LAFs[:, 0, 2] = LAFs[:, 0]
        work_LAFs[:, 1, 2] = LAFs[:, 1]
        work_LAFs[:, 0, 0] = LAFs[:, 2]
        work_LAFs[:, 0, 1] = LAFs[:, 3]
        work_LAFs[:, 1, 0] = LAFs[:, 4]
        work_LAFs[:, 1, 1] = LAFs[:, 5]
    else:
        raise ValueError('Unknown LAF format')

    return work_LAFs


def bsvd2x2(As):
    # https://github.com/ducha-aiki/affnet/blob/master/LAF.py

    Su = torch.bmm(As, As.permute(0, 2, 1))
    phi = 0.5 * torch.atan2(Su[:, 0, 1] + Su[:, 1, 0] + 1e-12, Su[:, 0, 0] - Su[:, 1, 1] + 1e-12)
    Cphi = torch.cos(phi)
    Sphi = torch.sin(phi)
    U = torch.zeros(As.size(0), 2, 2)
    if As.is_cuda:
        U = U.cuda()
    U[:, 0, 0] = Cphi
    U[:, 1, 1] = Cphi
    U[:, 0, 1] = -Sphi
    U[:, 1, 0] = Sphi
    Sw = torch.bmm(As.permute(0, 2, 1), As)
    theta = 0.5 * torch.atan2(Sw[:, 0, 1] + Sw[:, 1, 0] + 1e-12, Sw[:, 0, 0] - Sw[:, 1, 1] + 1e-12)
    Ctheta = torch.cos(theta)
    Stheta = torch.sin(theta)
    W = torch.zeros(As.size(0), 2, 2)
    if As.is_cuda:
        W = W.cuda()
    W[:, 0, 0] = Ctheta
    W[:, 1, 1] = Ctheta
    W[:, 0, 1] = -Stheta
    W[:, 1, 0] = Stheta
    SUsum = Su[:, 0, 0] + Su[:, 1, 1]
    SUdif = torch.sqrt((Su[:, 0, 0] - Su[:, 1, 1]) ** 2 + 4 * Su[:, 0, 1] * Su[:, 1, 0] + 1e-12)
    if As.is_cuda:
        SIG = torch.zeros(As.size(0), 2, 2).cuda()
        SIG[:, 0, 0] = torch.sqrt((SUsum + SUdif) / 2.0)
        SIG[:, 1, 1] = torch.sqrt((SUsum - SUdif) / 2.0)
    else:
        SIG = torch.zeros(As.size(0), 2, 2)
        SIG[:, 0, 0] = torch.sqrt((SUsum + SUdif) / 2.0)
        SIG[:, 1, 1] = torch.sqrt((SUsum - SUdif) / 2.0)
    S = torch.bmm(torch.bmm(U.permute(0, 2, 1), As), W)
    C = torch.sign(S)
    C[:, 0, 1] = 0
    C[:, 1, 0] = 0
    V = torch.bmm(W, C)
    return (U, SIG, V)


def check_angle_return_rotation(u):
    angles = []

    for i in range(0, len(u)):
        angle_rad = np.arctan2(u[i][1][0], u[i][0][0])
        angle = np.rad2deg(angle_rad)
        angles.append(angle)
    return np.array(angles)

def getLAFelongationfromLAFs(LAFs):
    # https://github.com/ducha-aiki/affnet/blob/master/LAF.
    LAFs = torch.from_numpy(LAFs).float()
    u, s, v = bsvd2x2(LAFs[:, :2, :2])

    rotation_angles = []
    for idx, coordinate in enumerate(LAFs):
        angle = np.array(np.arctan2(u[idx][1][0], u[idx][0][0]))
        rotation_angles.append(angle)

    return np.array(rotation_angles)

def getLAFelongation(keypoints_format):
    # https://github.com/ducha-aiki/affnet/blob/master/LAF.
    LAFs = convertLAFs_to_A23format(keypoints_format)
    coordinates = keypoints_format[:,3:]
    LAFs = torch.from_numpy(LAFs).float()
    u, s, v = bsvd2x2(LAFs[:, :2, :2])

    rotation_angles = []
    for idx, coordinate in enumerate(coordinates):
        if (coordinate[0] == coordinate[3]) and (coordinate[1] == -coordinate[2]):
            angle = np.arctan2(coordinate[2], coordinate[0])
        else:
            angle = np.array(np.arctan2(u[idx][1][0], u[idx][0][0]))
        rotation_angles.append(angle)

    return np.array(rotation_angles)
