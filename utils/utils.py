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
import math


# Calculates rotation matrix to euler angles (in radians)
def rotationMatrixToEulerAngles(R):
    # Check if this is a rotation matrix
    eps = 1e-6
    RtR = np.dot(np.transpose(R), R)
    norm = np.linalg.norm(np.identity(3, dtype=R.dtype) - RtR)
    assert (norm < eps)

    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > eps:
        # R is not singular
        r = np.arctan2(R[1, 0], R[0, 0])
        p = np.arctan2(-R[2, 0], sy)
        y = np.arctan2(R[2, 1], R[2, 2])
    else:
        r = 0
        p = np.arctan2(-R[2, 0], sy)
        y = np.arctan2(-R[1, 2], R[1, 1])

    # return rotation angles in radians (roll, pitch, yaw)
    return np.array([r, p, y])


# returns a list of splits for training, validation and test data
def splitTrainValTest(data, splitFracts):
    splits = []
    leftIDX = 0
    for _, fract in enumerate(splitFracts):
        # get end index of current split according to fraction
        rightIDX = leftIDX + math.ceil((fract * len(data)))

        # get the current split
        splits.append(data[leftIDX:rightIDX])

        # the current split's right index is the next split's left index
        leftIDX = rightIDX
    return splits


# translates an array data (with data.shape[0] samples) into a list of batches of size batchSize
def batch_feed(data, batchSize):
    batchedData = [
        data[idx:idx + batchSize] for idx in range(0, len(data), batchSize)
    ]
    return batchedData


# converts matrix to homogeneous coordinates
def mat2homogen(mat):
    return np.concatenate([mat, np.ones([1, mat.shape[1]])], axis=0)


# convert array of integers to 1-hot encoding matrix
def array2hot(array, classes):
    hot = np.eye(classes)[np.array(array).reshape(-1)]
    return hot.reshape(list(array.shape) + [classes])


def pointIsWithinImg(point, img):
    # point locations may be at subpixel precision, cast to integer values
    x = np.int(np.round(point[0]))
    y = np.int(np.round(point[1]))
    surpassesBorder = any([x < 0, x > img.shape[1], y < 0, y > img.shape[0]])
    return not surpassesBorder


# calculates the clockwise angle (in radians and degrees) between two vectors
def degOfVects(v1, v2):
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    rad = (ang1 - ang2) % (2 * np.pi)
    return rad, np.rad2deg(rad)


# convert subpixel coordinates to pixel coordinates safely, while checking coordinates stay within image
# IN : subpx is [nSamples x (x,y)] and image is [y x x]
# OUT: px is [nSamples x (x,y)]
def subpx2Px(subpx, image):
    px = np.round(subpx).astype(np.int32)
    px[:, 0] = np.clip(px[:, 0], 0, image.shape[1] - 1)
    px[:, 1] = np.clip(px[:, 1], 0, image.shape[0] - 1)
    return px


# function to clear all previously defined tensorflow flags,
# which may be needed e.g. when executing a script more than once
def clearFlagsTF(flags):
    keys_list = [keys for keys in flags._flags()]
    for keys in keys_list:
        flags.__delattr__(keys)
