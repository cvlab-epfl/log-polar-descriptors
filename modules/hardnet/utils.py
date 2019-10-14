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

import torch
import torch.nn.init
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# resize image to size 32x32
cv2_scale36 = lambda x: cv2.resize(
    x, dsize=(36, 36), interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(
    x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
cv2_scale_64 = lambda x: cv2.resize(
    x, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))

# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))
np_reshape_ = lambda x: np.reshape(x, (len(x), len(x[0]), 1))


def show_images(images, file_to_save, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    try:
        plt.savefig(file_to_save)
    except Exception as ex:
        print(ex)
    plt.close(fig)
    return plt


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def save_hist(folder, root, name, x, bins=50, range=(1, 4), cum=False):
    plt.cla()
    plt.clf()

    # n, bins, patches = plt.hist(x=x, bins=bins, color='#0504aa',
    #                              alpha=0.7, rwidth=0.85, cumulative=True, density = True,
    #                                  histtype='step')
    # n, bins, patches = plt.hist(x=x, bins=bins, weights=np.ones(len(x)) / len(x), color='#0504aa', rwidth=0.85)

    sns.distplot(x,
                 hist=True,
                 hist_kws=dict(cumulative=True, alpha=0.3),
                 bins=bins)
    sns.distplot(x,
                 hist_kws={
                     'weights': np.ones(len(x)) / len(x),
                     'alpha': 1.0,
                     "color": "y"
                 },
                 kde=False,
                 bins=bins)

    plt.xlim(1, 3)

    # hist_plot.savefig("output.png")
    #
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Keypoint scale ratio')
    # plt.ylabel('Probability density')
    # plt.title('Histogram of keypoint scale ratios f')
    # plt.xlim((1, 3))

    # Set a clean upper y-axis limit.

    if cum:
        name = name + '_cum'

    if 'STN' in root:
        plt.savefig(os.path.join(folder, name + '_stn_scales_hist.png'))
    else:
        plt.savefig(os.path.join(folder, name + '_ptn_scales_hist.png'))
