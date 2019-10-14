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
import scipy as sc
from scipy import spatial

def get_ranks(des1, des2):
    # desc1 is [N x 128], des2 is [N x 128] or [N+D x 128] if using D distractor keypoints

    # [image1Descriptors x image2Descriptors] distance matrix to match with nearest neighbor
    allDist     = spatial.distance.cdist(des1, des2, metric='euclidean')
    print('all distances are calculated')
    # sort indices of image2Descriptors according to their closeness to image1Descriptors
    des2SortIdx = np.argsort(allDist, axis=1)
    print('all distances are sorted')
    #print("Got fooled by {} distractors.".format(np.sum(des2SortIdx[:,0:10]>500)))

    # ground-truth matches are on the main diagonal of the left hand side [N x N] submatrix, get ranks of those elements
    ranks       = [np.where(des2SortIdx[ddx1, :] == ddx1)[0][0] for ddx1, _ in enumerate(des1)]
    print('ranks are calculated')

    return ranks
