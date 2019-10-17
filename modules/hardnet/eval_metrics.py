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
"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

"""
import numpy as np


def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(
        np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(
        labels[:threshold_index] ==
        0)  # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(
        labels[threshold_index:] ==
        0)  # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)
