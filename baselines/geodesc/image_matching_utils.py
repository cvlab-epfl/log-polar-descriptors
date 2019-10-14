#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
Conduct pair-wise image matching.
"""

import os
import sys
import time

from threading import Thread
from queue import Queue

import cv2
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

model_path = "../model/geodesc.pb"
batch_size = 512


def extract_deep_features_directly(sess, patches, qtz=False):
    # resize patches to be suitable for geodesc
    # patches  = [cv2.resize(patch, (32, 32)) for patch in patches]

    num_patch = patches.shape[0]

    if num_patch % batch_size > 0:
        loop_num = int(np.floor(float(num_patch) / float(batch_size)))
    else:
        loop_num = int(num_patch / batch_size - 1)

    def _worker(patch_queue, sess, all_feat):
        """The worker thread."""
        while True:
            patch_data = patch_queue.get()
            if patch_data is None:
                return
            feat = sess.run(
                "squeeze_1:0",
                feed_dict={"input:0": np.expand_dims(patch_data, -1)})
            all_feat.append(feat)
            patch_queue.task_done()

    all_feat = []
    patch_queue = Queue()
    worker_thread = Thread(target=_worker, args=(patch_queue, sess, all_feat))
    worker_thread.daemon = True
    worker_thread.start()

    # enqueue
    for i in range(loop_num + 1):
        if i < loop_num:
            patch_queue.put(patches[i * batch_size:(i + 1) * batch_size])
        else:
            patch_queue.put(patches[i * batch_size:])
    # poison pill
    patch_queue.put(None)
    # wait for extraction.
    worker_thread.join()

    all_feat = np.concatenate(all_feat, axis=0)
    # quantize output features.
    all_feat = (all_feat * 128 + 128).astype(np.uint8) if qtz else all_feat
    return all_feat


def main(argv=None, session=None, patches=None):  # pylint: disable=unused-argument
    """Program entrance."""

    sess = session
    # skip SIFT keypoint detection and directly operate on given patches
    deep_feat = extract_deep_features_directly(sess, patches, qtz=False)

    return deep_feat


if __name__ == '__main__':
    tf.app.run()
