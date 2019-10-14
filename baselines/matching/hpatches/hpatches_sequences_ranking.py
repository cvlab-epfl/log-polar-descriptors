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
import os
import sys
import cv2

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd().split('baselines')[0])

from modules.hardnet.models import HardNet
from tqdm import tqdm
from baselines.matching.utils.metrics import get_ranks

from configs.defaults import _C as cfg
import argparse

np.random.seed(42)


class HPatchesDataset():
    def __init__(self, images_dir, root_dir, type):

        self.images_dir = images_dir
        self.root_dir = root_dir
        self.type = type

        self.skip_sequences = [
            'v_talent', 'v_artisans', 'v_there', 'v_grace', 'v_sunseason',
            'v_colors', 'v_astronautis', 'v_underground', 'i_leuven',
            'i_contruction', 'i_crownnight', 'i_dc'
        ]

        self.sequences = [
            x for x in sorted(os.listdir(self.root_dir))
            if self.type in x and x not in self.skip_sequences
        ]

        self.folders = ['2']

        self.padTo = 1500
        self.anchors, self.positives = self.get_pair_of_images_and_keypoints()

    def __len__(self):
        return len(self.anchors)

    def get_pair_of_images_and_keypoints(self):

        anchors, positives = [], []
        for idx in tqdm(range(len(self.sequences)), total=len(self.sequences)):

            sequence = self.sequences[idx]

            kps_path = os.path.join(args.hpatches_keypoints, sequence, '2')
            source_keypoints = np.load(
                os.path.join(kps_path, 'keypoints_source.npy'))
            target_keypoints = np.load(
                os.path.join(kps_path, 'keypoints_target.npy'))

            img_source = os.path.join(self.images_dir, sequence, '1.ppm')
            img_target = os.path.join(self.images_dir, sequence, '2.ppm')

            anchors.append([img_source, source_keypoints])
            positives.append([img_target, target_keypoints])

        return anchors, positives

    def __getitem__(self, idx):

        image_data = self.anchors[idx], self.positives[idx]

        image_anchor = cv2.imread(image_data[0][0], cv2.IMREAD_GRAYSCALE)
        image_anchor = np.expand_dims(image_anchor, 0)

        image_positive = cv2.imread(image_data[1][0], cv2.IMREAD_GRAYSCALE)
        image_positive = np.expand_dims(image_positive, 0)

        img_a_keypoints, img_p_keypoints = image_data[0][1].squeeze(
        ), image_data[1][1].squeeze()

        keypoints_a, keypoints_p = np.array([
            [x[0], x[1]] for x in img_a_keypoints
        ]), np.array([[x[0], x[1]] for x in img_p_keypoints])
        scales_a, scales_p = np.array([x[2]
                                       for x in img_a_keypoints]), np.array(
                                           [x[2] for x in img_p_keypoints])
        orientations_a, orientations_p = np.array([
            x[3] for x in img_a_keypoints
        ]), np.array([x[3] for x in img_p_keypoints])

        if image_anchor.shape[1] > self.padTo or image_anchor.shape[
                2] > self.padTo:
            print(image_data[0][0])
            raise RuntimeError(
                "Image {} exceeds acceptable size, can't apply padding beyond image size."
                .format(image_data[0][0]))

        fillHeight = self.padTo - image_anchor.shape[1]
        fillWidth = self.padTo - image_anchor.shape[2]
        padLeft = int(np.round(fillWidth / 2))
        padRight = int(fillWidth - padLeft)
        padUp = int(np.round(fillHeight / 2))
        padDown = int(fillHeight - padUp)

        image_anchor = np.pad(
            image_anchor,
            pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
            mode='reflect')  # use either "reflect" or "symmetric"

        keypoints_locations_a, keypoints_locations_p = [], []

        for kpIDX, kp_loc in enumerate(keypoints_a):  # iterate over keypoints

            normKp_a = 2 * np.array([[(kp_loc[0] + padLeft) / (self.padTo),
                                      (kp_loc[1] + padUp) / (self.padTo)]]) - 1

            keypoints_locations_a.append(normKp_a)

        fillHeight = self.padTo - image_positive.shape[1]
        fillWidth = self.padTo - image_positive.shape[2]
        padLeft = int(np.round(fillWidth / 2))
        padRight = int(fillWidth - padLeft)
        padUp = int(np.round(fillHeight / 2))
        padDown = int(fillHeight - padUp)

        image_positive = np.pad(
            image_positive,
            pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
            mode='reflect')  # use either "reflect" or "symmetric"

        for kpIDX, kp_loc in enumerate(keypoints_p):  # iterate over keypoints

            normKp_p = 2 * np.array([[(kp_loc[0] + padLeft) / (self.padTo),
                                      (kp_loc[1] + padUp) / (self.padTo)]]) - 1

            keypoints_locations_p.append(normKp_p)

        image_positive = torch.from_numpy(np.array(image_positive))
        image_anchor = torch.from_numpy(np.array(image_anchor))

        orientations_a, orientations_p = np.array([np.deg2rad(orient) for orient in orientations_a]), \
                                        np.array([np.deg2rad(orient) for orient in orientations_p])

        theta_a = [
            torch.from_numpy(
                np.array(keypoints_locations_a)).float().squeeze(),
            torch.from_numpy(scales_a).float(),
            torch.from_numpy(np.array(orientations_a)).float(),
            torch.from_numpy(np.array(keypoints_a)).float().squeeze()
        ]

        theta_p = [
            torch.from_numpy(
                np.array(keypoints_locations_p)).float().squeeze(),
            torch.from_numpy(scales_p).float(),
            torch.from_numpy(np.array(orientations_p)).float(),
            torch.from_numpy(np.array(keypoints_p)).float().squeeze()
        ]

        return image_data[0][0], image_data[1][0], \
               image_anchor,image_positive, \
               theta_a, theta_p


def get_descriptors_from_list_of_keypoints(model, dataset):

    # limit amount of descs
    max_desc_amount = 20000

    pbar = tqdm(enumerate(dataset), total=len(dataset))

    print(len(dataset.anchors))
    total_rank_0 = []

    sequences, all_descriptors_a, all_descriptors_p = [], [], []

    for batch_idx, data in pbar:

        image_a_path, image_p_path, image_anchor, image_positive, theta_a, theta_p = data
        img_a_filename, img_p_filename = os.path.basename(
            image_a_path), os.path.basename(image_p_path)

        sequences.append(image_a_path)

        imgs_a, img_keypoints_a = image_anchor.to(device), [
            theta_a[0].to(device), theta_a[1].to(device), theta_a[2].to(device)
        ]
        imgs_p, img_keypoints_p = image_positive.to(device), [
            theta_p[0].to(device), theta_p[1].to(device), theta_p[2].to(device)
        ]

        with torch.no_grad():

            descriptors_a, patches_a = model(
                {img_a_filename: imgs_a}, img_keypoints_a,
                [img_a_filename] * len(img_keypoints_a[0]))

            descriptors_p, patches_p = model(
                {img_p_filename: imgs_p}, img_keypoints_p,
                [img_p_filename] * len(img_keypoints_p[0]))

            descriptors_a, descriptors_p = descriptors_a.cpu().data.numpy(
            ).squeeze(), descriptors_p.cpu().data.numpy().squeeze()

            all_descriptors_a.extend(descriptors_a), all_descriptors_p.extend(
                descriptors_p)

            if len(all_descriptors_a) > max_desc_amount: break

    print(sequences)
    print(len(all_descriptors_a))

    rank = get_ranks(np.array(all_descriptors_a), np.array(all_descriptors_p))

    stats = {}
    stats["rankCounts"] = np.zeros(len(all_descriptors_a))

    for rankOccurs in rank:
        stats["rankCounts"][rankOccurs] += 1
    cumCounts = np.cumsum(stats["rankCounts"])
    with np.errstate(all='ignore'):
        stats["rankCDF"] = cumCounts / cumCounts[-1]
    total_rank_0.append(stats["rankCDF"][0])
    print('Total stats as retrieval : ' + str(stats["rankCDF"][0]))

    return stats["rankCDF"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HardNet Training")

    parser.add_argument(
        "--config_file",
        default=
        "/cvlabsrc1/cvlab/datasets_anastasiia/descriptors/iccv_2019/hardnet_ptn/configs/init_one_example_ptn_96.yml",
        help="path to config file",
        type=str)

    parser.add_argument("--hpatches_dataset",
                        default="dl/HPatches/hpatches-sequences-release/",
                        help="path to config file",
                        type=str)

    parser.add_argument(
        "--hpatches_keypoints",
        default=
        "/cvlabdata2/cvlab/datasets_patrick/data/hpatches/hpatches-SIFT-keypoints/",
        help="path to config file",
        type=str)

    parser.add_argument("--sequence_type",
                        default="v_",
                        help="v_ - viewpoint i_ - illumination",
                        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print(args)

    if args.config_file != "": cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if not cfg.TRAINING.NO_CUDA:
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    hpatches_dataset = HPatchesDataset(args.hpatches_dataset,
                                       args.hpatches_keypoints,
                                       args.sequence_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HardNet(transform=cfg.TEST.TRANSFORMER,
                    coords=cfg.TEST.COORDS,
                    patch_size=cfg.TEST.IMAGE_SIZE,
                    scale=cfg.TEST.SCALE,
                    is_desc256=cfg.TEST.IS_DESC_256,
                    orientCorrect=cfg.TEST.ORIENT_CORRECTION)

    checkpoint = os.path.join(cfg.TEST.MODEL_WEIGHTS)

    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()
    model.to(device)

    ranks = get_descriptors_from_list_of_keypoints(model, hpatches_dataset)
