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

import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd().split('baselines')[0])

import numpy as np
import torch
from modules.hardnet.models import HardNet
from tqdm import tqdm
from PIL import Image
from baselines.matching.utils.metrics import get_ranks
import cv2
from importlib import import_module

from configs.defaults import _C as cfg
import argparse

np.random.seed(42)

import tensorflow as tf


class AMOSDataset():
    '''https://github.com/pultarmi/AMOS_patches'''
    def __init__(self, root_dir, padTo=1500):

        self.root_dir = root_dir
        # read Train and Test folders
        self.test_path = os.path.join(self.root_dir, 'Train')
        # read  data - [0] - patches, [1] - LAFs, [2] - view from which patches were extracted
        self.test_LAFs = torch.load(os.path.join(self.root_dir,
                                                 'train_patches.pt'),
                                    map_location=torch.device('cpu'))
        self.images_in_views = np.load(
            os.path.join(self.root_dir, 'images_in_views.npy')).item()

        # name of views - sorted - corresponding to the data [3] in the .pt file
        print(len(self.test_LAFs[1]))

        self.views = sorted(os.listdir(self.test_path))
        print(len(self.views))
        self.skip_views = [
            '00034154', '00036180', '00036169', '00011611', '00004431'
        ]
        self.padTo = padTo
        self.all_images, self.anchors, self.positives = self.get_pair_of_images_and_keypoints(
        )

    def __len__(self):
        return len(self.all_images)

    def get_pair_of_images_and_keypoints(self):

        anchors, positives = [], []
        all_images = {}

        for idx in tqdm(range(len(self.test_LAFs[1])),
                        total=len(self.test_LAFs[1])):

            LAFs = self.test_LAFs[1][idx]
            view_idx = int(self.test_LAFs[2][idx].numpy()[0])
            view = self.views[view_idx]

            if view in self.skip_views: continue

            #images_in_view = os.listdir(os.path.join(self.test_path, view))
            images_in_view = self.images_in_views[view]
            images, keypoints, orientations = np.random.choice(images_in_view, 2, replace=False),\
                                              LAFs[:,2].cpu().data.numpy(), \
                                              np.zeros(2)

            images = [
                os.path.join(self.test_path, view, image) for image in images
            ]
            read_img = Image.open(images[0])
            width, height = read_img.size

            if width > self.padTo or height > self.padTo:
                continue

            scale = np.sqrt(LAFs[0, 0] * LAFs[1, 1] - LAFs[0, 1] * LAFs[1, 0] +
                            1e-10).cpu().data.numpy()

            anchors.append([images[0], keypoints, orientations[0], scale])
            positives.append([images[1], keypoints, orientations[1], scale])

            for idx, img in enumerate(images):
                if img in all_images.keys():
                    all_images[img].append(
                        [keypoints, orientations[idx], scale])
                else:
                    all_images[img] = [[keypoints, orientations[idx], scale]]

        return list(all_images.items()), anchors, positives

    def __getitem__(self, idx):

        image_data = self.all_images[idx]
        image_path, keypoints, orientations, scales = image_data[0], \
                                np.array([x[0] for x in image_data[1]]),\
                                np.array([x[1] for x in image_data[1]]),\
                                np.array([x[2] for x in image_data[1]])

        image_anchor = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_anchor = np.expand_dims(image_anchor, 0)

        if image_anchor.shape[1] > self.padTo or image_anchor.shape[
                2] > self.padTo:
            print(image_path)
            raise RuntimeError(
                "Image {} exceeds acceptable size, can't apply padding beyond image size."
                .format(image_anchor_path))

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

        image_anchor = torch.from_numpy(np.array(image_anchor))

        keypoints_locations = []
        for kpIDX, kp_loc in enumerate(keypoints):  # iterate over keypoints

            normKp = 2 * np.array([[(kp_loc[0] + padLeft) / (self.padTo),
                                    (kp_loc[1] + padUp) / (self.padTo)]]) - 1
            keypoints_locations.append(normKp)

        theta = [
            torch.from_numpy(np.array(keypoints_locations)).float().squeeze(),
            torch.from_numpy(scales).float(),
            torch.from_numpy(np.array(orientations)).float(),
            torch.from_numpy(np.array(keypoints)).float().squeeze()
        ]

        return image_path, image_anchor, theta


def reshape_descriptors_for_anchors_positives(anchors, positives, descriptors,
                                              patches_for_images):

    new_anchors, new_positives = [], []

    for idx in range(len(anchors)):
        anchor, positive = anchors[idx], positives[idx]
        anchor_name, positive_name = os.path.basename(
            anchor[0]), os.path.basename(positive[0])
        anchor_keypoints, positive_keypoints = str(anchor[1]), str(positive[1])

        desc_anchor = descriptors[anchor_name][anchor_keypoints]
        desc_positive = descriptors[positive_name][positive_keypoints]

        new_anchors.append(desc_anchor)
        new_positives.append(desc_positive)

    return new_anchors, new_positives


def get_descriptors_from_list_of_keypoints(settings, model_extractor, model,
                                           dataset):

    pbar = tqdm(enumerate(dataset), total=len(dataset))

    stats = {}
    stats["rankCounts"] = np.zeros(len(dataset.anchors))
    print(len(dataset.anchors))

    descriptors_for_images = {}
    patches_for_images = {}

    for batch_idx, data in pbar:

        image_path, image_anchor, theta = data
        image_filename = os.path.basename(image_path)

        imgs, img_keypoints = image_anchor.to(device), [
            theta[0].to(device), theta[1].to(device), theta[2].to(device)
        ]

        with torch.no_grad():

            descriptors, patches_ = model_extractor(
                {image_filename: imgs}, img_keypoints,
                [image_filename] * len(img_keypoints[0]))

            patches = np.squeeze(patches_).to("cpu").numpy()
            del descriptors

            if args.method == "GeoDesc":

                sys.path.insert(
                    0,
                    os.path.join(os.getcwd().split('baselines')[0],
                                 'third_party', 'geodesc'))

                from third_party.geodesc.examples import image_matching_utils
                # reshape patches to [batchSize x patchSize x patchSize]
                # do patch-wise z-standardization as in example script extract_features_of_hpatches.py
                patches_a = np.array([
                    (patch - np.mean(patch)) / (np.std(patch) + 1e-8)
                    for patch in patches
                ])

                # do dimension expansion within image_matching
                # patches = np.expand_dims(patches, axis=-1)

                descriptors = image_matching_utils.main(session=model,
                                                        patches=patches_a)

            if args.method == "SIFT":

                patches_a = torch.clamp(torch.round(patches_ * 255), 0, 255)
                with torch.no_grad():
                    descriptors = model(patches_a)
                # descriptors are of shape [N x 128]
                descriptors = np.round(
                    512. * descriptors.data.cpu().numpy()).astype(np.float32)

            if image_filename in descriptors_for_images:
                for idx, keypoint in enumerate(theta[3]):
                    descriptors_for_images[image_filename][str(
                        keypoint.data.cpu().numpy())] = descriptors[idx]

            else:
                descriptors_for_images[image_filename] = {}
                patches_for_images[image_filename] = {}

                for idx, keypoint in enumerate(theta[3]):
                    descriptors_for_images[image_filename][str(
                        keypoint.data.cpu().numpy())] = descriptors[idx]

    anchors_desc, positive_desc = reshape_descriptors_for_anchors_positives(
        dataset.anchors, dataset.positives, descriptors_for_images,
        patches_for_images)

    rank = get_ranks(np.array(anchors_desc), np.array(positive_desc))

    for rankOccurs in rank:
        stats["rankCounts"][rankOccurs] += 1
    cumCounts = np.cumsum(stats["rankCounts"])
    with np.errstate(all='ignore'):
        stats["rankCDF"] = cumCounts / cumCounts[-1]

    print(stats["rankCDF"][0])

    return stats["rankCDF"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HardNet Training")

    parser.add_argument(
        "--config_file",
        default=
        "/cvlabsrc1/cvlab/datasets_anastasiia/descriptors/iccv_2019/release/hardnet_ptn/configs/init_one_example_ptn_96.yml",
        help="path to config file",
        type=str)

    parser.add_argument("--amos_dataset",
                        default="dl/AMOS/AMOS_views_v3/",
                        help="path to config file",
                        type=str)

    parser.add_argument("--scale",
                        default=16,
                        help="path to config file",
                        type=int)

    parser.add_argument("--method",
                        default="SIFT",
                        help="path to config file",
                        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    print(args)
    settings = {}

    print(args.method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "SIFT":

        from third_party.pytorch_sift.pytorch_sift import SIFTNet
        model = SIFTNet(patch_size=32)
        model.eval()
        if torch.cuda.is_available(): model.cuda()

    if args.method == "GeoDesc":
        from third_party.geodesc.utils.tf import load_frozen_model
        modelPath = os.path.join("third_party/geodesc/model/geodesc.pb")
        # create deep feature extractor
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

        graph = load_frozen_model(modelPath, print_nodes=False)
        model = tf.Session(graph=graph,
                           config=tf.ConfigProto(gpu_options=gpu_options))

    if args.config_file != "": cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if not cfg.TRAINING.NO_CUDA:
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amos_dataset = AMOSDataset(args.amos_dataset, cfg.TRAINING.PAD_TO)

    model_extractor = HardNet(transform='STN',
                              coords=cfg.TEST.COORDS,
                              patch_size=cfg.TEST.IMAGE_SIZE,
                              scale=args.scale,
                              is_desc256=cfg.TEST.IS_DESC_256,
                              orientCorrect=cfg.TEST.ORIENT_CORRECTION)

    checkpoint = os.path.join(cfg.TEST.MODEL_WEIGHTS)

    model_extractor.load_state_dict(torch.load(checkpoint)['state_dict'])
    model_extractor.eval()

    model_extractor.to(device)

    get_descriptors_from_list_of_keypoints(settings, model_extractor, model,
                                           amos_dataset)
