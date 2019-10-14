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
import os

import sys
import cv2

sys.path.insert (0, os.getcwd())
sys.path.insert (0, os.getcwd().split('baselines')[0])

from configs.defaults import _C as cfg
import argparse


from tqdm import tqdm
from baselines.matching.utils.metrics import get_ranks
import cv2
import torch

from modules.hardnet.models import HardNet
import tensorflow as tf

np.random.seed(42)

class HPatchesDataset():

    def __init__(self, images_dir, root_dir, sequences_type):

        self.images_dir = images_dir
        self.root_dir = root_dir
        self.sequences_type = sequences_type

        self.skip_sequences = ['v_talent',  'v_artisans', 'v_there', 'v_grace',
                               'v_sunseason',  'v_colors', 'v_astronautis',  'v_underground',
                               'i_leuven', 'i_contruction', 'i_crownnight',  'i_dc']
        self.folders = ['2']

        self.sequences = [x for x in sorted(os.listdir(self.root_dir))
                          if self.sequences_type in x and x not in self.skip_sequences]

        self.padTo = 1500
        self.anchors, self.positives = self.get_pair_of_images_and_keypoints()

    def __len__(self):
        return len(self.anchors)

    def get_pair_of_images_and_keypoints(self):

        anchors, positives = [], []
        for idx in tqdm(range(len(self.sequences)), total=len(self.sequences)):

            sequence =  self.sequences[idx]

            kps_path = os.path.join(args.hpatches_keypoints, sequence, '2')
            source_keypoints = np.load(os.path.join(kps_path, 'keypoints_source.npy'))
            target_keypoints = np.load(os.path.join(kps_path, 'keypoints_target.npy'))

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

        img_a_keypoints, img_p_keypoints = image_data[0][1].squeeze(), image_data[1][1].squeeze()

        keypoints_a, keypoints_p = np.array([[x[0],x[1]] for x in img_a_keypoints]), np.array([[x[0],x[1]] for x in img_p_keypoints])
        scales_a, scales_p = np.array([x[2] for x in img_a_keypoints]), np.array([x[2] for x in img_p_keypoints])
        orientations_a, orientations_p = np.array([x[3] for x in img_a_keypoints]),np.array([x[3] for x in img_p_keypoints])

        if image_anchor.shape[1] > self.padTo or image_anchor.shape[2] > self.padTo:
            print(image_data[0][0])
            raise RuntimeError("Image {} exceeds acceptable size, can't apply padding beyond image size.".format(image_data[0][0]))

        fillHeight = self.padTo - image_anchor.shape[1]
        fillWidth = self.padTo - image_anchor.shape[2]
        padLeft = int(np.round(fillWidth / 2))
        padRight = int(fillWidth - padLeft)
        padUp = int(np.round(fillHeight / 2))
        padDown = int(fillHeight - padUp)

        image_anchor = np.pad(image_anchor,
                     pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
                     mode='reflect')  # use either "reflect" or "symmetric"


        keypoints_locations_a,  keypoints_locations_p = [], []

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


        image_positive = np.pad(image_positive,
                     pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
                     mode='reflect')  # use either "reflect" or "symmetric"

        for kpIDX, kp_loc in enumerate(keypoints_p):  # iterate over keypoints

            normKp_p = 2 * np.array([[(kp_loc[0] + padLeft) / (self.padTo),
                                    (kp_loc[1] + padUp) / (self.padTo)]]) - 1

            keypoints_locations_p.append(normKp_p)


        image_positive =  torch.from_numpy(np.array(image_positive))
        image_anchor =  torch.from_numpy(np.array(image_anchor))

        orientations_a, orientations_p = np.array([np.deg2rad(orient) for orient in orientations_a]), \
                                        np.array([np.deg2rad(orient) for orient in orientations_p])


        theta_a = [torch.from_numpy(np.array(keypoints_locations_a)).float().squeeze(),
                 torch.from_numpy(scales_a).float(),
                 torch.from_numpy(np.array(orientations_a)).float(),
                 torch.from_numpy(np.array(keypoints_a)).float().squeeze()]

        theta_p = [torch.from_numpy(np.array(keypoints_locations_p)).float().squeeze(),
                 torch.from_numpy(scales_p).float(),
                 torch.from_numpy(np.array(orientations_p)).float(),
                 torch.from_numpy(np.array(keypoints_p)).float().squeeze()]

        return image_data[0][0], image_data[1][0], \
               image_anchor,image_positive, \
               theta_a, theta_p


def get_descriptors_from_list_of_keypoints(model_extractor, model, dataset):

    max_desc_amount = 20000

    pbar      = tqdm(enumerate(dataset),total=len(dataset))

    print(len(dataset.anchors))
    total_rank_0 = []

    all_descriptors_a, all_descriptors_p = [], []

    for batch_idx, data in pbar:

        image_a_path, image_p_path, image_anchor, image_positive, theta_a, theta_p = data
        image_a_filename = os.path.basename(image_a_path)
        image_p_filename = os.path.basename(image_p_path)

        imgs_a, img_keypoints_a = image_anchor.to(device), [theta_a[0].to(device), theta_a[1].to(device), theta_a[2].to(device)]
        imgs_p, img_keypoints_p = image_positive.to(device), [theta_p[0].to(device), theta_p[1].to(device), theta_p[2].to(device)]

        with torch.no_grad():
            torch.cuda.empty_cache()
            if args.method == "GeoDesc":

                descriptors_a, patches_ = model_extractor({image_a_filename: imgs_a}, img_keypoints_a,
                                                 [image_a_filename] * len(img_keypoints_a[0]))
                patches_a = np.squeeze(patches_).to("cpu").numpy()
                del descriptors_a, patches_
                torch.cuda.empty_cache()
                descriptors_p, patches_ = model_extractor({image_p_filename: imgs_p}, img_keypoints_p,
                                                 [image_p_filename] * len(img_keypoints_p[0]))

                patches_p = np.squeeze(patches_).to("cpu").numpy()
                del descriptors_p, patches_
                torch.cuda.empty_cache()
                import third_party.geodesc.examples.image_matching_utils as geodescMatch
                # reshape patches to [batchSize x patchSize x patchSize]
                # do patch-wise z-standardization as in example script extract_features_of_hpatches.py
                patches_a = np.array([(patch - np.mean(patch)) / (np.std(patch) + 1e-8) for patch in patches_a])
                patches_p = np.array([(patch - np.mean(patch)) / (np.std(patch) + 1e-8) for patch in patches_p])

                # do dimension expansion within image_matching
                # patches = np.expand_dims(patches, axis=-1)

                descriptors_a = geodescMatch.main(session=model, patches=patches_a)
                descriptors_p = geodescMatch.main(session=model, patches=patches_p)

            if args.method == "SIFT":

                descriptors_a, patches_a = model_extractor({image_a_filename: imgs_a}, img_keypoints_a,
                                                 [image_a_filename] * len(img_keypoints_a[0]))
                del descriptors_a
                torch.cuda.empty_cache()
                descriptors_p, patches_p = model_extractor({image_p_filename: imgs_p}, img_keypoints_p,
                                                 [image_p_filename] * len(img_keypoints_p[0]))
                del descriptors_p

                # SIFT expects patches to be of shape [N x 1 x pxSize x pxSize]
                # (this corresponds to the shape that the STN module outputs)

                # map pixel values from [0, 1] to [0, 255]
                patches_a = torch.clamp(torch.round(patches_a * 255), 0, 255)
                patches_p = torch.clamp(torch.round(patches_p * 255), 0, 255)

                with torch.no_grad(): descriptors_a = model(patches_a)
                with torch.no_grad(): descriptors_p = model(patches_p)

                # descriptors are of shape [N x 128]
                descriptors_a = np.round(512. * descriptors_a.data.cpu().numpy()).astype(np.float32)
                descriptors_p = np.round(512. * descriptors_p.data.cpu().numpy()).astype(np.float32)


            all_descriptors_a.extend(descriptors_a)
            all_descriptors_p.extend(descriptors_p)

            if len(all_descriptors_a) > max_desc_amount: break

    print(len(all_descriptors_a))
    rank = get_ranks(np.array(all_descriptors_a), np.array(all_descriptors_p))

    stats = {}
    stats["rankCounts"] = np.zeros(len(all_descriptors_a))

    for rankOccurs in rank: stats["rankCounts"][rankOccurs] += 1
    cumCounts = np.cumsum(stats["rankCounts"])
    with np.errstate(all='ignore'): stats["rankCDF"] = cumCounts / cumCounts[-1]
    total_rank_0.append(stats["rankCDF"][0])
    print('Total stats as retrieval : ' + str(stats["rankCDF"][0]))

    return stats["rankCDF"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="HardNet Training")

    parser.add_argument(
        "--config_file", default= "/cvlabsrc1/cvlab/datasets_anastasiia/descriptors/iccv_2019/hardnet_ptn/configs/init_one_example_ptn_96.yml", help="path to config file", type=str
    )

    parser.add_argument(
        "--hpatches_dataset", default= "dl/HPatches/hpatches-sequences-release/", help="path to config file", type=str
    )

    parser.add_argument(
        "--sequence_type", default= "v_", help="v_ - viewpoint i_ - illumination", type=str
    )

    parser.add_argument(
        "--hpatches_keypoints", default= "/cvlabdata2/cvlab/datasets_patrick/data/hpatches/hpatches-SIFT-keypoints/", help="path to config file", type=str
    )

    parser.add_argument(  "--method", default= "SIFT", help="method", type=str )

    parser.add_argument(  "--scale", default= 12, help="int", type=int )


    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print(args)

    if args.config_file != "": cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if not cfg.TRAINING.NO_CUDA:
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))

    hpatches_dataset = HPatchesDataset(args.hpatches_dataset, args.hpatches_keypoints, args.sequence_type)

    get_descriptors_from_list_of_keypoints(model_extractor, model, hpatches_dataset)