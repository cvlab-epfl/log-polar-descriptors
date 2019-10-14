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
import argparse
import random
import numpy as np

from scipy.spatial.distance import cdist
from tqdm import tqdm
import cv2 as cv

import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset

parser = argparse.ArgumentParser(description='Transformer HardNet')


class transformerData(Dataset):
    def __init__(self, cfg, sequences=None):

        self.cfg = cfg
        self.padTo = cfg.TRAINING.PAD_TO
        self.n_triplets = cfg.TRAINING.N_TRIPLETS  # number of triplet samples to be generated

        self.sequences_meta = {
        }  # dictionary containing all loaded sequences' meta information, self.sequences_meta[seqName][imagePairIdx][patchIdx]
        self.sequences_meta_untouched = {
        }  # dictionary containing meta information not updated yet
        self.rewrote_meta = False  # whether the meta information has already been updated once or not
        self.sequences_img = {
        }  # dictionary containing all loaded sequences' images, self.sequences_meta[seqName][imagePairIdx][{img,padLeft,padUp}]

        # construct indices of the network's input data set, get_pairs is called at the start of each epoch
        print('Generating {} triplets of data'.format(self.n_triplets))
        self.list_of_indices = self.get_pairs(sequences)

    # get ratio of anchor and positive patch scales for a given pair index
    def scale_filter(self, scales_img_a, scales_img_p, random_indx):
        scale_a, scale_p = scales_img_a[random_indx], scales_img_p[random_indx]
        ratio = scale_a / scale_p if scale_a > scale_p else scale_p / scale_a

        return ratio

    def load_images(self, sequences):
        for sequence in tqdm(sequences):
            # if sequence's images already loaded then omitt
            if sequence in self.sequences_img: continue
            print('Loading images of sequence: {}'.format(sequence))

            # for a given sequence get list of unique image pair IDs
            listIDs = [[pair["imgID"] for pair in pairID]
                       for pairID in self.sequences_meta[sequence].values()]
            imgIDs = np.unique(np.array(listIDs).flatten())

            for imgID in imgIDs:
                # load the corresponding image and cast from [Height x Width] to [1 x Heigth x Width]
                img = cv.imread(
                    os.path.join(self.cfg.INPUT.COLMAP_IMAGES, sequence,
                                 'dense/images', imgID + ".jpg"),
                    cv.IMREAD_GRAYSCALE)
                img = np.expand_dims(img, 0)

                # mirror-pad the input image for crops surpassing the image borders
                # and to make all images square-shaped as well as uniformly sized
                if img.shape[1] > self.padTo or img.shape[2] > self.padTo:
                    raise RuntimeError(
                        "Image {} exceeds acceptable size, can't apply padding beyond image size."
                        .format(imgID))

                fillHeight = self.padTo - img.shape[1]
                fillWidth = self.padTo - img.shape[2]
                padLeft = int(np.round(fillWidth / 2))
                padRight = int(fillWidth - padLeft)
                padUp = int(np.round(fillHeight / 2))
                padDown = int(fillHeight - padUp)

                img = np.pad(
                    img,  # needed to copy input image
                    pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
                    mode='reflect')  # use either "reflect" or "symmetric"

                # store image and number of padded pixels to the left and above of the coordinate origin
                if not sequence in self.sequences_img:
                    self.sequences_img[sequence] = {}
                if not imgID in self.sequences_img[sequence]:
                    self.sequences_img[sequence][imgID] = {}
                self.sequences_img[sequence][imgID]["img"] = img
                self.sequences_img[sequence][imgID]["padLeft"] = padLeft
                self.sequences_img[sequence][imgID]["padUp"] = padUp

    def rewrite_meta(self, sequences):
        for sequence in tqdm(sequences):
            print("Rewriting keypoint meta information for sequence {}".format(
                sequence))
            for imgPairID, pair_meta in self.sequences_meta[sequence].items(
            ):  # iterate through dictionary of image pair meta data

                # get the pair indices for both anchor and positive
                pair_indx_a = np.where(
                    np.sum(imgPairID.split('-') == pair_meta[0]["pairID"],
                           axis=1) == 2)[0][0]
                pair_indx_p = np.where(
                    np.sum(imgPairID.split('-') == pair_meta[1]["pairID"],
                           axis=1) == 2)[0][0]
                imgID_a, imgID_p = pair_meta[0]['imgID'], pair_meta[1]['imgID']

                self.updateKpCoordinates(sequence, imgPairID,
                                         [imgID_a, imgID_p],
                                         [pair_indx_a, pair_indx_p])
                self.updateKpOrientations(sequence, imgPairID,
                                          [imgID_a, imgID_p],
                                          [pair_indx_a, pair_indx_p])
                self.updateKpScales(sequence, imgPairID, [imgID_a, imgID_p],
                                    [pair_indx_a, pair_indx_p])

        self.rewrote_meta = True  # remember not to rewrite meta information again in any subsequent epoch

    def updateKpCoordinates(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            # look up by how many pixels the image's origin is shifted due to padding
            padLeft = self.sequences_img[sequence][
                imgIDAnchorPositive[0]]["padLeft"]
            padUp = self.sequences_img[sequence][
                imgIDAnchorPositive[0]]["padUp"]

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["locations"]
            for kpIDX, kp_loc in enumerate(
                    keypoints):  # iterate over keypoints

                # correct keypoint locations for padding and map from subpixel space to [-1,+1]x[-1,+1] grid,
                # notice that axes are transposed: keypoints are (x,y) but images are [Y x X]
                normKp = 2 * np.array([[(kp_loc[0] + padLeft) / (self.padTo),
                                        (kp_loc[1] + padUp) /
                                        (self.padTo)]]) - 1

                # update (x,y) keypoint locations
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["locations"][kpIDX] = normKp

    def updateKpOrientations(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["orientations"]
            for kpIDX, kp_orient in enumerate(
                    keypoints):  # iterate over keypoints

                # correct for different orientations of patches
                if self.cfg.TRAINING.ORIENT_CORRECTION:
                    # check whether image is first in pair, we take this image's SIFT orientations
                    firstImgID = imgPairID.split('-')[0]
                    isFirstImg = imgIDAnchorPositive[0] == firstImgID

                    # rotate patches of both images by the first image's SIFT orientations
                    # (add +90 degrees to correct for SIFT orientation origin)
                    if isFirstImg:
                        orient = np.deg2rad(kp_orient + 0)
                    else:  # if current image comes second then retrieve first image's orientations
                        firstPairIDX = pair_indx[1 - idxAnchorPositive]
                        # we may have in-place updated the first keypoint's orientations before, so need to look up untouched orientations
                        orient = self.sequences_meta_untouched[sequence][
                            imgPairID][1 - idxAnchorPositive]["keypoints"][
                                firstPairIDX]["orientations"][kpIDX]
                        orient = np.deg2rad(orient + 0)

                    # if patch comes from the first image then additionally correct for planar rotation difference
                    orientsDiffTrue = self.sequences_meta[sequence][imgPairID][
                        idxAnchorPositive]["keypoints"][pairIDX][
                            'orientsDiffTrue'][kpIDX]
                    orient += 0 if not isFirstImg else orientsDiffTrue
                else:
                    # rotate patches of first image by the first image's SIFT orientations
                    # and patches of second image by the second image's SIFT orientations
                    orient = np.deg2rad(kp_orient + 0)

                # update keypoint orientations
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["orientations"][kpIDX] = orient

    # function to update keypoint scales:
    # multiply by SIFT scaling constant and divide by image width
    # --> both operations are now directly done within the transformer module, so this function is now redundant
    def updateKpScales(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["scales"]

            for kpIDX, kp_scale in enumerate(
                    keypoints):  # iterate over keypoints

                # get scale of SIFT keypoints (and correct for width of image plus padding)
                scale = kp_scale

                # update keypoint scales
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["scales"][kpIDX] = scale

    # given a list of names of sequences already used in batch return a random name of unused sequence
    def select_random_sequence(self, already_used_sequence):
        '''
        Method randomly selects sequence name and checks whether is has been already used.

        Parameters
        ----------
        already_used_sequence - set of names of already used sequences

        Returns random sequence name
        -------
        '''

        sequences = list(self.sequences_meta.keys())
        seq = random.choice(sequences)

        # if sequence was already used in current batch, select another one
        while seq in already_used_sequence:
            seq = random.choice(sequences)
        return seq

    # given a sequence and an image pair ID, look up the individual image's pair indices
    def get_pair_idx(self, seq_name, img_pair_id):
        # get meta for selected image pair
        pair_meta = self.sequences_meta[seq_name][img_pair_id]

        # get indices in meta for selected image pair
        pair_idx_a = np.where(
            np.sum(img_pair_id.split('-') == pair_meta[0]["pairID"], axis=1) ==
            2)[0][0]
        pair_idx_p = np.where(
            np.sum(img_pair_id.split('-') == pair_meta[1]["pairID"], axis=1) ==
            2)[0][0]
        return pair_idx_a, pair_idx_p

    # given current sequence's name and list of used image pair IDs
    # return randomly drawn image pair ID not yet in list of used image pair IDs
    def draw_imgPairID(self, seq, used_imgPairIDs):
        '''
        Method to select random pair of images from sequence,
        validates already selected sequences and existence of image pair in sequence meta data

        Parameters
        ----------
        seq - current sequence
        used_imgPairIDs - set of already used image pairs

        Returns random image pair ID
        -------

        '''

        allpairIDs = list(self.sequences_meta[seq].keys())

        # draw random image pair ID and look up its keypoint pairs
        imgPairID = random.choice(allpairIDs)
        pair_idx_a, _ = self.get_pair_idx(seq, imgPairID)
        enoughKps = len(
            self.sequences_meta[seq][imgPairID][0]['keypoints'][pair_idx_a]
            ["keypointID"]) >= self.cfg.TRAINING.MIN_KEYPOINTS_PER_IMAGE

        # select only image pairs where min number of keypoints >= args.min_keypoints_per_image
        while not enoughKps:
            imgPairID = random.choice(allpairIDs)
            # if pair of images was already used, select another one
            if imgPairID in used_imgPairIDs:
                imgPairID = random.choice(allpairIDs)
                # validate if pair_id exists in meta, if not, select other image pair
                if imgPairID not in self.sequences_meta[seq]:
                    print(
                        'Current pair {} doesnt exist in sequence: {} '.format(
                            imgPairID, seq))
                    imgPairID = random.choice(allpairIDs)

            # get anchor indices in meta for selected image pair to see if there are enough keypoints (omit positives, will be the same amount of keypoints)
            pair_idx_a, _ = self.get_pair_idx(seq, imgPairID)
            enoughKps = len(
                self.sequences_meta[seq][imgPairID][0]['keypoints'][pair_idx_a]
                ["keypointID"]) >= self.cfg.TRAINING.MIN_KEYPOINTS_PER_IMAGE

        return imgPairID

    # for a given sequence and random pair index get its meta information (pair ID, keypoint locations, scale)
    # such that keypoints can be subsampled according to whether their meta information meets certain criteria
    def get_additional_info_from_meta(self, seq_name, img_pair_id):
        # get indices in meta for selected image pair
        pair_meta = self.sequences_meta[seq_name][img_pair_id]
        pair_idx_a, pair_idx_p = self.get_pair_idx(seq_name, img_pair_id)

        locations_img_a = pair_meta[0]['keypoints'][pair_idx_a]['locations']
        locations_img_p = pair_meta[1]['keypoints'][pair_idx_p]['locations']

        scales_img_a = pair_meta[0]['keypoints'][pair_idx_a]['scales']
        scales_img_p = pair_meta[1]['keypoints'][pair_idx_p]['scales']

        orientations_img_a = pair_meta[0]['keypoints'][pair_idx_a][
            'orientations']
        orientations_img_p = pair_meta[1]['keypoints'][pair_idx_p][
            'orientations']

        imgID_a = pair_meta[0]['imgID']
        imgID_p = pair_meta[1]['imgID']

        scale_correct_a = pair_meta[0]['keypoints'][pair_idx_a]['scaleMapping']
        orient_correct_a = pair_meta[1]['keypoints'][pair_idx_p][
            'orientsDiffTrue']

        return locations_img_a, locations_img_p, scales_img_a, scales_img_p, orientations_img_a, orientations_img_p, imgID_a, imgID_p, scale_correct_a, orient_correct_a

    # given list of names of sequences
    # adds the sequences' patch images and meta files to the classes corresponding dictionary properties
    def load_sequences(self, sequences):
        for sequence in tqdm(sequences):
            # (useful check not to re-load sequences after every training epoch)
            if len(self.sequences_meta) == len(sequences): continue
            print('Loading sequence: {}'.format(sequence))

            # set up dictionaries by sequence name
            self.sequences_meta[sequence] = np.load(
                os.path.join(self.cfg.INPUT.SOURCE, sequence) +
                '_meta.npy', allow_pickle=True).item()  # dictionary containing meta information
            self.sequences_meta_untouched[sequence] = np.load(
                os.path.join(self.cfg.INPUT.SOURCE, sequence) +
                '_meta.npy', allow_pickle=True).item()

    # get_pairs is called at the beginning of each epoch
    def get_pairs(self, sequences):
        '''
        Main method that generates list of all patch pairs
        (hard dependencies on n_triplets, min_keypoints_per_image_pair and batch_size)
        for one epoch, unique sequences and pairs in batch.

        Parameters
        ----------
        sequences - sequences to use during generation

        Returns generated list of indices of patch pairs
        -------

        '''

        # read sequences, images and rewrite meta information---these dictionaries remain unchanged across epochs
        self.load_sequences(
            sequences)  # load sequences (patches and meta information)
        self.load_images(sequences)  # load images of sequences
        if not self.rewrote_meta:
            self.rewrite_meta(
                sequences
            )  # update meta information, subsampling criteria should use the updated information

        list_of_indices = [
        ]  # sequence name, image pair ID, patch pair index and [imgID_a,imgID_p] [kpLoc_a,kpLoc_p], [kpOrient_a,kpOrient_p], [kpScale_a,kpScale_p]
        current_batch_filled = 0  # number of samples within batch
        total_generated_pairs = 0  # total number of samples
        already_used_sequence, already_used_pair = set(), set()

        while total_generated_pairs < self.n_triplets:
            # select random sequence to draw two paired images from
            random_seq = self.select_random_sequence(already_used_sequence)
            already_used_sequence.add(
                random_seq)  # keep track of used sequences

            # draw image pair ID not used yet
            random_pair = self.draw_imgPairID(random_seq, already_used_pair)
            already_used_pair.add(random_pair)  # keep track of used images

            # check current batch is filled
            if current_batch_filled != self.cfg.TRAINING.BATCH_SIZE:
                count_selected_keypoints_per_image = 0

                # get locations, scales and orientations for image pair
                locations_img_a, locations_img_p, scales_img_a, scales_img_p, orientations_img_a, orientations_img_p, imgID_a, imgID_p, scale_correct_a, orient_correct_a = \
                    self.get_additional_info_from_meta(random_seq, random_pair)

                # set of already used indices and already used locations (of anchor, considering distance thresholding)
                used_indices, unique_locations_a = set(), []
                # set initial min distance, just to add first patch
                min_dist = 100

                # get random indices for selected image pair, shuffle order for each epoch
                pair_idx_a, _ = self.get_pair_idx(random_seq, random_pair)
                numberKps = len(self.sequences_meta[random_seq][random_pair][0]
                                ['keypoints'][pair_idx_a]["keypointID"])
                indices = np.random.permutation(numberKps)

                while count_selected_keypoints_per_image < self.cfg.TRAINING.MIN_KEYPOINTS_PER_IMAGE_PAIR:
                    if total_generated_pairs % 5000 == 0:
                        print('>> Generated pairs {}'.format(
                            total_generated_pairs),
                              flush=True)

                    # draw a random keypoint pair IDX
                    random_indx = random.choice(indices)

                    # get anchor location to check for minimum distance between current and previous anchors
                    location_a = locations_img_a[random_indx]
                    if len(unique_locations_a) > 0:
                        min_dist = cdist(np.array([location_a]),
                                         np.array(unique_locations_a)).min()

                    # compute scale ratio for current pair of keypoints
                    ratio = 1.0 if not self.cfg.TRAINING.ENABLE_SCALE_FILTERING else self.scale_filter(
                        scales_img_a, scales_img_p, random_indx)

                    # check if:
                    # (a) current patch index is not used yet
                    # (b) min_dist is above distance threshold
                    # (c) ratio is below ratio threshold

                    rescaleConst = 0.0007  # map from distance threshold from pixel space to [-1,1] regular grid space
                    # TODO: adjust this when changing image size 1500
                    while random_indx in used_indices \
                            or min_dist < (rescaleConst * self.cfg.TRAINING.MIN_THRESHOLD_DISTANCE) \
                            or ratio > self.cfg.TRAINING.MAX_RATIO:
                        # ... otherwise select new patch index
                        random_indx = random.choice(indices)
                        location_a = locations_img_a[random_indx]

                        # check for minimum distance between current and previous anchors
                        if len(unique_locations_a) > 0:
                            min_dist = cdist(
                                np.array([location_a]),
                                np.array(unique_locations_a)).min()
                        # compute scale ratio for current pair of keypoints
                        if self.cfg.TRAINING.ENABLE_SCALE_FILTERING:
                            ratio = self.scale_filter(scales_img_a,
                                                      scales_img_p,
                                                      random_indx)

                    # keep track of selected index and anchor location
                    used_indices.add(random_indx), unique_locations_a.append(
                        location_a)

                    # store in list:
                    # sequence name, image pair ID, patch pair index and meta information as well as ID of image
                    list_of_indices.append([random_seq, random_pair, random_indx, \
                                            [locations_img_a[random_indx], locations_img_p[random_indx]], \
                                            [scales_img_a[random_indx], scales_img_p[random_indx]], \
                                            [orientations_img_a[random_indx], orientations_img_p[random_indx]], \
                                            [imgID_a[0], imgID_p[0]], \
                                            [scale_correct_a[random_indx],orient_correct_a[random_indx]]
                                            ])

                    current_batch_filled += 1
                    total_generated_pairs += 1
                    count_selected_keypoints_per_image += 1

                # check if current batch is filled, if yes, reset sets
                if current_batch_filled == self.cfg.TRAINING.BATCH_SIZE:
                    already_used_sequence, already_used_pair = set(), set()
                    current_batch_filled = 0

            # check if current batch is filled, if yes, reset sets
            else:
                already_used_sequence, already_used_pair = set(), set()
                current_batch_filled = 0

        # self.list_of_indices = list_of_indices
        return list_of_indices

    # Dataset iterator called at runtime to get a new item from the batch
    def __getitem__(self, idx):
        # get sequence ID, image pair ID, patch pair IDX and meta information of current item in batch
        seqID, img_pair_idx, patch_pair_idx, locations, scales, orientations, imgID, scale_orient_correct = self.list_of_indices[
            idx]

        # get images of anchor and positive keypoint
        img_a = self.sequences_img[seqID][imgID[0]]
        img_p = self.sequences_img[seqID][imgID[1]]

        meta_a, meta_p = [locations[0], scales[0], orientations[0]], \
                         [locations[1], scales[1], orientations[1]]

        return (img_a, img_p, meta_a, meta_p, imgID[0], imgID[1],
                scale_orient_correct[0], scale_orient_correct[1])

    def __len__(self):
        # length of generated list
        return self.n_triplets


class transformerValTestData(Dataset):
    def __init__(self, cfg, is_test=False, test_name=None):
        self.cfg = cfg
        self.n_triplets = self.cfg.TRAINING.N_TRIPLETS  # number of triplet samples to be generated
        self.sequences_meta = {
        }  # dictionary containing all loaded sequences' meta information, self.sequences_meta[seqName][imagePairIdx][patchIdx]
        self.rewrote_meta = False  # whether the meta information has already been updated once or not, validation and test data is initialized only once anyways ...
        self.sequences_img = {
        }  # dictionary containing all loaded sequences' images
        self.padTo = cfg.TRAINING.PAD_TO  # mirror-padding the input images to squares of shape [self.padTo x self.padTo]

        # set to STN, PTN or PTNlinear, depending on the passed paths
        if 'STN' in self.cfg.INPUT.VAL_SPLIT:
            self.type = 'STN'
        else:
            self.type = 'PTN'

        src = self.cfg.INPUT.SOURCE if is_test else self.cfg.INPUT.VAL_SPLIT
        label = 'test' if is_test else 'val'
        suffix = '_p50_th20' if is_test else '_p20_th20'
        name = test_name if is_test else self.type

        # load data structures: a single file for all validation sequences and one file per test sequence

        # load sequence, pair_id, img1_idx, patch_a_idx, pair_id, img2_idx, patch_p_idx, match
        # (constructed via function create_validation_set in ValidationsetCreator.py)
        self.list_of_indices = np.load(src + '/' + name + '_' + label +
                                       '_sequence_indices' + suffix + '.npy', allow_pickle=True)
        # load meta information (a second time to update keypoint orientations on the second image),
        # alternatively: changing keypoint meta-information updating order may make re-loading unnecessary
        self.sequences_meta = np.load(src + '/' + name + '_' + label +
                                      '_sequence_meta' + suffix +
                                      '.npy', allow_pickle=True).item()
        self.sequences_meta_untouched = np.load(src + '/' + name + '_' +
                                                label + '_sequence_meta' +
                                                suffix + '.npy', allow_pickle=True).item()
        # load images
        sequences = list(self.sequences_meta.keys()
                         )  # (class transformerData gets sequences passed)
        self.load_images(sequences)

        if not self.rewrote_meta: self.rewrite_meta(sequences)

    def load_images(self, sequences):
        for sequence in tqdm(sequences):
            # if sequence's images already loaded then omitt
            if sequence in self.sequences_img: continue
            print('Loading images of sequence: {}'.format(sequence))

            # for a given sequence get list of unique image pair IDs
            listIDs = [[pair["imgID"] for pair in pairID]
                       for pairID in self.sequences_meta[sequence].values()]
            imgIDs = np.unique(np.array(listIDs).flatten())

            for imgID in imgIDs:
                # load the corresponding image and cast from [Height x Width] to [1 x Heigth x Width]
                img = cv.imread(
                    os.path.join(self.cfg.INPUT.COLMAP_IMAGES, sequence,
                                 'dense/images', imgID + ".jpg"),
                    cv.IMREAD_GRAYSCALE)
                img = np.expand_dims(img, 0)

                # mirror-pad the input image for crops surpassing the image borders
                # and to make all images square-shaped as well as uniformly sized
                if img.shape[1] > self.padTo or img.shape[2] > self.padTo:
                    raise RuntimeError(
                        "Image {} exceeds acceptable size, can't apply padding beyond image size."
                        .format(imgID))

                fillHeight = self.padTo - img.shape[1]
                fillWidth = self.padTo - img.shape[2]
                padLeft = int(np.round(fillWidth / 2))
                padRight = int(fillWidth - padLeft)
                padUp = int(np.round(fillHeight / 2))
                padDown = int(fillHeight - padUp)

                img = np.pad(
                    img,
                    pad_width=((0, 0), (padUp, padDown), (padLeft, padRight)),
                    mode='reflect')  # use either "reflect" or "symmetric"

                # store image and number of padded pixels to the left and above of the coordinate origin
                if not sequence in self.sequences_img:
                    self.sequences_img[sequence] = {}
                if not imgID in self.sequences_img[sequence]:
                    self.sequences_img[sequence][imgID] = {}
                self.sequences_img[sequence][imgID]["img"] = img
                self.sequences_img[sequence][imgID]["padLeft"] = padLeft
                self.sequences_img[sequence][imgID]["padUp"] = padUp

    def rewrite_meta(self, sequences):
        for sequence in tqdm(sequences):
            print("Rewriting keypoint meta information for sequence {}".format(
                sequence))
            for imgPairID, pair_meta in self.sequences_meta[sequence].items(
            ):  # iterate through dictionary of image pair meta data

                # get the pair indices for both anchor and positive
                pair_indx_a = np.where(
                    np.sum(imgPairID.split('-') == pair_meta[0]["pairID"],
                           axis=1) == 2)[0][0]
                pair_indx_p = np.where(
                    np.sum(imgPairID.split('-') == pair_meta[1]["pairID"],
                           axis=1) == 2)[0][0]
                imgID_a, imgID_p = pair_meta[0]['imgID'], pair_meta[1]['imgID']

                # update locations, orientations and scales
                self.updateKpCoordinates(sequence, imgPairID,
                                         [imgID_a, imgID_p],
                                         [pair_indx_a, pair_indx_p])
                self.updateKpOrientations(sequence, imgPairID,
                                          [imgID_a, imgID_p],
                                          [pair_indx_a, pair_indx_p])
                self.updateKpScales(sequence, imgPairID, [imgID_a, imgID_p],
                                    [pair_indx_a, pair_indx_p])

        self.rewrote_meta = True  # remember not to rewrite meta information again in any subsequent epoch

    def updateKpCoordinates(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            # look up by how many pixels the image's origin is shifted due to padding
            padLeft = self.sequences_img[sequence][
                imgIDAnchorPositive[0]]["padLeft"]
            padUp = self.sequences_img[sequence][
                imgIDAnchorPositive[0]]["padUp"]

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["locations"]

            for kpIDX, kp_loc in enumerate(
                    keypoints):  # iterate over keypoints

                # correct keypoint locations for padding and map from subpixel space to [-1,+1]x[-1,+1] grid,
                # notice that axes are transposed: keypoints are (x,y) but images are [Y x X]
                normKp = 2 * np.array([[(kp_loc[0] + padLeft) / self.padTo,
                                        (kp_loc[1] + padUp) / self.padTo]]) - 1

                # update (x,y) keypoint locations
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["locations"][kpIDX] = normKp

                # also keep track from which image pair ID this keypoint is from
                if not "imgPairID" in self.sequences_meta[sequence][imgPairID][
                        idxAnchorPositive]["keypoints"][pairIDX]:
                    self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                        "keypoints"][pairIDX]["imgPairID"] = len(keypoints) * [
                            None
                        ]  # creates a list of empty entries, this is horrible ...
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["imgPairID"][kpIDX] = imgPairID

    def updateKpOrientations(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["orientations"]
            for kpIDX, kp_orient in enumerate(
                    keypoints):  # iterate over keypoints

                # correct for different orientations of patches
                if self.cfg.TEST.ENABLE_ORIENTATION_FILTERING:
                    # check whether image is first in pair, we take this image's SIFT orientations
                    firstImgID = imgPairID.split('-')[0]
                    isFirstImg = imgIDAnchorPositive[0] == firstImgID

                    # rotate patches of both images by the first image's SIFT orientations
                    if isFirstImg:
                        orient = np.deg2rad(kp_orient)
                    else:  # if current image comes second then retrieve first image's orientations
                        firstPairIDX = pair_indx[1 - idxAnchorPositive]
                        # we may have in-place updated the first keypoint's orientations before, so need to look up untouched orientations
                        orient = self.sequences_meta_untouched[sequence][
                            imgPairID][1 - idxAnchorPositive]["keypoints"][
                                firstPairIDX]["orientations"][kpIDX]
                        orient = np.deg2rad(orient)

                    # if patch comes from the first image then additionally correct for planar rotation difference (in radians)
                    orientsDiffTrue = self.sequences_meta[sequence][imgPairID][
                        idxAnchorPositive]["keypoints"][pairIDX][
                            'orientsDiffTrue'][kpIDX]
                    orient += 0 if not isFirstImg else orientsDiffTrue
                else:
                    # rotate patches of first image by the first image's SIFT orientations
                    # and patches of second image by the second image's SIFT orientations
                    orient = np.deg2rad(kp_orient)

                # update keypoint orientations
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["orientations"][kpIDX] = orient

    # function to update keypoint scales:
    # multiply by SIFT scaling constant and divide by image width
    # --> both operations are now directly done within the transformer module, so this function is now redundant
    def updateKpScales(self, sequence, imgPairID, imgID, pair_indx):
        for idxAnchorPositive, imgIDAnchorPositive in enumerate(
                imgID):  # iterate over anchor and positive keypoint

            pairIDX = pair_indx[idxAnchorPositive]
            keypoints = self.sequences_meta[sequence][imgPairID][
                idxAnchorPositive]["keypoints"][pairIDX]["scales"]

            for kpIDX, kp_scale in enumerate(
                    keypoints):  # iterate over keypoints

                # get scale of SIFT keypoints (and correct for width of image plus padding)
                scale = kp_scale  # / self.padTo

                # update keypoint scales
                self.sequences_meta[sequence][imgPairID][idxAnchorPositive][
                    "keypoints"][pairIDX]["scales"][kpIDX] = scale

    # given a sequence and an image pair ID, look up the individual image's pair indices
    def get_pair_idx(self, seq_name, img_pair_id, img1_idx, img2_idx):
        # get meta for selected image pair
        pair_meta = self.sequences_meta[seq_name][img_pair_id]

        # get indices in meta for selected image pair,
        # img1_idx==0 always, but img2_idx may be randomly assigned either 0 or 1
        pair_idx_a = np.where(
            np.sum(img_pair_id.split('-') == pair_meta[int(img1_idx)]
                   ["pairID"],
                   axis=1) == 2)[0][0]
        pair_idx_p_or_n = np.where(
            np.sum(img_pair_id.split('-') == pair_meta[int(img2_idx)]
                   ["pairID"],
                   axis=1) == 2)[0][0]
        return pair_idx_a, pair_idx_p_or_n

    def __getitem__(self, idx):
        # note: imgPairID1==imgPairID2, img1_idx==0, img2_idx==1 for positive and img2_idx==random_img for negative,
        #       match==1 for positive and match==0 for negative, pair_indx_a, pair_indx_p, pair_indx_r,
        #       scale_correct and orient_correct contain the correction terms for scale and orientation projection
        seqID, imgPairID1, img1_idx, patch_a_idx, imgPairID2, img2_idx, patch_p_or_n_idx, match, pair_idx_a, pair_indx_p, \
        pair_idx_p_or_n, scale_correct, orient_correct = self.list_of_indices[idx]

        match_indices = [int(pair_idx_p_or_n), int(pair_indx_p)]

        # get images of anchor and positive or negative keypoint
        imgID_a = self.sequences_meta[seqID][imgPairID1][int(
            img1_idx)]["imgID"][0]
        imgID_p_or_n = self.sequences_meta[seqID][imgPairID2][int(
            img2_idx)]["imgID"][0]
        img_a, img_p_or_n = self.sequences_img[seqID][imgID_a], \
                            self.sequences_img[seqID][imgID_p_or_n]

        # get keypoint pair indices (passing imgPairID1, assuming that imgPairID1==imgPairID2 but img2_idx may not always be 1)

        # get meta information of anchor and positive keypoint
        meta_a = [
            self.sequences_meta[seqID][imgPairID1][int(img1_idx)]["keypoints"][
                int(pair_idx_a)]["locations"][int(patch_a_idx), :],
            self.sequences_meta[seqID][imgPairID1][int(img1_idx)]["keypoints"][
                int(pair_idx_a)]["scales"][int(patch_a_idx)],
            self.sequences_meta[seqID][imgPairID1][int(img1_idx)]["keypoints"][
                int(pair_idx_a)]["orientations"][int(patch_a_idx)],
            self.sequences_meta[seqID][imgPairID1][int(img1_idx)]["keypoints"][
                int(pair_idx_a)]["imgPairID"][int(patch_a_idx)]
        ]
        meta_p_or_n = [
            self.sequences_meta[seqID][imgPairID2][int(img2_idx)]["keypoints"]
            [match_indices[int(match)]]["locations"][int(patch_p_or_n_idx), :],
            self.sequences_meta[seqID][imgPairID2][int(img2_idx)]["keypoints"][
                match_indices[int(match)]]["scales"][int(patch_p_or_n_idx)],
            self.sequences_meta[seqID][imgPairID2][int(img2_idx)]["keypoints"]
            [match_indices[int(match)]]["orientations"][int(patch_p_or_n_idx)],
            self.sequences_meta[seqID][imgPairID2][int(img2_idx)]["keypoints"][
                match_indices[int(match)]]["imgPairID"][int(patch_p_or_n_idx)]
        ]

        # get untouched meta information of anchor and positive keypoint,
        # this may be needed e.g. when requiring the original pixel-space keypoint coordinates (as for 3D COLMAP reconstruction)
        meta_a_untouched = [
            self.sequences_meta_untouched[seqID][imgPairID1][int(img1_idx)]
            ["keypoints"][int(pair_idx_a)]["locations"][int(patch_a_idx), :],
            self.sequences_meta_untouched[seqID][imgPairID1][int(img1_idx)]
            ["keypoints"][int(pair_idx_a)]["scales"][int(patch_a_idx)],
            self.sequences_meta_untouched[seqID][imgPairID1][int(img1_idx)]
            ["keypoints"][int(pair_idx_a)]["orientations"][int(patch_a_idx)]
        ]
        meta_p_or_n_untouched = [
            self.sequences_meta_untouched[seqID][imgPairID2][int(
                img2_idx)]["keypoints"][match_indices[int(
                    match)]]["locations"][int(patch_p_or_n_idx), :],
            self.sequences_meta_untouched[seqID][imgPairID2][int(
                img2_idx)]["keypoints"][match_indices[int(match)]]["scales"][
                    int(patch_p_or_n_idx)],
            self.sequences_meta_untouched[seqID][imgPairID2][int(
                img2_idx)]["keypoints"][match_indices[int(
                    match)]]["orientations"][int(patch_p_or_n_idx)]
        ]

        return img_a, img_p_or_n, meta_a, meta_p_or_n, imgID_a, imgID_p_or_n, meta_a_untouched, meta_p_or_n_untouched, \
               float(img2_idx), \
               float(scale_correct), float(orient_correct), \
               int(match)

    def __len__(self):
        return len(self.list_of_indices)


# class to handle the external Brown data set
class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self,
                 cfg,
                 train=True,
                 transform=None,
                 batch_size=None,
                 load_random_triplets=False,
                 *arg,
                 **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.cfg = cfg
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = cfg.TRAINING.N_TRIPLETS
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.batch_size,
                                                   self.labels,
                                                   self.n_triplets)

    @staticmethod
    def generate_triplets(batch_size, labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for _ in tqdm(range(num_triplets)):
            if len(already_idxs) >= batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append(
                [indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


class Augmentor(object):
    """Class to augment data by randomly jittering
       the anchor keypoints' coordinates, orientations and scales at training time
    """
    def __init__(
            self,
            cfg,
            device,
    ):
        self.cfg = cfg
        self.padTo = cfg.TRAINING.PAD_TO
        self.device = device

        # location augmentation l ~ min(G(0.5), 5) in pixels, which is a bit more skewed than a binomial distribution
        #self.geometr = torch.distributions.geometric.Geometric(probs=0.5)
        self.binom = torch.distributions.binomial.Binomial(total_count=3,
                                                           probs=0.1)
        # orientation augmentation o ~ N(0,25/2) in degrees, such that 95% of samples fall within \pm 25
        self.normal = torch.distributions.normal.Normal(loc=0, scale=25.0)
        # scale ratio augmentation r ~ Gamma(shape=0.5,rate=1/scale=1.0), or skew it even more to the left ...
        self.gamma = torch.distributions.gamma.Gamma(1.5, 3.5)

        self.maxHeight = self.padTo  # standardized size of the square images
        self.pi = torch.Tensor([np.pi]).to(self.device)

    def augmentLoc(self, kpLoc):
        # perform augmentation on anchor keypoint coordinates
        batchSize = kpLoc.shape[0]
        #augmLoc = torch.min(self.binom.sample((batchSize, 2)).squeeze(), torch.tensor(5.0)).to(self.device)
        augmLoc = self.binom.sample((batchSize, 2)).to(self.device)
        kpLoc = self.maxHeight * (
            kpLoc + 1
        ) / 2  # invert mapping from pixel space to standard grid space (or sample directly in standard space)
        kpLoc += augmLoc  # augment in pixel-space
        kpLoc = 2 * kpLoc / self.maxHeight - 1  # re-map to standardized grid space [-1, +1]
        return kpLoc

    def augmentRot(self, rotation):
        # perform augmentation on anchor keypoint orientations
        batchSize = rotation.shape[0]
        augmRot = self.normal.sample((batchSize, 1)).squeeze().to(self.device)
        augmRot = self.deg2rad(
            augmRot)  # map to radians (or sample directly in radian space)
        rotation += augmRot
        return rotation % (2 * np.pi)

    def augmentScale(self, scaling_a, scaling_p):
        # perform augmentation on keypoint scale ratios
        batchSize = scaling_a.shape[0]
        # elementwise comparison to check which scale dominates
        maxAnchor = (scaling_a >= scaling_p).float()
        augmRatio = self.gamma.sample((batchSize, 1)).squeeze().to(self.device)

        # augment the ratio, this currently only increases the ratio (additively)
        scaling_a += maxAnchor * scaling_p * augmRatio
        scaling_p += (1 - maxAnchor) * scaling_a * augmRatio
        return scaling_a, scaling_p

    def deg2rad(self, deg):
        rad = deg * self.pi / 180.0
        return rad
