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

from yacs.config import CfgNode as CN

_C = CN()

_C.INPUT = CN()

_C.INPUT.SOURCE = '/cvlabdata1/cvlab/datasets_eduard/colmap-patches-fix-dumped/pairs-2000-kps-2000-size-32-thscale-1.25-thori-22.50/'

_C.INPUT.TRAINING_PAIRS = '/cvlabsrc1/cvlab/datasets_anastasiia/descriptors/colmap_dataset'

# path
_C.INPUT.BROWN_DATASET = 'data/sets/'
_C.INPUT.COLMAP_IMAGES = 'data/sets/'
_C.INPUT.VAL_SPLIT = 'data/sets/'


_C.INPUT.IMAGE_SIZE = 32

_C.LOGGING = CN()


_C.LOGGING.ENABLE_LOGGING = True

_C.LOGGING.LOG_DIR = 'data/logs/'
_C.LOGGING.MODEL_DIR = 'data/models/'
_C.LOGGING.IMGS_DIR = 'data/images/'

_C.LOGGING.LOG_INTERVAL = 10

_C.TRAINING = CN()

_C.TRAINING.MODEL_DIR = 'data/models/'

_C.TRAINING.EXPERIMENT_NAME = 'liberty_train'
_C.TRAINING.LABEL = '_x64'
_C.TRAINING.TRANSFORMER = 'STN'
_C.TRAINING.SCALE = 128
_C.TRAINING.PAD_TO = 1500

_C.TRAINING.COORDS = 'log'
_C.TRAINING.IMAGE_SIZE = 32

_C.TRAINING.IS_DESC_256 = False
_C.TRAINING.SOFT_AUG = True
_C.TRAINING.ORIENT_CORRECTION = True
_C.TRAINING.ENABLE_SCALE_FILTERING = False
_C.TRAINING.MAX_RATIO = 2

_C.TRAINING.ENABLE_ORIENTATION_FILTERING = True
_C.TRAINING.ORIENTATION_FILTER_VALUE = 25

_C.TRAINING.TRAINING_SET = 'colmap_dataset'

_C.TRAINING.LOSS = 'triplet_margin'

_C.TRAINING.BATCH_REDUCE = 'min'

_C.TRAINING.NUM_WORKERS = 8

_C.TRAINING.PIN_MEMORY = True

_C.TRAINING.RESUME = ''

_C.TRAINING.START_EPOCH = 0

_C.TRAINING.EPOCHS = 10

_C.TRAINING.BATCH_SIZE = 1024

_C.TRAINING.TEST_BATCH_SIZE = 1024

_C.TRAINING.N_TRIPLETS = 5000000

_C.TRAINING.MIN_KEYPOINTS_PER_IMAGE = 250
_C.TRAINING.MIN_KEYPOINTS_PER_IMAGE_PAIR = 100

_C.TRAINING.MIN_THRESHOLD_DISTANCE = 7.0

_C.TRAINING.MARGIN = 1.0

_C.TRAINING.ANCHOR_SWAP = True

_C.TRAINING.LR = 10

_C.TRAINING.LR_DECAY = 1e-6

_C.TRAINING.W_DECAY = 1e-4

_C.TRAINING.OPTIMIZER = 'adam'

# frequency for cyclic learning rate
_C.TRAINING.FREQ = 10

_C.TRAINING.FLIPROT = False

_C.TRAINING.AUG = False

# enables CUDA training
_C.TRAINING.NO_CUDA = False

# ID number of GPU
_C.TRAINING.GPU_ID = 3

_C.TRAINING.SEED = 42

_C.TEST = CN()

_C.TEST.MODEL_WEIGHTS = ''
_C.TEST.TEST_BATCH_SIZE = 400
_C.TEST.EVAL_INTERVAL = 50
_C.TEST.ENABLE_ORIENTATION_FILTERING = False

_C.TEST.SCALE = 96
_C.TEST.PAD_TO = 1500
_C.TEST.COORDS = 'log'
_C.TEST.IMAGE_SIZE = 32
_C.TEST.IS_DESC_256 = False
_C.TEST.ORIENT_CORRECTION = False
_C.TEST.TRANSFORMER = "PTN"
