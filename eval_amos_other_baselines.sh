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

cp baselines/geodesc/image_matching_utils.py third_party/geodesc/examples/image_matching_utils.py

# config_file is needed to extract patches

python baselines/matching/amos/amos_other_baselines.py --method=SIFT --config_file=configs/init_one_example_stn_16.yml
python baselines/matching/amos/amos_other_baselines.py --method=GeoDesc --config_file=configs/init_one_example_stn_16.yml

# to run the evaluation on the old dataset download it from:
# https://drive.google.com/file/d/10rOtO3E8kXrgcksvlTtFFeRVphZRedmU/view?usp=sharing
# download keypoints and patches from here - wget http://cmp.felk.cvut.cz/~qqpultar/train_patches.pt
# and copy folder Train/ and train_patches.pt to dl/AMOS/Handpicked_v3_png/ then run:


# python baselines/matching/amos/amos_other_baselines.py --method=SIFT --config_file=configs/init_one_example_stn_16.yml --amos_dataset=dl/AMOS/Handpicked_v3_png/
# python baselines/matching/amos/amos_other_baselines.py --method=GeoDesc --config_file=configs/init_one_example_stn_16.yml --amos_dataset=dl/AMOS/Handpicked_v3_png/