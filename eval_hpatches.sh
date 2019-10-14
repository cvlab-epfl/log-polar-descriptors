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

# Automatically downloading and extracting HPatches sequences dataset, SIFT keypoints and matches for HPatches dataset

(   if [ ! -d "dl/HPatches" ]; then
        mkdir dl/HPatches
        wget "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz" -q --show-progress -O dl/HPatches/hpatches-sequences-release.tar.gz
        wget "https://drive.switch.ch/index.php/s/uzoE2i4uaotQUYt/download?path=%2F&files=hpatches-SIFT-keypoints.tar.gz" -q --show-progress -O dl/HPatches/hpatches-SIFT-keypoints.tar.gz
        tar xC dl/HPatches/ -f dl/HPatches/hpatches-SIFT-keypoints.tar.gz
        tar xC dl/HPatches/ -f dl/HPatches/hpatches-sequences-release.tar.gz
    fi
)

python baselines/matching/hpatches/hpatches_sequences_ranking.py --sequence_type='v_' --config_file=configs/init_one_example_ptn_96.yml --hpatches_keypoints=dl/HPatches/hpatches-SIFT-keypoints
python baselines/matching/hpatches/hpatches_sequences_ranking.py --sequence_type='i_' --config_file=configs/init_one_example_ptn_96.yml --hpatches_keypoints=dl/HPatches/hpatches-SIFT-keypoints

python baselines/matching/hpatches/hpatches_sequences_ranking.py --sequence_type='v_' --config_file=configs/init_one_example_stn_16.yml --hpatches_keypoints=dl/HPatches/hpatches-SIFT-keypoints
python baselines/matching/hpatches/hpatches_sequences_ranking.py --sequence_type='i_' --config_file=configs/init_one_example_stn_16.yml --hpatches_keypoints=dl/HPatches/hpatches-SIFT-keypoints
