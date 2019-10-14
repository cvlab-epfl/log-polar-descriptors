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

(   if [ ! -d "dl/HPatches" ]; then
        wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -q --show-progress
        mkdir dl/HPatches/
        mv hpatches-sequences-release.tar.gz dl/HPatches/hpatches-sequences-release.tar.gz
        tar xC dl/HPatches/ -f dl/HPatches/hpatches-sequences-release.tar.gz
    fi
)

# Download pre-extracted SIFT keypoints for HPatches dataset from here:
# https://drive.google.com/file/d/1euJeKLEnGeemfM_mgguvr5HMZwwpuK0v/view?usp=sharing
# and copy to dl/HPatches/

python baselines/matching/hpatches/hpatches_sequences_ranking_other_baselines.py --sequence_type='v_' --scale=12 --method=GeoDesc --config_file=configs/init_one_example_stn_16.yml
python baselines/matching/hpatches/hpatches_sequences_ranking_other_baselines.py --sequence_type='i_' --scale=12 --method=GeoDesc --config_file=configs/init_one_example_stn_16.yml

python baselines/matching/hpatches/hpatches_sequences_ranking_other_baselines.py --sequence_type='v_' --scale=12 --method=SIFT --config_file=configs/init_one_example_stn_16.yml
python baselines/matching/hpatches/hpatches_sequences_ranking_other_baselines.py --sequence_type='i_' --scale=12 --method=SIFT --config_file=configs/init_one_example_stn_16.yml

