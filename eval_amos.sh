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

(   if [ ! -d "dl/AMOS" ]; then
        mkdir -p dl/AMOS/Handpicked_v3_png
        mkdir -p dl/AMOS/AMOS_views_v3
        wget "http://cmp.felk.cvut.cz/~qqpultar/AMOS_views_v3.zip" -q --show-progress -O dl/AMOS/AMOS_views_v3.zip
        wget "https://drive.switch.ch/index.php/s/uzoE2i4uaotQUYt/download?path=%2F&files=Handpicked_v3_png.tar.gz" -q --show-progress -O dl/AMOS/Handpicked_v3_png/Handpicked_v3_png.tar.gz
        wget "http://cmp.felk.cvut.cz/~qqpultar/train_patches.pt" -q --show-progress -O dl/AMOS/AMOS_views_v3/train_patches.pt
        wget "https://drive.switch.ch/index.php/s/yZ6q1TRvpx2SNfV/download" -q --show-progress -O dl/AMOS/AMOS_views_v3/images_in_views.npy
        wget "https://drive.switch.ch/index.php/s/4Jb1rKmK0iVxIZ7/download" -q --show-progress -O dl/AMOS/Handpicked_v3_png/images_in_views.npy
        tar xC dl/AMOS/Handpicked_v3_png/ -f dl/AMOS/Handpicked_v3_png/Handpicked_v3_png.tar.gz
        unzip dl/AMOS/AMOS_views_v3.zip -d dl/AMOS/
        cp dl/AMOS/AMOS_views_v3/train_patches.pt dl/AMOS/Handpicked_v3_png/

    fi
)

python baselines/matching/amos/amos.py --config_file=configs/init_one_example_ptn_96.yml
python baselines/matching/amos/amos.py --config_file=configs/init_one_example_stn_16.yml

# evaluation on old dataset
#python baselines/matching/amos/amos.py --config_file=configs/init_one_example_ptn_96.yml --amos_dataset=dl/AMOS/Handpicked_v3_png/
#python baselines/matching/amos/amos.py --config_file=configs/init_one_example_stn_16.yml --amos_dataset=dl/AMOS/Handpicked_v3_png/
