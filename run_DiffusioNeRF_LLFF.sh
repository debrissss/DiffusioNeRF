 #!/bin/bash

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/runtime_env.sh"

###parameters for the scenes

##LLFF 训练迭代次数与后面的 few-shot 设置对应，终端实际迭代次数ITERS除以三
#（即不同 few-shot 下的迭代次数0表示全部视图，963表示使用的视图数）扩散正则化开始生效的迭代步数。

SCALE=(0.33 0.2 0.3 0.33 0.3 0.01 0.1 0.005)
BOUND=(4.5 3.5 3.5 3.0 3.0 3.0 2.5 3.0)
TEST=("trex" "horns" "fortress" "fern" "room" "flower" "orchids" "leaves")
ITERS=(15000 12000 10000 15000)
FEW_SHOT=(0 9 6 3)
DIFF_REG_START_ITER=(2000 1000 1000 1000)


### scenes
### 8 scenes (0, 1, ..., 7,...,)
### Choose the exact index to run only a specific scene (see the previous "TEST" variable)
for i_dataset in 3
do

### folders for dataset and output base on the chosen scenes and few-shot setting

##LLFF
DATASET="data/nerf_llff_data/"${TEST[i_dataset]}
WORK_BASE=("test_LLFF/test_"${TEST[i_dataset]} "test_LLFF/test_"${TEST[i_dataset]}"/few_shot9" "test_LLFF/test_"${TEST[i_dataset]}"/few_shot6" "test_LLFF/test_"${TEST[i_dataset]}"/few_shot3")
DATASET_NAME=(${TEST[i_dataset]} ${TEST[i_dataset]}" 9-views" ${TEST[i_dataset]}" 6-views" ${TEST[i_dataset]}" 3-views")



### few-shot setting
###  	0--> no few-shot; 1--> 9-views; 2--> 6-views; 3--> 3-views
### Choose a specific index to run only a specific few-shot setting
for i_shot in 3
do


### Run LLFF
###--diff_reg：启用扩散正则化（diffusion regularization）。--loss_dist：启用距离损失。--loss_fg：启用前景损失。--use_TVdepth：使用深度正则。
### training and test on the last checkpoint
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" main_nerf.py $DATASET --workspace ${WORK_BASE[i_shot]}/test_DiffusioNeRF --fp16 --few_shot ${FEW_SHOT[i_shot]} --scale ${SCALE[i_dataset]} --bound ${BOUND[i_dataset]} --dataset_name "${DATASET_NAME[i_shot]}" --implementation_name "DiffusioNeRF" --iters ${ITERS[i_shot]} --diff_reg --loss_dist --loss_fg --use_depth --diff_reg_start_iter ${DIFF_REG_START_ITER[i_shot]} --dist_lambda 2e-5 --fg_lambda 1e-4 --patch_size 4 --depth_reg_lambda 0.1 --use_nll_color --use_nll_sigma --fre_nll_color --smoothing --smoothing_lambda 0.00001 --smooth_sampling_method near_pixel


done

done




#Aug.neurtv
