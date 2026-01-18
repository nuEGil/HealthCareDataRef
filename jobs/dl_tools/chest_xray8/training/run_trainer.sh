#!/bin/bash
python jobs/dl_tools/chest_xray8/training/task.py \
--model_type ResNet50_add_c_head \
--tag mass_v0 \
--data_set_dir NoF_Eff_Inf_Mas \
--n_classes 1 \
--learning_rate 0.001 \
--batch_size 32 \
--epochs 100 

# python jobs/dl_tools/chest_xray8/training/task.py \
# --model_type BlockStack \
# --tag mass_v0 \
# --data_set_dir NoF_Eff_Inf_Mas \
# --n_classes 1 \
# --learning_rate 0.001 \
# --batch_size 32 \
# --epochs 100 

# python jobs/dl_tools/chest_xray8/training/task.py \
# --tag uf_res50_ch_bin_v1 \
# --data_set_dir NoF_Eff_Inf_Mas \
# --n_classes 1 \
# --learning_rate 0.01 \
# --batch_size 32 \
# --epochs 100


# python jobs/dl_tools/chest_xray8/training/task.py \
# --tag uf_res50_ch_bin_v2 \
# --data_set_dir NoF_Eff_Inf_Mas \
# --n_classes 1 \
# --learning_rate 0.1 \
# --batch_size 32 \
# --epochs 100

