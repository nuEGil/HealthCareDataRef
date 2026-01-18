#!/bin/bash
python jobs/dl_tools/chest_xray8/training/task.py \
--tag blockstack_v0 \
--data_set_dir NoF_Eff_Inf_Mas \
--n_classes 4 \
--learning_rate 0.001 \
--batch_size 16 \
--epochs 100

# python jobs/dl_tools/chest_xray8/training/task.py \
# --tag blockstack_v1 \
# --data_set_dir NoF_Eff_Inf_Mas \
# --n_classes 4 \
# --learning_rate 0.01 \
# --batch_size 32 \
# --epochs 10

# python jobs/dl_tools/chest_xray8/training/task.py \
# --tag blockstack_v2 \
# --data_set_dir NoF_Eff_Inf_Mas \
# --n_classes 4 \
# --learning_rate 0.1 \
# --batch_size 32 \
# --epochs 10