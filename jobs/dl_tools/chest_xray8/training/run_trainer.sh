#!/bin/bash

python jobs/dl_tools/chest_xray8/training/task.py \
--tag uf_v0 \
--data_set_dir NoF_Eff_Inf_Mas \
--n_classes 4 \
--learning_rate 0.001 \
--batch_size 32 \
--epochs 1