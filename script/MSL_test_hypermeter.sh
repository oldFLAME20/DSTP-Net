#!/bin/bash

# 搜索范围


# 公共参数
dataset="MSL"
data_path="./data/MSL/"
input_c=55
output_c=55
lambd=0.01
gamma=0.5
a1=1.0
a2=0.01
a3=0.01
beta=0.5
n_memory=10
n_memory_patch=10
device=cuda:2
topk=5
patch_len=7


      echo "=============================="
      echo ">>> Running with a2=$a2, a3=$a3,n_memory=$n_memory"
      echo "=============================="

      # 第一阶段：train
      /home/ming/anaconda3/envs/lgx-memto/bin/python main_gcn_patch_no_scale_integration.py \
        --anormly_ratio 1.0 \
        --num_epochs 100 \
        --batch_size 32 \
        --mode train \
        --dataset $dataset \
        --data_path $data_path \
        --input_c $input_c \
        --output_c $output_c \
        --n_memory $n_memory \
        --n_memory_patch $n_memory_patch \
        --lambd $lambd \
        --lr 1e-5 \
        --device $device  \
        --memory_initial False \
        --phase_type None \
        --gamma $gamma \
        --a1 $a1 \
        --a2 $a2 \
        --a3 $a3 \
        --beta $beta \
        --patch_len $patch_len \
        --topk $topk

      # 第二阶段：memory_initial
      /home/ming/anaconda3/envs/lgx-memto/bin/python main_gcn_patch_no_scale_integration.py \
        --anormly_ratio 1.0 \
        --num_epochs 100 \
        --batch_size 32 \
        --mode memory_initial \
        --dataset $dataset \
        --data_path $data_path \
        --input_c $input_c \
        --output_c $output_c \
        --device $device  \
        --n_memory $n_memory \
        --n_memory_patch $n_memory_patch \
        --lambd $lambd \
        --lr 5e-5 \
        --memory_initial True \
        --phase_type second_train \
        --gamma $gamma \
        --a1 $a1 \
        --a2 $a2 \
        --a3 $a3 \
        --beta $beta \
        --patch_len $patch_len \
        --topk $topk

#      beta_list=(1 2 5 10)
#      for beta in "${beta_list[@]}"; do

      # 第三阶段：test
      /home/ming/anaconda3/envs/lgx-memto/bin/python main_gcn_patch_no_scale_integration.py \
        --anormly_ratio 1.0 \
        --num_epochs 10 \
        --batch_size 32 \
        --mode test \
        --dataset $dataset \
        --data_path $data_path \
        --input_c $input_c \
        --output_c $output_c \
        --device $device  \
        --n_memory $n_memory \
        --n_memory_patch $n_memory_patch \
        --memory_initial False \
        --phase_type test \
        --gamma $gamma \
        --a1 $a1 \
        --a2 $a2 \
        --a3 $a3 \
        --beta $beta \
        --patch_len $patch_len \
        --topk $topk

      echo ">>> Finished (a2=$a2, a3=$a3 and n_memory=$n_memory)"
      echo ""

#  done
#done