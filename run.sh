python /project/train/src_repo/split.py
cd /project/train/src_repo/STDC-Seg
export CUDA_VISIBLE_DEVICES=0
python  train.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode train \
--n_workers_train 4 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar \
--n_img_per_gpu 16  \
--save_iter_sep 1000 