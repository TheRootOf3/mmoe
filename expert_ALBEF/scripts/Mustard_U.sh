python -m torch.distributed.launch --nproc_per_node=1 --use_env Mustard.py \
--config ./configs/Mustard_U.yaml \
--output_dir ./output/Mustard-U \
--checkpoint ./output/Mustard/checkpoint_best.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env
