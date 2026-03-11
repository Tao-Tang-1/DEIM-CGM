#CUDA_VISIBLE_DEVICES=0 torchrun  train.py -c configs/deimv2/deimv2_dinov3_s_wheat.yml --use-amp --seed=0
CUDA_VISIBLE_DEVICES=0 torchrun  train.py -c configs/deimv2/ablation_experiments/deimv2_dinov3_s_wheat_ABC.yml --use-amp --seed=0
#CUDA_VISIBLE_DEVICES=0 torchrun  train.py -c configs/deimv2/wheat/deimv2_dinov3_s_wheat_resize2048.yml --use-amp --seed=0