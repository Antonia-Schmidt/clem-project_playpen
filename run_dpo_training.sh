#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 1 --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 2 --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 3 --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 6 --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --model_condition best_models --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --model_condition same_family_model --hf_login "" --wandb_login "" --cache_dir ./output/DPO_29_01_2025
