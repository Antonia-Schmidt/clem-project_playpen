#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 1 --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 2 --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 3 --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --neg 6 --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --model_condition best_models --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
CUDA_VISIBLE_DEVICES=0 python3 DPO_training.py --model_condition same_family_model --hf_login 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 --wandb_login 29fc614754925fca7f38d6d2193b3f5afa8485a9 --cache_dir ./output/DPO_29_01_2025
