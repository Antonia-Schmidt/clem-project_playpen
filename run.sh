#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-8B-Instruct' --model_adapter ''

CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Llama-3.1-70B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-70B-Instruct' --model_adapter ''
#python3 ./train_model_qlora.py --model_name "Qwen/Qwen2.5-Coder-32B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_qwen-coder.csv' --model_adapter ''
