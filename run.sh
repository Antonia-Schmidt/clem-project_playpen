#!/bin/bash
python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_llama.csv' --model_adapter ''
