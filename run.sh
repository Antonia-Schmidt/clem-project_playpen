#!/bin/bash
python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/D70001.csv' --model_adapter ''
python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/D70002.csv' --model_adapter ''
python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/D70003.csv' --model_adapter ''
python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/D70005.csv' --model_adapter ''