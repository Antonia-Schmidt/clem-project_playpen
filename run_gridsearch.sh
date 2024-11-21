#!/bin/bash
python3 ./train_model_qlora_gridsearch.py --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --output_dir './output' --training_dataset './data/training_data/D30003.csv' --model_adapter ''

