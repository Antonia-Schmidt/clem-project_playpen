#!/bin/bash
python3 ./train_model_qlora.py --model_name "unsloth/Meta-Llama-3.1-70B-bnb-4bit" --output_dir './output' --training_dataset './data/training_data/D30001.csv' --model_adapter ''

