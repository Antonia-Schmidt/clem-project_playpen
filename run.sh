#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-8B-Instruct-multi-step-collator' --steps '700' --model_adapter ''
#CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/warm-up_900samples.csv' --test_dataset './data/training_data/warm-up_400samples.csv' --hf_model_name 'llama-3.1-8B-Instruct-only-warmup' --steps '225' --model_adapter ''


#CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Llama-3.1-70B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-70B-Instruct-0.1k-warmup' --steps '600' --model_adapter ''
#CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Llama-3.1-70B-Instruct" --output_dir './output' --training_dataset './data/training_data/DABL02_VTrain_rehearsal.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-70B-Instruct-rehearsal' --steps '820' --model_adapter ''

#CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "mistralai/Mistral-Small-24B-Instruct-2501" --output_dir './output' --training_dataset './data/training_data/DABL02_VTrain_rehearsal_mistral-small.csv' --test_dataset './data/training_data/DFINAL_VTest_mistral-small.csv' --hf_model_name 'Mistral-Small-24B-Instruct-rehearsal'  --steps '820'  --model_adapter ''
#CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "mistralai/Mistral-Small-24B-Instruct-2501" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain_mistral_small.csv' --test_dataset './data/training_data/DFINAL_VTest_mistral_small.csv' --hf_model_name 'Mistral-Small-24B-Instruc-0.1k-warmup'  --steps '600'  --model_adapter ''


# CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --output_dir './output' --training_dataset './data/training_data/DABL02_VTrain_rehearsal.csv' --test_dataset './data/training_data/DFINAL_VTest.csv' --hf_model_name 'llama-3.1-8B-Instruct-rehearsal-steps' --steps '820' --model_adapter ''
# CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "Qwen/Qwen2.5-Coder-32B-Instruct" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain_qwen-coder.csv' --test_dataset './data/training_data/DFINAL_VTest_qwen-coder.csv' --hf_model_name 'Qwen-coder-32B-Instruct' --model_adapter ''
# CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "mistralai/Mistral-Small-24B-Instruct-2501" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain_qwen-coder.csv' --test_dataset './data/training_data/DFINAL_VTest_qwen-coder.csv' --hf_model_name 'Mistral-Small-24B-Instruct' --model_adapter ''
# CUDA_VISIBLE_DEVICES=2 python3 ./train_model_qlora.py --model_name "unsloth/Mistral-Small-24B-Instruct-2501" --output_dir './output' --training_dataset './data/training_data/DFINAL_VTrain_mistral_small.csv' --test_dataset './data/training_data/DFINAL_VTest_mistral_small.csv' --hf_model_name 'Mistral-Small-24B-Instruct-2501' --steps '600' --model_adapter ''


