#!/bin/bash

# Check if a model name is provided
if [ $# -eq 0 ]; then
    echo "Please provide a model name as an argument."
    echo "Usage: ./train.sh <model_name>"
    exit 1
fi

MODEL_NAME="$1"
OUTPUT_DIR="./output"
DATA_DIR="./data/training_data"

# Iterate over all datasets starting with "D"
for dataset in "$DATA_DIR"/D*.csv; do
    # Extract the dataset name without path and extension
    dataset_name=$(basename "$dataset" .csv)
    
    # Run the training script
    python3 ./train_model_qlora.py \
        --model_name "$MODEL_NAME" \
        --output_dir "${OUTPUT_DIR}/${dataset_name}" \
        --training_dataset "$dataset" \
        --model_adapter ''
    
    echo "Finished training on $dataset_name"
done

echo "All training runs completed."