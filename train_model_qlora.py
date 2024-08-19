"""
This file contains the Object-Oriented Version of the Model Fine-tuning process.
All Configs can be loaded with default values. If needed those values can be changed.
to run this file use  python .\train_model.py from the Semantic-Parsing-Research folder
"""
# login to huggingface hub
import subprocess
import argparse

from src.config.configurations import CustomLoraConfiguration, CustomTrainingArguments, CustomBitsAndBitesConfiguration, CustomInferenceConfig
from src.model_wrapper.model import CustomTextToSqlModel

# subprocess.Popen('huggingface-cli login --token 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 --add-to-git-credential ', shell=True)

if __name__ == "__main__":
    # initilize argparser
    parser = argparse.ArgumentParser(description="Training script for QLORA training of huggingface models")
    parser.add_argument("--model_name", help="The huggingface model id")
    parser.add_argument("--output_dir", help="The Base directory where the output should be stored", default='./output')
    parser.add_argument("--path_to_data", help="The path to the dataset files")
    parser.add_argument("--training_dataset", help="The path to training dataset", default=None)
    parser.add_argument("--inference_dataset", help="The path to Inference dataset", default=None)
    parser.add_argument("--model_adapter", help="path to the adapter if available", default='')

    # get all the args
    args = parser.parse_args()
    train: bool = args.training_dataset is not None
    inference: bool = args.inference_dataset is not None

    # SFT Trainer parameters
    max_seq_length = 500  # Maximum sequence length to use can be adapted depending on the input
    packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
    device_map = {"": 0}  # Load the entire model on the GPU 0

    # load custom configurations. If values need to be changed pass them as arguments
    lora_config: CustomLoraConfiguration = CustomLoraConfiguration()
    bnb_config: CustomBitsAndBitesConfiguration = CustomBitsAndBitesConfiguration(
        use_4bit=True
    )
    training_arguments: CustomTrainingArguments = CustomTrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1
    )
    inference_config: CustomInferenceConfig = CustomInferenceConfig(
        do_sample=False,
        max_new_tokens=200,
    )

    # Initialize model
    model: CustomTextToSqlModel = CustomTextToSqlModel(
        model_name=args.model_name,
        path_dataset_train=args.training_dataset,
        path_dataset_inference=args.inference_dataset,
        model_adapter_name=args.model_adapter,
        output_dir=args.output_dir,
        lora_config=lora_config,
        bnb_config=bnb_config,
        training_arguments=training_arguments,
        inference_config=inference_config,
        max_seq_length=max_seq_length,
        packing=packing,
        device_map=device_map,
        run_name="code_llama_7B_fine_tuned_query_generation",
        train=train,
        inference=inference
    )

    # train the model
    if train:
        model.train_model()

        # save the model
        model.save_model()

    # Do Inference
    if inference:
        model.inference()