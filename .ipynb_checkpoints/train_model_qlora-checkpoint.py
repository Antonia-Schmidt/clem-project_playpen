"""
This file contains the Object-Oriented Version of the Model Fine-tuning process.
All Configs can be loaded with default values. If needed those values can be changed.
to run this file use  python .\train_model.py from the Semantic-Parsing-Research folder
"""
# login to huggingface hub
import subprocess
import argparse
import torch
from huggingface_hub import ModelCard, ModelCardData, whoami, EvalResult

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from src.config.configurations import CustomLoraConfiguration, CustomTrainingArguments, CustomBitsAndBitesConfiguration, CustomInferenceConfig, CustomUnslothModelConfig
from src.model_wrapper.model import CustomTextToSqlModel

# subprocess.Popen('huggingface-cli login --token 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 --add-to-git-credential ', shell=True)

chat_template_mapping: dict = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct" :'llama-3',
    "Nicohst/my_test_model": 'llama-3'
}

def check_chat_template_mapping(model_name: str):
    try:
        mapping: str = chat_template_mapping[model_name]
        logging.info(f'found chat template {mapping} for model {args.model_name}')

    except KeyError:
        print(f'For the model {model_name}, no suitable chat template was found.')
        

if __name__ == "__main__":
    # initilize argparser
    parser = argparse.ArgumentParser(description="Training script for QLORA training of huggingface models")
    parser.add_argument("--model_name", help="The huggingface model id")
    parser.add_argument("--output_dir", help="The Base directory where the output should be stored", default='./output')
    parser.add_argument("--training_dataset", help="The path to training dataset", default=None)
    parser.add_argument("--model_adapter", help="The path to training dataset", default=None)


    # get all the args
    args = parser.parse_args()
    train: bool = args.training_dataset is not None

    # check that template mapping exists
    check_chat_template_mapping(args.model_name)

    # SFT Trainer parameters
    max_seq_length = 2048  # Maximum sequence length to use can be adapted depending on the input
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
        num_train_epochs=1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        hub_model_id='Llama-3-Clembench-Runs-Successful-Episodes'
    )
    inference_config: CustomInferenceConfig = CustomInferenceConfig(
        do_sample=False,
        max_new_tokens=200,
    )

    # Initialize model
    model: CustomTextToSqlModel = CustomTextToSqlModel(
        model_name=args.model_name,
        path_dataset_train=args.training_dataset,
        path_dataset_inference='UNUSED RN',  # since this is used as training script only
        output_dir=args.output_dir,
        lora_config=lora_config,
        bnb_config=bnb_config,
        unsloth_config = CustomUnslothModelConfig(max_seq_length=max_seq_length),
        training_arguments=training_arguments,
        inference_config=inference_config,
        max_seq_length=max_seq_length,
        packing=packing,
        device_map=device_map,
        train=train,
        model_adapter=args.model_adapter,
        chat_template=chat_template_mapping[args.model_name]
    )

    if train:
        # model.train_model()

        # model.trainer.push_to_hub()
        model.initialize_training()

        # save the model
        model.save_model()