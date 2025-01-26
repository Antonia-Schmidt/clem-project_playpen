"""
This file contains the Object-Oriented Version of the Model Fine-tuning process.
All Configs can be loaded with default values. If needed those values can be changed.
to run this file use  python .\train_model.py from the Semantic-Parsing-Research folder
"""

import argparse
import logging

# login to huggingface hub
import subprocess

import torch
from huggingface_hub import EvalResult, ModelCard, ModelCardData, whoami

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

from src.config.configurations import (
    CustomBitsAndBitesConfiguration,
    CustomInferenceConfig,
    CustomLoraConfiguration,
    CustomTrainingArguments,
    CustomUnslothModelConfig,
)
from src.model_wrapper.model import CustomTextToSqlModel

# subprocess.Popen('huggingface-cli login --token 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 --add-to-git-credential ', shell=True)

chat_template_mapping: dict = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3",
    "Nicohst/my_test_model": "llama-3",
    "meta-llama/Llama-3.1-70B": "llama-3",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit": "llama-3",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": "llama-3",
}


def check_chat_template_mapping(model_name: str):
    try:
        mapping: str = chat_template_mapping[model_name]
        logging.info(f"found chat template {mapping} for model {args.model_name}")

    except KeyError:
        print(f"For the model {model_name}, no suitable chat template was found.")


def get_model_hub_id(
    base_model_name: str, learning_strategy: str, experiment_name: str
) -> str:
    return f'clembench-playpen/{base_model_name.replace("/", "-")}_{learning_strategy}_{experiment_name}'


if __name__ == "__main__":
    # initilize argparser
    parser = argparse.ArgumentParser(
        description="Training script for QLORA training of huggingface models"
    )
    parser.add_argument("--model_name", help="The huggingface model id")
    parser.add_argument(
        "--output_dir",
        help="The Base directory where the output should be stored",
        default="./output",
    )
    parser.add_argument(
        "--training_dataset", help="The path to training dataset", default=None
    )
    parser.add_argument(
        "--model_adapter", help="The path to the adapter", default=None
    )

    # get all the args
    args = parser.parse_args()
    train: bool = args.training_dataset is not None

    # check that template mapping exists
    check_chat_template_mapping(args.model_name)

    # SFT Trainer parameters
    max_seq_length = (
        1024  # Maximum sequence length to use can be adapted depending on the input
    )
    packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
    device_map = {"": 0}  # Load the entire model on the GPU 0

    ##### PARAMTERS FOR SEARCH
    schedulers = ['linear', 'cosine']
    optimizers = ['adamw_8bit', 'sgd']
    learningRates = [2e-4, 2e-2, 2e-6]
    loraRandA = [(32, 64), (64, 128), (128, 256), (256, 512)]
    decays = [0.01, 0.001]


    run_number = 1
    for scheduler in schedulers:
        for optimizer in optimizers:
            for learning_rate in learningRates:
                for decay in decays:
                    for loraR, loraA in loraRandA:
                        print("RUN NUMBER: ", run_number)
                        if run_number <= 9:
                            experiment_name = f'D9000{run_number}'
                        elif run_number <= 99:
                            experiment_name = f'D900{run_number}'
                        else:
                            experiment_name = f'D90{run_number}'
                        
                        if run_number != 28:
                            print('skipped_run ', run_number)
                            run_number += 1
                            continue

                        # set lora confing
                        lora_config: CustomLoraConfiguration = CustomLoraConfiguration(
                            lora_r=loraR, lora_alpha=loraA
                        )
                        bnb_config: CustomBitsAndBitesConfiguration = CustomBitsAndBitesConfiguration(
                            use_4bit=True
                            )

                        training_arguments: CustomTrainingArguments = CustomTrainingArguments(
                            per_device_train_batch_size=4,
                            gradient_accumulation_steps=1,
                            num_train_epochs=1,
                            fp16=not torch.cuda.is_bf16_supported(),
                            bf16=torch.cuda.is_bf16_supported(),
                            optim=optimizer,
                            lr_scheduler_type=scheduler,
                            hub_model_id=None,
                            learning_rate=learning_rate,
                            weight_decay=decay
                        )
                        inference_config: CustomInferenceConfig = CustomInferenceConfig(
                            do_sample=False,
                            max_new_tokens=200,
                        )

                        # prepare the model hub ID according to the specified format.
                        model_hub_id: str = get_model_hub_id(
                            base_model_name=args.model_name,
                            learning_strategy="SFT",
                            experiment_name=experiment_name,
                        )

                        training_arguments.hub_model_id = model_hub_id
                        print(training_arguments.hub_model_id)

                        # Initialize model
                        model: CustomTextToSqlModel = CustomTextToSqlModel(
                            model_name=args.model_name,
                            path_dataset_train=args.training_dataset,
                            path_dataset_inference="UNUSED RN",  # since this is used as training script only
                            output_dir=args.output_dir,
                            lora_config=lora_config,
                            bnb_config=bnb_config,
                            unsloth_config=CustomUnslothModelConfig(max_seq_length=max_seq_length),
                            training_arguments=training_arguments,
                            inference_config=inference_config,
                            max_seq_length=max_seq_length,
                            packing=packing,
                            device_map=device_map,
                            train=train,
                            model_adapter=args.model_adapter,
                            chat_template=chat_template_mapping[args.model_name],
                        )

                        if train:
                            model.train_model()

                            # save the models
                            model.save_model()

                        # free the memory that the model used
                        del model

                        # increment with episodes amount since each model will be saved after each episode.
                        run_number += 1
