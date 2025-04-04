"""
This file contains the Object-Oriented Version of the Model Fine-tuning process.
All Configs can be loaded with default values. If needed those values can be changed.
to run this file use  python .\train_model.py from the Semantic-Parsing-Research folder
"""
from src.model_wrapper.model import CustomTextToSqlModel


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

# subprocess.Popen('huggingface-cli login --token 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 --add-to-git-credential ', shell=True)

chat_template_mapping: dict = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3",
    "Nicohst/my_test_model": "llama-3",
    "meta-llama/Llama-3.1-70B": "llama-3",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit": "llama-3",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit": "llama-3",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "chatml",
    'meta-llama/Llama-3.1-70B-Instruct':  "llama-3",
    "unsloth/Qwen2.5-Coder-32B-Instruct": "chatml",
    "mistralai/Mistral-Small-24B-Instruct-2501": "chatml",
    "unsloth/Mistral-Small-Instruct-2409": "mistral",
    "unsloth/Mistral-Small-24B-Instruct-2501": "mistral"
}


def check_chat_template_mapping(model_name: str):
    try:
        #mapping: str = chat_template_mapping[model_name]
        #logging.info(f"found chat template {mapping} for model {args.model_name}")
        return "llama-3"

    except KeyError:
        print(f"For the model {model_name}, no suitable chat template was found returning default llama-3.")
        return "llama-3"


def get_model_hub_id(
    base_model_name: str, learning_strategy: str, episodes: int, dataset_name, training_steps = 0
) -> str:
    hub_id = f'clembench-playpen/{base_model_name}_playpen_{learning_strategy}-e3_{dataset_name.split("/")[-1].split("_")[0].replace(".csv", "")}'
    if training_steps is not None:
        print(training_steps)
        hub_id = hub_id + f'_{training_steps/1000}K-steps'
    return hub_id


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
        "--test_dataset", help="The path to test dataset", default=None
    )
    parser.add_argument(
        "--hf_model_name", help="The desired name of the model in huggingface used in the huggingface id evenutally", default=None
    )

    parser.add_argument(
        "--steps", help="The desired name of the model in huggingface used in the huggingface id evenutally", default=None
    )
    parser.add_argument(
        "--model_adapter", help="The path to training dataset", default=None
    )

    # get all the args
    args = parser.parse_args()
    train: bool = args.training_dataset is not None
    max_steps = int(args.steps)
    if max_steps == 0:
        max_steps = None

    # check that template mapping exists
    check_chat_template_mapping(args.model_name)

    # SFT Trainer parameters
    max_seq_length = (
        1024  # Maximum sequence length to use can be adapted depending on the input
    )
    packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
    device_map = {"": 1}  # Load the entire model on the GPU 0

    # load custom configurations. If values need to be changed pass them as arguments
    lora_config: CustomLoraConfiguration = CustomLoraConfiguration()
    bnb_config: CustomBitsAndBitesConfiguration = CustomBitsAndBitesConfiguration(
        use_4bit=True
    )
    training_arguments: CustomTrainingArguments = CustomTrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=0, # 1
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        hub_model_id=None,
        max_steps=max_steps,

    )

    inference_config: CustomInferenceConfig = CustomInferenceConfig(
        do_sample=False,
        max_new_tokens=200,
    )

    # prepare the model hub ID according to the specified format.
    model_hub_id: str = get_model_hub_id(
        base_model_name=args.hf_model_name,
        episodes=training_arguments.num_train_epochs,
        learning_strategy="SFT",
        dataset_name=args.training_dataset,
        training_steps=max_steps
    )

    training_arguments.hub_model_id = model_hub_id
    print(training_arguments.hub_model_id)
    print(chat_template_mapping[args.model_name])

    # Initialize model
    model: CustomTextToSqlModel = CustomTextToSqlModel(
        model_name=args.model_name,
        path_dataset_train=args.training_dataset,
        path_dataset_test=args.test_dataset,
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

    # safe the configuration files
    model.save_config()

    if train:
        if max_steps == 600:
            if args.model_name.startswith("mistral"):
                print("Train with warm up Mistral")
                model.train_model_with_wramup('./data/training_data/warm-up_400samples_mistral-small.csv')
            else:
                print("Train with warm up LLama")
                model.train_model_with_wramup('./data/training_data/warm-up_400samples.csv')
        else:
            print("Train with multistep-collator")
            #model.train_model()
            # model.train_model_with_collator()
            #model.train_model_with_wramup('./data/training_data/warm-up_400samples.csv')
            model.train_model_with_multi_step_collator()

        # save the model
        model.save_model()