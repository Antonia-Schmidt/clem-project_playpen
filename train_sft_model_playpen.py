"""
This file contains the Object-Oriented Version of the Model Fine-tuning process.
All Configs can be loaded with default values. If needed those values can be changed.
to run this file use  python .\train_model.py from the Semantic-Parsing-Research folder
"""
from datetime import datetime

from src.model_wrapper.model import CustomTextToSqlModel

import argparse
import logging
import torch

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# All necessary configuration files
from src.config.configurations import (
    CustomBitsAndBitesConfiguration,
    CustomInferenceConfig,
    CustomLoraConfiguration,
    CustomTrainingArguments,
    CustomUnslothModelConfig,
)

chat_template_mapping: dict = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3",
    "Nicohst/my_test_model": "llama-3",
    "meta-llama/Llama-3.1-70B": "llama-3",
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


def prepare_model_hub_id(base_model_name: str, learning_strategy: str, dataset_name: str) -> str:
    now = datetime.now()
    ts = now.strftime("%Y-%m-%dT%H-%M-%S")
    hub_id = f'clembench-playpen/{ts}-{base_model_name}_playpen_{learning_strategy}-{dataset_name.split("/")[-1].split("_")[0].replace(".csv", "")}'
    return hub_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for QLORA training of huggingface models")
    parser.add_argument("--model_name", help="The huggingface model id of the model to be fine-tuned. E.g.: clembench-playpen/Mistral-Small-24B-Instruct")
    parser.add_argument("--output_dir",help="The Base directory where the output should be stored. Defaults to ./output",default="./output",)
    parser.add_argument("--training_dataset", help="The path to training dataset", default=None)
    parser.add_argument("--dev_dataset", help="The path to development dataset", default=None, )
    parser.add_argument("--warm_up_dataset", help="The path to warm-up dataset only needed when training is set to warm-up", default=None, )

    parser.add_argument("--hf_model_name", help="The desired name of the model in huggingface used in the huggingface id evenutally", default=None)

    # Options for adjusting the training parameters
    parser.add_argument("--steps", help="The desired number of steps for the training process", default=0)
    parser.add_argument("--epochs", help="The number of epochs to train for", default=1,)
    parser.add_argument("--batch_size", help="The batch size to use for training", default=4,)
    parser.add_argument("--learning_rate", help="The learning rate to use for training", default=0.001,)
    parser.add_argument("--weight_decay", help="The weight decay to use for training", default=0.0,)
    parser.add_argument("--warmup_steps", help="The number of warmup steps to use for training", default=5,)
    parser.add_argument("--optimizer", help="The optimizer used in training", default="adamw_8bit",)
    parser.add_argument("--max-seq-len", help="The maximum sequence length in trianing", default=1024,)

    # lora options
    parser.add_argument("--lora-r", help="The lora r", default=64,)
    parser.add_argument("--lora-a", help="Lora alpha", default=32,)

    # quantization parameter
    parser.add_argument("--load_in_4bit", help="Load in for bit otherwise in full precision", default=True,)

    # training style
    parser.add_argument("--training-style", help="The way the model is trained; Options: cold_start, warm_up, rehearsal, completion_only, multi_step_completion", default="cold_start",)


    # options for rehearsal, warmup, training on completions

    # get all the args
    args = parser.parse_args()
    train: bool = args.training_dataset is not None

    # set training steps and epoch parameters
    max_steps = int(args.steps) if int(args.steps) is not None or int(args.steps) == 0 else None
    epochs = int(args.epochs) if int(args.steps) is None else 0

    logging.info(f"Setting number of training-steps to: {args.steps} and epochs to: {args.epochs}")

    # setting training arguments
    max_seq_length: int = args.max_seq_len
    batch_size: int = int(args.batch_size)
    optimizer: str = args.optimizer
    learning_rate: float = args.learning_rate
    weight_decay: float = args.weight_decay
    warmup_steps: int = args.warmup_steps
    lora_r: int = args.lora_r
    lora_alpha: int = args.lora_alpha
    load_in_4bit: bool = args.load_in_4bit

    # fixed settings
    packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
    device_map = {"": 1}  # Load the entire model on the GPU 0

    # check that template mapping exists
    check_chat_template_mapping(args.model_name)



    # load custom configurations. If values need to be changed pass them as arguments
    lora_config: CustomLoraConfiguration = CustomLoraConfiguration(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    bnb_config: CustomBitsAndBitesConfiguration = CustomBitsAndBitesConfiguration(
        use_4bit=load_in_4bit,
    )
    training_arguments: CustomTrainingArguments = CustomTrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        fp16= not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=optimizer,
        hub_model_id=None,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
    )

    # prepare the model hub ID according to the specified format.
    training_arguments.hub_model_id =  prepare_model_hub_id(
        base_model_name=args.hf_model_name if args.hf_model_name else args.model_name.replace("/", "-"),
        learning_strategy="SFT",
        dataset_name=args.training_dataset,
    )
    logging.info(f"Model will be uploaded with model ID: {training_arguments.hub_model_id}")
    logging.info(f"Chat template {chat_template_mapping[args.model_name]} will be used")

    # Initialize model
    model: CustomTextToSqlModel = CustomTextToSqlModel(
        model_name=args.model_name,
        path_dataset_train=args.training_dataset,
        path_dataset_test=args.test_dataset,
        output_dir=args.output_dir,
        lora_config=lora_config,
        bnb_config=bnb_config,
        unsloth_config=CustomUnslothModelConfig(max_seq_length=max_seq_length),
        training_arguments=training_arguments,
        max_seq_length=max_seq_length,
        packing=packing,
        device_map=device_map,
        train=train,
        chat_template=chat_template_mapping[args.model_name],
    )

    # safe the configuration files
    model.save_config()

    # start the training
    if args.training_style in ["cold_start", "rehearsal"]:
        model.train_model()
    elif args.training_style == "warm_up":
        if args.warm_up_dataset is None:
            KeyError("When training with warm up, warm up dataset path must be provided using --warm_up_dataset.")
            exit(1)
        model.train_model_with_wramup(args.warm_up_dataset)

    elif args.training_style == "completion_only":
        model.train_model_with_completion_only_collator()
    elif args.training_style == "multi_step_completion":
        model.train_model_with_multi_step_completion_collator()

    else:
        logging.error(f"Unknown training style: {args.training_style} cose from available options:  cold_start, warm_up, rehearsal, completion_only, multi_step_completion")

    # save the model
    model.save_model()