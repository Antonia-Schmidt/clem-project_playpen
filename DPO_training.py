import os
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from trl import DPOTrainer, DPOConfig
import wandb
import argparse

def load_hf_dataset():
    dataset = load_dataset(f"{args.hf_repo}/DPO_{args.neg}neg", split = "train")
    dataset_dict = dataset.train_test_split(test_size=0.04)
    dataset_new = {'chosen':[], 'rejected':[]}
    #TODO: reduce to one line the following (one function)
    for data_point in dataset_dict['train']['chosen']:
      dataset_new['chosen'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['train']['rejected']:
      dataset_new['rejected'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['test']['chosen']:
      dataset_new['chosen'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    for data_point in dataset_dict['test']['rejected']:
      dataset_new['rejected'].append(tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    new_dataset = Dataset.from_dict(dataset_new)
    dataset_dict = new_dataset.train_test_split(test_size=0.04)
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPO training')
    parser.add_argument('--neg', default=1, type=int, help='number of negative samples per every positive one in the dataset')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', help='hf login token')
    parser.add_argument('--wandb_login', help='wandb login token')
    parser.add_argument('--cache_dir', default = '/mnt/cimec-storage6/shared/hf_llms_checkpoints/', help='cache directory to store models')
    parser.add_argument('--hf_repo', default='clembench-playpen', help='huggingface repository to store the created datasets')
    args = parser.parse_args()

    login(f"{args.hf_login}")
    os.environ['WANDB_API_KEY'] = f"{args.wandb_login}"
    wandb.login()
    cache_dir_models = args.cache_dir

    max_seq_length = 1024
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        cache_dir=cache_dir_models,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    dataset_dict = load_hf_dataset()

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # TODO: lora hyperparameter tuning Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    PatchDPOTrainer()

    project_name = "llama3.1_dpo_playpen"
    entity = "wandb"
    wandb.init(project=project_name, name="llama3.1_dpo_playpen")

    training_arguments = DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=5e-6,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        # max_steps=20,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",  # enable logging to W&B
        output_dir="outputs",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_arguments,
        beta=0.1,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length,
    )

    dpo_trainer.train()
    model.save_pretrained("playpen_lora")
    model.push_to_hub(f"{args.hf_repo}/meta-llama-Meta-Llama-3.1-8B-Instruct_DPO_{args.neg}neg")