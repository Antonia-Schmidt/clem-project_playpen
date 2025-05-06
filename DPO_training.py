import os
from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login, create_repo
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
import argparse
import time
import json

def load_hf_dataset(tokenizer):
    dataset = load_dataset(f"{args.hf_repo_target}/{args.dataset_name}", split="train")

    initial_split = dataset.train_test_split(test_size=0.04)
    train_dataset = initial_split['train']

    #TODO: check from herer to test_dataset
    unique_games = set(dataset['game'])
    test_indices = []

    for game in unique_games:
        game_indices = [i for i, g in enumerate(dataset['game']) if g == game]
        samples_per_game = max(1, int(len(dataset) * 0.04 / len(unique_games)))
        samples_to_take = min(samples_per_game, len(game_indices))
        test_indices.extend(game_indices[:samples_to_take])

    test_dataset = dataset.select(test_indices)
    dataset_new = {'chosen': [], 'rejected': []}
    for split_name, split_data in [('train', train_dataset), ('test', test_dataset)]:
        for choice_type in ['chosen', 'rejected']:
            for data_point in split_data[choice_type]:
                dataset_new[choice_type].append(
                    tokenizer.apply_chat_template(data_point, tokenize=False, add_generation_prompt=False))
    new_dataset = Dataset.from_dict(dataset_new)
    test_size = len(test_dataset) / (len(train_dataset) + len(test_dataset))
    dataset_dict = new_dataset.train_test_split(test_size=test_size)
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPO training')
    #TODO: infer model condition from the dataset and not given as parameter
    parser.add_argument('--aborted_interactions', default=True, choices=[True, False], help='integrating aborted interactions as negative samples')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', default="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", help='hf login token')
    parser.add_argument('--wandb_login', default="29fc614754925fca7f38d6d2193b3f5afa8485a9", help='wandb login token')
    parser.add_argument('--base_model', default="llama-SFT-base_merged_fp16_D90053_copy_32GB", help='base model for training')
    parser.add_argument('--dataset_name', help='base model for training')
    parser.add_argument('--cache_dir', default = '/mnt/cimec-storage6/shared/hf_llms_checkpoints/', help='cache directory to store models')
    parser.add_argument('--hf_repo_base', default='clembench-playpen', help='huggingface repository where the base model is stored')
    parser.add_argument('--hf_repo_target', default='clembench-playpen', help='huggingface repository to save the trained model')
    parser.add_argument('--use_unsloth', default=False, choices=[True, False], help='huggingface repository to save the trained model')
    #parser.add_argument('--players', default="player 1", choices=['player 1', 'all'])


    args = parser.parse_args()

    login(f"{args.hf_login}")
    os.environ['WANDB_API_KEY'] = f"{args.wandb_login}"
    wandb.login()
    cache_dir_models = args.cache_dir

    max_seq_length = 1024
    model_name = f"{args.hf_repo_base}/{args.base_model}"

    if args.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            cache_dir=cache_dir_models,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            #fix_tokenizer=False     #This is works only if the base model is llama3.1 unsloth 4bit
        )
        tokenizer.truncation_side = 'left'      #is this needed? (keep_last)
        dataset_dict = load_hf_dataset(tokenizer)

        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir_models,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        dataset_dict = load_hf_dataset(tokenizer)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ])
        model = get_peft_model(base_model, lora_config)

    PatchDPOTrainer()

    project_name = f"playpen_{args.base_model}_F"
    entity = "wandb"
    wandb.init(project=project_name, name=f"dpo_{args.dataset_name}")

    training_arguments = DPOConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=3,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=5e-6,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # max_steps=20,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        report_to="wandb",
        #push_to_hub=False,
        output_dir="./outputs", #f"{args.hf_repo}/meta-llama-Meta-Llama-3.1-8B-Instruct_DPO_{args.neg}neg{'_'+args.model_condition if args.model_condition else ''}_sblurry___",       #TODO: this does not work with dpo_trainer.push_to_hub
        logging_steps=1,
        #TODO: use this lines to avoid overfitting with old and new data
        eval_steps=0.2,  # 0.2
        save_steps=0.2,  # 0.2
        eval_strategy = "steps",
        logging_strategy = "steps",
        save_strategy = "steps",
        load_best_model_at_end=True
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

    #TODO: migliora questa funzione (riduci, sposta)
    train_dataloader = dpo_trainer.get_train_dataloader()
    #Modified from the original HF trainer.py
    def compute_tokens(train_dl: train_dataloader, tokenizer) -> int:
        train_tokens = 0
        pad_token_id = tokenizer.pad_token_id
        for batch in train_dl:
            chosen_mask = batch["chosen_input_ids"] != pad_token_id
            tokens_chosen = chosen_mask.sum().item()
            rejected_mask = batch["rejected_input_ids"] != pad_token_id
            tokens_rejected = rejected_mask.sum().item()
            tokens = tokens_chosen + tokens_rejected
            train_tokens += tokens
        return train_tokens

    training_tokens = compute_tokens(train_dataloader, tokenizer)
    dataset_type = 'dialogue' if 'dialogue' in args.dataset_name else 'turn'
    trained_model_id = f"{args.base_model}_{dataset_type}"
    model_hub_id = f"{args.hf_repo_target}/{trained_model_id}"

    create_repo(repo_id=model_hub_id, repo_type="model", private=False, exist_ok=True)
    dpo_trainer.hub_model_id = model_hub_id

    start_time = time.time()
    dpo_trainer.train()
    end_time = time.time()

    model.save_pretrained("playpen_lora") #TODO: needed to save_pretrained?
    dpo_trainer.push_to_hub()
