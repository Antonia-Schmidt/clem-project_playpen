from unsloth import FastLanguageModel

model_to_merge = "llama-3.1-8B-Instruct-v1.6-only-full-precision-lora_v3_playpen_SFT-e3_DFINAL_0.7K-steps"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"clembench-playpen/{model_to_merge}",
)

#model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
model.push_to_hub_merged(f"clembench-playpen/llama-3.1-8B-Instruct-fp_SFT_e1_DFINAL_merged_fp16", tokenizer, save_method = "merged_16bit")
