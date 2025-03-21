from unsloth import FastLanguageModel
from huggingface_hub import login

login(f"")
model_to_merge = 'llama-3.1-70B-Instruct_playpen_SFT_DFINAL_0.6K-steps'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"clembench-playpen/{model_to_merge}",
)

model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
model.push_to_hub_merged(f"clembench-playpen/{model_to_merge}_merged_fp16", tokenizer, save_method = "merged_16bit")
