import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import BaseAgent

class HuggingFaceAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8 #kwargs['batch_size']

    def init_pipeline(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
        )
        self.pipe.tokenizer.padding_side = "left"

    def preprocess_input(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def postprocess_pipeline_output(self, output):
        return output[0]['generated_text'].strip()

    def encode(self, texts):
        prompts = [self.preprocess_input(text) for text in texts]
        encoded_texts = self.tokenizer(prompts, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device)
        return encoded_texts

    def decode(self, outputs):
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]
        return responses

    def raw_batch_interact(self, texts, do_sample=True):
        encoded_texts = self.encode(texts)
        with torch.no_grad():
            outputs = self.model.generate(**encoded_texts, max_new_tokens=365, do_sample=do_sample)
        responses = self.decode(outputs)
        return responses

    def batch_interact(self, texts, do_sample=True):
        prompts = [self.preprocess_input(text) for text in texts] # XXX: apply those chat-specific templates beforehand and make them into pipeline batch and directly feed the pipeline
        outputs = self.pipe(prompts, return_full_text=False, max_new_tokens=365, do_sample=True)
        responses = [self.postprocess_pipeline_output(output) for output in outputs]

        return responses

    def interact(self, text, do_sample=True):
        return self.batch_interact([text], do_sample)[0]

    def batch_compute_likelihood(self, input_texts, target_data):
        """ Compute the log-likelihood of the target data given the input text. """
        # We should pad after concatenating with target_outputs
        prompts = [self.preprocess_input(text) for text in input_texts] # apply those chat-specific templates
        data_appended_prompt = [p + d for p, d in zip(prompts, target_data)] # append the target responses to the prompts
        encoded_texts = self.tokenizer(data_appended_prompt, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device)
        encoded_data = self.tokenizer(target_data, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device) # this is actually for getting the attention mask to know which part of the input is the response

        with torch.no_grad():
            outputs = self.model(**encoded_texts, return_dict=True)

        vocab_distribution = torch.log_softmax(outputs.logits, dim=-1)
        data_token_logprobs = torch.gather(vocab_distribution[:,:-1,:], 2, encoded_data.input_ids.unsqueeze(-1)[:,1:,:])
        true_data_token_logprobs = (data_token_logprobs * encoded_data.attention_mask.unsqueeze(-1)[:, 1:, :]).squeeze(-1) # get only the logprobs of the response tokens
        data_log_likelihood = true_data_token_logprobs.sum(dim=1) / encoded_data.attention_mask.sum(dim=1)

        return data_log_likelihood
    
    def compute_data_likelihood(self, input_text, target_datum):
        return self.batch_compute_likelihood([input_text], [target_datum])[0]


class HuggingFaceChatAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_output_token = "[/INST]"

    def preprocess_input(self, text):
        messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": text},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat_prompt

    def postprocess_output(self, response):
        return response.split(self.model_output_token)[-1].strip()

class Llama2Agent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['model_size'].lower() in ['7b', '13b', '70b']
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-hf", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-hf", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class Llama2ChatAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['model_size'].lower() in ['7b', '13b', '70b']
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-chat-hf", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-chat-hf", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class MistralAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        # self.init_pipeline()
        # self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class MistralInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class MixtralInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", padding_size='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"

class ZephyrAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.model_output_token = "<|assistant|>"
        self.init_pipeline()

class GemmaAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        assert kwargs['model_size'].lower() in ['2b', '7b']
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.init_pipeline()

class GemmaInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        assert kwargs['model_size'].lower() in ['2b', '7b']
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}-it", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}-it", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.model_output_token = "\nmodel\n"
        self.init_pipeline()
