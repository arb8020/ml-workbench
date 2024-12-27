from typing import NamedTuple, Tuple, Union
import jax
from models import load_gpt2_params, load_llama_params
from training import generate, SamplerConfig
from core import initialize_rotation_factors
import numpy as np
from transformers import GPT2Tokenizer, LlamaTokenizer, AutoTokenizer  # Import LlamaTokenizer
import torch
from pathlib import Path
import json
# needs blobfile pip installed
# needs tiktoken pip installed 

def parse_command(input_str: str, sampler_config: SamplerConfig) -> Tuple[str, SamplerConfig]:
    parts = input_str.split('--')
    prompt = parts[0].strip()
    
    if len(parts) > 1:
        params = {}
        
        for part in parts[1:]:
            if part.startswith('temp='):
                params['temp'] = float(part.split('=')[1])
            elif part.startswith('max_tokens='):
                params['max_tokens'] = int(part.split('=')[1])
            elif part.startswith('min_p='):
                params['min_p'] = float(part.split('=')[1])
            elif part.startswith('top_p='):
                params['top_p'] = float(part.split('=')[1])
            elif part.startswith('top_k='):
                params['top_k'] = int(part.split('=')[1])
            elif part.startswith('key='):
                params['key'] = jax.random.PRNGKey(int(part.split('=')[1]))
            
        return prompt, sampler_config._replace(**params)
    
    return prompt, sampler_config

def run_cli():
    model_choice = input('select a model: (0: LLaMA, 1: GPT-2): ').strip()
    choice_map = {'0': 'llama', '1': 'gpt2'}
    
    if model_choice not in choice_map:
        print("invalid choice. please select '0' for LLaMA or '1' for GPT-2.")
        return
    
    model_name = choice_map[model_choice]
    tokenizer = None 
    
    if model_name == 'llama':
        path_input = input("please enter the path to where you've downloaded LLaMA 3.2's weights/etc: ").strip()
        try:
            print('loading llama')
            params, model_config = load_llama_params(path_input)
            print("loading LLaMA tokenizer...")
            model_id = "meta-llama/Llama-3.2-1B"
            tokenizer = AutoTokenizer.from_pretrained(path_input, local_files_only=True, 
                unk_token="<unk>",
                pad_token="<pad>",
                eos_token="</s>",
                bos_token="<s>"
            )
        except Exception as e:
            print(f"failed to load LLaMA model or tokenizer: {e}")
            return

    elif model_name == 'gpt2':
        try:
            print("loading GPT-2...")
            params, model_config = load_gpt2_params()
            print('loading GPT-2 Tokenizer...')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            print("GPT-2 Tokenizer loaded.")
            
        except Exception as e:
            print(f"failed to load GPT-2 model or tokenizer: {e}")
            return

    # Initialize sampler_config regardless of the model
    sampler_config = SamplerConfig(temp=0.7, max_tokens=20)
    
    print("ready!")
    if model_name == 'gpt2':
        print('example: my chipotle order is: a burrito bowl with --temp=1.5 --min_p=0.05 --max_tokens=30')
    elif model_name == 'llama':
        rotation_factors = initialize_rotation_factors(
            model_config.embedding_dim // model_config.n_heads,
            model_config.context_len,
            model_config.rope_theta
        )
        model_kwargs = {'rotation_factors': rotation_factors}
        print('example: my chipotle order is: a burrito bowl with --temp=1.0 --min_p=0.1 --max_tokens=50')
    
    while True:
        user_input = input(f'{model_name}> ').strip()
        if user_input.lower() == 'exit':
            print("exiting CLI.")
            break
        if not user_input:
            continue  
        prompt, sampler_config = parse_command(user_input, sampler_config)
        
        if not tokenizer:
            print("tokenizer is not loaded.")
            continue
        
        input_tokens = np.array(tokenizer.encode(prompt))
        
        output_tokens = generate(
            params,
            model_config,
            model_name,
            input_tokens,
            sampler_config,
            model_kwargs,
            stream=True,
            tokenizer=tokenizer

        )

        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f'generated text: {generated_text}')
        print(f'sampler config used: {sampler_config}')
            
        # except Exception as e:
        #     print(f"error: {e}")

if __name__ == '__main__':
    run_cli()