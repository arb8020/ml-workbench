from typing import NamedTuple, Tuple
import jax
from model import ModelConfig, load_gpt2_params 
from training import generate, SamplerConfig
import numpy as np
from transformers import GPT2Tokenizer

def parse_command(input_str: str, sampler_config: SamplerConfig) -> Tuple[str, SamplerConfig]:
    parts = input_str.split('--')
    prompt = parts[0].strip()
    
    if len(prompt) > 1:
        params = {}
        
        for part in parts[1:]:
            if part.startswith('temp='):
                params['temp'] = float(part.split('=')[1])
            elif part.startswith('tokens='):
                params['max_tokens'] = int(part.split('=')[1])
            elif part.startswith('top_k='):
                params['top_k'] = int(part.split('=')[1])
            
        return prompt, sampler_config._replace(**params)
    
    return prompt, sampler_config

def run_cli():
    print("loading GPT-2...")
    params, model_config = load_gpt2_params()
    print('loading Tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    sampler_config = SamplerConfig(temp=0.7, max_tokens=100, top_k=40)
    print("ready!")
    
    while True:
        try:
            user_input = input('gpt2> ')
            if user_input.lower() == 'exit':
                break
                
            prompt, sampler_config = parse_command(user_input, sampler_config)
            input_tokens = np.array(tokenizer.encode(prompt))
            
            output_tokens = generate(
                params,
                model_config,
                'gpt2',
                input_tokens,
                sampler_config,
                stream=True,
                tokenizer=tokenizer
            )
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    run_cli()