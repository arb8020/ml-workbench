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
    print("loading GPT-2...")
    params, model_config = load_gpt2_params()
    print('loading Tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    sampler_config = SamplerConfig(temp=0.7, max_tokens=20)
    print("ready!")
    print('example: my chipotle order is: a burrito bowl with --temp=1.5 --min_p=0.05 --max_tokens=30')
    
    while True:
        try:
            user_input = input('gpt2> ')
            if user_input.lower() == 'exit':
                break
            prompt, sampler_config = parse_command(user_input, sampler_config)
            
            input_tokens = np.array(tokenizer.encode(prompt))
            
            output_tokens = generate_kv(
                params,
                model_config,
                'gpt2',
                input_tokens,
                sampler_config,
                stream=True,
                tokenizer=tokenizer
            )

            print(f'sampler config used: {sampler_config}')
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    run_cli()