import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Optional, Tuple, Union
from transformers import GPT2LMHeadModel
from functools import partial
import torch
from pathlib import Path
import json

# Model Parameters

class GPTConfig(NamedTuple):
    """
    defines model parameters for gpt2 architecture models
    """
    vocab_size: int
    embedding_dim: int
    context_len: int
    n_heads: int
    n_layers: int
    n_kv_heads: Optional[int] = None

class LlamaConfig(NamedTuple):
    """
    defines model parameters for llama architecture models
    """
    vocab_size: int
    embedding_dim: int
    ffn_dim_multiplier: float
    multiple_of: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    context_len: int

ModelConfig = Union[GPTConfig, LlamaConfig]

def init_gpt2_params(key, model_config: ModelConfig, scaling_factor: float = 0.02) -> Dict[str, Any]:
    """
    template for initializing parameters of a GPT2 style transformer model
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    keys = jax.random.split(key, 7 + 10 * model_config.n_layers)

    params = {
        'token_embedding': jax.random.normal(keys[0], (model_config.vocab_size, model_config.embedding_dim)) * scaling_factor,
        'positional_embedding': jax.random.normal(keys[1], (model_config.context_len, model_config.embedding_dim)) * scaling_factor,
        'output_projection': jax.random.normal(keys[2], (model_config.embedding_dim, model_config.vocab_size)) * scaling_factor,
        'lnf': {
            'gamma': jnp.ones((model_config.embedding_dim,)),
            'beta': jnp.zeros((model_config.embedding_dim,)),
        },
    }

    for i in range(model_config.n_layers):
        block_key_start = 3 + i * 10
        params[f'block_{i}'] = {
            'attn_in': {
                'weight': jax.random.normal(keys[block_key_start], (model_config.embedding_dim, 3*model_config.embedding_dim)) * scaling_factor,
                'bias': jax.random.normal(keys[block_key_start+1], (3*model_config.embedding_dim,)) * scaling_factor,
            },
            'attn_out': {
                'weight': jax.random.normal(keys[block_key_start+2], (model_config.embedding_dim, model_config.embedding_dim)) * scaling_factor,
                'bias': jax.random.normal(keys[block_key_start+3], (model_config.embedding_dim,)) * scaling_factor,
            },
            'ln1': {
                'gamma': jnp.ones((model_config.embedding_dim,)),
                'beta': jnp.zeros((model_config.embedding_dim,)),
            },
            'ln2': {
                'gamma': jnp.ones((model_config.embedding_dim,)),
                'beta': jnp.zeros((model_config.embedding_dim,)),
            },
            'ffn_in': {
                'weight': jax.random.normal(keys[block_key_start+4], (model_config.embedding_dim, 4*model_config.embedding_dim)) * scaling_factor,
                'bias': jax.random.normal(keys[block_key_start+5], (4*model_config.embedding_dim,)) * scaling_factor,
            },
            'ffn_out': {
                'weight': jax.random.normal(keys[block_key_start+6], (4*model_config.embedding_dim, model_config.embedding_dim)) * scaling_factor,
                'bias': jax.random.normal(keys[block_key_start+7], (model_config.embedding_dim,)) * scaling_factor,
            },
        }

    return params

def load_gpt2_params(model_name: str = "gpt2") -> Tuple[Dict[str, Any], ModelConfig]:
    """
    loads GPT2 weights from hugging face and puts them into the predefined template
    radford et al, 2019: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config

    
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.n_embd,
        context_len=config.n_positions,
        n_head=config.n_head,
        n_layers=config.n_layer
    )

    converted_params = {}
    converted_params['token_embedding'] = jnp.array(model.transformer.wte.weight.detach().numpy())
    converted_params['positional_embedding'] = jnp.array(model.transformer.wpe.weight.detach().numpy())
    converted_params['output_projection'] = jnp.array(model.lm_head.weight.detach().numpy().T)
    converted_params['lnf'] = {
        'gamma': jnp.array(model.transformer.ln_f.weight.detach().numpy()),
        'beta': jnp.array(model.transformer.ln_f.bias.detach().numpy())
    }

    for i in range(model_config.n_layers):
        block = model.transformer.h[i]
        converted_block = {
            'attn_in': {
                'weight': jnp.array(block.attn.c_attn.weight.detach().numpy()),
                'bias': jnp.array(block.attn.c_attn.bias.detach().numpy())
            },
            'attn_out': {
                'weight': jnp.array(block.attn.c_proj.weight.detach().numpy()),
                'bias': jnp.array(block.attn.c_proj.bias.detach().numpy())
            },
            'ln1': {
                'gamma': jnp.array(block.ln_1.weight.detach().numpy()),
                'beta': jnp.array(block.ln_1.bias.detach().numpy())
            },
            'ln2': {
                'gamma': jnp.array(block.ln_2.weight.detach().numpy()),
                'beta': jnp.array(block.ln_2.bias.detach().numpy())
            },
            'ffn_in': {
                'weight': jnp.array(block.mlp.c_fc.weight.detach().numpy()),
                'bias': jnp.array(block.mlp.c_fc.bias.detach().numpy())
            },
            'ffn_out': {
                'weight': jnp.array(block.mlp.c_proj.weight.detach().numpy()),
                'bias': jnp.array(block.mlp.c_proj.bias.detach().numpy())
            },
        }
        converted_params[f'block_{i}'] = converted_block

    return converted_params, model_config

def init_llama_params(key, config: LlamaConfig, scaling_factor: float = 0.02) -> Dict[str, Any]:
    keys = jax.random.split(key, 2 + 8 * config.n_layers)
    
    params = {
        'tok_embeddings': jax.random.normal(keys[0], (config.vocab_size, config.embedding_dim)) * scaling_factor,
        'output': jax.random.normal(keys[1], (config.vocab_size, config.embedding_dim)) * scaling_factor,
        'norm': {
            'weight': jnp.ones((config.embedding_dim,))
        }
    }

    head_dim = config.embedding_dim // config.n_heads
    for i in range(config.n_layers):
        block_key_start = 2 + i * 8
        params[f'layer_{i}'] = {
            'attention': {
                'wq': jax.random.normal(keys[block_key_start], (config.embedding_dim, config.embedding_dim)) * scaling_factor,
                'wk': jax.random.normal(keys[block_key_start+1], (config.embedding_dim, head_dim * config.n_kv_heads)) * scaling_factor,
                'wv': jax.random.normal(keys[block_key_start+2], (config.embedding_dim, head_dim * config.n_kv_heads)) * scaling_factor,
                'wo': jax.random.normal(keys[block_key_start+3], (config.embedding_dim, config.embedding_dim)) * scaling_factor
            },
            'feed_forward': {
                'w1': jax.random.normal(keys[block_key_start+4], (config.embedding_dim, int(config.embedding_dim * config.ffn_dim_multiplier))) * scaling_factor,
                'w2': jax.random.normal(keys[block_key_start+5], (int(config.embedding_dim * config.ffn_dim_multiplier), config.embedding_dim)) * scaling_factor,
                'w3': jax.random.normal(keys[block_key_start+6], (config.embedding_dim, int(config.embedding_dim * config.ffn_dim_multiplier))) * scaling_factor
            },
            'attention_norm': {'weight': jnp.ones((config.embedding_dim,))},
            'ffn_norm': {'weight': jnp.ones((config.embedding_dim,))}
        }
    return params

def load_llama_params(model_path: str = "Llama3.2-1B") -> Tuple[Dict[str, Any], LlamaConfig]:
    model_path = Path(model_path)
    
    with open(model_path / "params.json", 'r') as f:
        config = json.load(f)
    config["embedding_dim"] = config.pop("dim")
    config["context_len"] = 8192
    model_config = LlamaConfig(**config)
    
    state_dict = torch.load(model_path / "consolidated.00.pth", map_location="cpu", weights_only=True)
    
    def to_numpy(x):
        return x.to(torch.float32).numpy()
    
    converted_params = {
        'tok_embeddings': jnp.array(to_numpy(state_dict['tok_embeddings.weight'])),
        'output': jnp.array(to_numpy(state_dict['output.weight'])),
        'norm': {'weight': jnp.array(to_numpy(state_dict['norm.weight']))}
    }
    
    for i in range(model_config.n_layers):
        prefix = f'layers.{i}.'
        converted_params[f'layer_{i}'] = {
            'attention': {
                'wq': jnp.array(to_numpy(state_dict[f'{prefix}attention.wq.weight'])).T,
                'wk': jnp.array(to_numpy(state_dict[f'{prefix}attention.wk.weight'])).T,
                'wv': jnp.array(to_numpy(state_dict[f'{prefix}attention.wv.weight'])).T,
                'wo': jnp.array(to_numpy(state_dict[f'{prefix}attention.wo.weight'])).T
            },
            'feed_forward': {
                'w1': jnp.array(to_numpy(state_dict[f'{prefix}feed_forward.w1.weight'])).T,
                'w2': jnp.array(to_numpy(state_dict[f'{prefix}feed_forward.w2.weight'])).T,
                'w3': jnp.array(to_numpy(state_dict[f'{prefix}feed_forward.w3.weight'])).T
            },
            'attention_norm': {'weight': jnp.array(to_numpy(state_dict[f'{prefix}attention_norm.weight']))},
            'ffn_norm': {'weight': jnp.array(to_numpy(state_dict[f'{prefix}ffn_norm.weight']))}
        }
    
    return converted_params, model_config
