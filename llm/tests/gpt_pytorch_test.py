import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Tuple, Any
from models import GPTConfig
from core import gelu_exact, layer_norm, grouped_query_attn, gpt2_forward
from test_utils import (
    convert_torch_to_jax, convert_jax_to_torch, 
    assert_equal_parameters, RTOL, ATOL, sample_input
)

def get_gpt2_param_mapping():
    """get mapping between JAX parameter paths and PyTorch parameter names."""
    return {
        'token_embedding': 'transformer.wte.weight',
        'positional_embedding': 'transformer.wpe.weight',
        'lnf.gamma': 'transformer.ln_f.weight',
        'lnf.beta': 'transformer.ln_f.bias',
        'output_projection': 'lm_head.weight'
    }

def get_gpt2_block_param_mapping(block_idx: int):
    """get parameter mapping for a transformer block."""
    prefix = f'transformer.h.{block_idx}.'
    return {
        f'block_{block_idx}.attn_in.weight': prefix + 'attn.c_attn.weight',
        f'block_{block_idx}.attn_in.bias': prefix + 'attn.c_attn.bias',
        f'block_{block_idx}.attn_out.weight': prefix + 'attn.c_proj.weight',
        f'block_{block_idx}.attn_out.bias': prefix + 'attn.c_proj.bias',
        f'block_{block_idx}.ln1.gamma': prefix + 'ln_1.weight',
        f'block_{block_idx}.ln1.beta': prefix + 'ln_1.bias',
        f'block_{block_idx}.ln2.gamma': prefix + 'ln_2.weight',
        f'block_{block_idx}.ln2.beta': prefix + 'ln_2.bias',
        f'block_{block_idx}.ffn_in.weight': prefix + 'mlp.c_fc.weight',
        f'block_{block_idx}.ffn_in.bias': prefix + 'mlp.c_fc.bias',
        f'block_{block_idx}.ffn_out.weight': prefix + 'mlp.c_proj.weight',
        f'block_{block_idx}.ffn_out.bias': prefix + 'mlp.c_proj.bias'
    }

@pytest.fixture
def gpt2_models():
    """initialize matched GPT2 models in PyTorch and JAX."""
    config = GPT2Config(
        vocab_size=100,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=128,
        activation_function='gelu'
    )
    
    torch_model = GPT2LMHeadModel(config)
    torch_model.eval()
    
    jax_config = GPTConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.n_embd,
        context_len=config.n_positions,
        n_heads=config.n_head,
        n_layers=config.n_layer
    )
    
    jax_params = {}
    
    jax_params['token_embedding'] = convert_torch_to_jax(torch_model.transformer.wte.weight)
    jax_params['positional_embedding'] = convert_torch_to_jax(torch_model.transformer.wpe.weight)
    
    jax_params['lnf'] = {
        'gamma': convert_torch_to_jax(torch_model.transformer.ln_f.weight),
        'beta': convert_torch_to_jax(torch_model.transformer.ln_f.bias)
    }
    
    jax_params['output_projection'] = convert_torch_to_jax(torch_model.lm_head.weight, transpose_weights=True)
    
    for i in range(config.n_layer):
        block = torch_model.transformer.h[i]
        jax_params[f'block_{i}'] = {
            'attn_in': {
                'weight': convert_torch_to_jax(block.attn.c_attn.weight),
                'bias': convert_torch_to_jax(block.attn.c_attn.bias)
            },
            'attn_out': {
                'weight': convert_torch_to_jax(block.attn.c_proj.weight),
                'bias': convert_torch_to_jax(block.attn.c_proj.bias)
            },
            'ln1': {
                'gamma': convert_torch_to_jax(block.ln_1.weight),
                'beta': convert_torch_to_jax(block.ln_1.bias)
            },
            'ln2': {
                'gamma': convert_torch_to_jax(block.ln_2.weight),
                'beta': convert_torch_to_jax(block.ln_2.bias)
            },
            'ffn_in': {
                'weight': convert_torch_to_jax(block.mlp.c_fc.weight),
                'bias': convert_torch_to_jax(block.mlp.c_fc.bias)
            },
            'ffn_out': {
                'weight': convert_torch_to_jax(block.mlp.c_proj.weight),
                'bias': convert_torch_to_jax(block.mlp.c_proj.bias)
            }
        }
    
    return jax_params, jax_config, torch_model

def test_gelu_equivalence():
    """test that JAX GELU implementation matches PyTorch."""
    x = jnp.linspace(-5, 5, 100)
    torch_x = torch.from_numpy(np.array(x))
    
    jax_output = gelu_exact(x)
    torch_output = torch.nn.functional.gelu(torch_x)
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )

def test_layer_norm_equivalence(gpt2_models):
    """test that JAX layer normalization matches PyTorch."""
    jax_params, config, torch_model = gpt2_models
    
    x = jnp.array(np.random.randn(10, config.embedding_dim))
    torch_x = convert_jax_to_torch(x)
    
    jax_output = layer_norm(
        x,
        jax_params['lnf']['gamma'],
        jax_params['lnf']['beta']
    )
    
    torch_output = torch_model.transformer.ln_f(torch_x)
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )

def test_attention_equivalence(gpt2_models):
    """test that JAX attention mechanism matches PyTorch."""
    jax_params, config, torch_model = gpt2_models
    block_idx = 0
    seq_len = 8

    x = jnp.array(np.random.randn(seq_len, config.embedding_dim))
    torch_x = convert_jax_to_torch(x).unsqueeze(0)  # Add batch dimension for PyTorch

    mask = jnp.triu(jnp.ones((seq_len, seq_len)) * float('-inf'), k=1)
    torch_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    torch_mask = torch_mask.masked_fill(torch_mask == 0, float('-inf'))

    block_params = jax_params[f'block_{block_idx}']

    jax_output, _ = grouped_query_attn(
        x,
        block_params['attn_in']['weight'],
        block_params['attn_in']['bias'],
        block_params['attn_out']['weight'],
        block_params['attn_out']['bias'],
        config,
        mask,
        block_idx
    )

    torch_block = torch_model.transformer.h[block_idx]
    with torch.no_grad():
        torch_output = torch_block.attn(torch_x, attention_mask=torch_mask)[0]

    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output.squeeze(0)),
        rtol=RTOL,
        atol=ATOL
    )

def test_full_forward_pass(gpt2_models, sample_input):
    """test that full model forward pass matches PyTorch."""
    jax_params, config, torch_model = gpt2_models
    jax_tokens, torch_tokens = sample_input

    mask = jnp.triu(jnp.ones((len(jax_tokens), len(jax_tokens))) * float('-inf'), k=1)
    
    jax_output, _ = gpt2_forward(jax_params, config, jax_tokens, mask)
    
    with torch.no_grad():
        torch_output = torch_model(torch_tokens).logits
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )

def test_numerical_stability_gpt2(gpt2_models):
    """test model behavior with extreme input values."""
    jax_params, config, torch_model = gpt2_models
    
    large_tokens = jnp.array([config.vocab_size-1] * 8)
    torch_large_tokens = torch.tensor([config.vocab_size-1] * 8)
    
    mask = jnp.triu(jnp.ones((len(large_tokens), len(large_tokens))) * float('-inf'), k=1)
    
    jax_output, _ = gpt2_forward(jax_params, config, large_tokens, mask)
    with torch.no_grad():
        torch_output = torch_model(torch_large_tokens).logits
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )
    
    zero_tokens = jnp.zeros(8, dtype=jnp.int32)
    torch_zero_tokens = torch.zeros(8, dtype=torch.long)
    
    jax_output, _ = gpt2_forward(jax_params, config, zero_tokens, mask)
    with torch.no_grad():
        torch_output = torch_model(torch_zero_tokens).logits
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )

def test_parameter_conversion(gpt2_models):
    """test that parameter conversion between frameworks is correct."""
    jax_params, _, torch_model = gpt2_models
    
    param_mapping = get_gpt2_param_mapping()
    for i in range(len([k for k in jax_params.keys() if 'block_' in k])):
        param_mapping.update(get_gpt2_block_param_mapping(i))
    
    torch_params = dict(torch_model.named_parameters())
    
    assert_equal_parameters(jax_params, torch_params, param_mapping)