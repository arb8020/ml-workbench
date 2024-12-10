import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Tuple, Any
from model import ModelConfig, gpt2_forward, gelu_exact, layer_norm, multi_head_attn
from training import (
    init_optimizer, AdamConfig, SGDConfig, MomentumConfig, 
    adamw_update, sgd_update, momentum_update
)
from functools import partial

# constants
RTOL = 1e-4  
ATOL = 1e-4  

# utility Functions
def convert_torch_to_jax(tensor: torch.Tensor, transpose_weights: bool = False) -> jax.Array:
    """
    convert PyTorch tensor to JAX array.
    transpose_weights: if True, transpose weight matrices to match JAX convention
    """
    array = jnp.array(tensor.detach().cpu().numpy())
    if transpose_weights:
        return array.T
    return array

def convert_jax_to_torch(array: jax.Array, transpose_weights: bool = False) -> torch.Tensor:
    """convert JAX array to PyTorch tensor."""
    if transpose_weights:
        array = array.T
    return torch.from_numpy(np.array(array))

def get_param_mapping():
    """returns mapping between JAX parameter paths and PyTorch parameter names."""
    return {
        'token_embedding': 'transformer.wte.weight',
        'positional_embedding': 'transformer.wpe.weight',
        'lnf.gamma': 'transformer.ln_f.weight',
        'lnf.beta': 'transformer.ln_f.bias',
        'output_projection': 'lm_head.weight'
    }

def get_block_param_mapping(block_idx: int):
    """returns parameter mapping for a transformer block."""
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

def compare_parameters(jax_params: Dict, torch_model: torch.nn.Module, rtol: float = RTOL, atol: float = ATOL):
    """compare JAX and PyTorch parameters."""
    # build complete parameter mapping
    param_mapping = get_param_mapping()
    for i in range(len(jax_params.keys())):
        if f'block_{i}' in jax_params:
            param_mapping.update(get_block_param_mapping(i))
    
    # get flattened JAX parameters with their paths
    flat_jax_params = {
        '/'.join(str(x) for x in path): value 
        for path, value in jax.tree_util.tree_leaves_with_path(jax_params)
    }
    
    # convert PyTorch parameters to dict
    torch_params = dict(torch_model.named_parameters())
    
    # compare parameters
    for jax_path, jax_value in flat_jax_params.items():
        if jax_path not in param_mapping:
            continue
            
        torch_name = param_mapping[jax_path]
        torch_param = torch_params[torch_name]
        
        # determine if this parameter needs transposition
        needs_transpose = 'weight' in torch_name and 'embedding' not in torch_name
        
        np.testing.assert_allclose(
            jax_value,
            convert_torch_to_jax(torch_param, transpose_weights=needs_transpose),
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch in parameter {jax_path} (torch: {torch_name})"
        )

def load_identical_models() -> Tuple[Dict[str, Any], ModelConfig, GPT2LMHeadModel]:
    """load identical models in both frameworks with same weights."""
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
    
    jax_config = ModelConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.n_embd,
        context_len=config.n_positions,
        n_head=config.n_head,
        n_blocks=config.n_layer
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

# Fixtures
@pytest.fixture
def models():
    """fixture providing initialized models with identical weights."""
    return load_identical_models()

@pytest.fixture
def sample_input():
    """fixture providing sample input sequence."""
    return jnp.array([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])

@pytest.fixture
def optimizer_configs():
    """fixture providing optimizer configurations."""
    return {
        'adam': AdamConfig(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01),
        'sgd': SGDConfig(lr=0.01),
        'momentum': MomentumConfig(lr=0.01, beta=0.9)
    }

def test_gelu_equivalence():
    """test that our GELU implementation matches PyTorch's."""
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

def test_layer_norm_equivalence(models):
    """test that our layer normalization matches PyTorch's."""
    jax_params, config, torch_model = models
    
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

def test_attention_equivalence(models):
    """test that our attention mechanism matches PyTorch's."""
    jax_params, config, torch_model = models
    block_idx = 0

    seq_len = 8
    batch_size = 1
    x = jnp.array(np.random.randn(batch_size, seq_len, config.embedding_dim))
    torch_x = convert_jax_to_torch(x)

    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    
    torch_mask = torch.tril(torch.ones(seq_len, seq_len))
    torch_mask = torch_mask.masked_fill(torch_mask == 0, float('-inf'))
    torch_mask = torch_mask.unsqueeze(0)  

    block_params = jax_params[f'block_{block_idx}']

    jax_output = jax.vmap(
        multi_head_attn,
        in_axes=(0, None, None, None, None, None, None)  
    )(
        x,
        block_params['attn_in']['weight'],
        block_params['attn_in']['bias'],
        block_params['attn_out']['weight'],
        block_params['attn_out']['bias'],
        config.n_head,
        causal_mask
    )

    torch_block = torch_model.transformer.h[block_idx]
    with torch.no_grad():
        torch_output = torch_block.attn(torch_x, attention_mask=torch_mask)[0]

    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )


def test_full_forward_pass(models, sample_input):
    """test that full model forward pass matches PyTorch."""
    jax_params, config, torch_model = models
    jax_tokens, torch_tokens = sample_input
    
    jax_output = gpt2_forward(jax_params, config, jax_tokens)
    
    with torch.no_grad():
        torch_output = torch_model(torch_tokens).logits
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )

def test_sgd_equivalence(models, optimizer_configs):
    """test SGD optimizer matches PyTorch."""
    jax_params, config, torch_model = models
    
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=optimizer_configs['sgd'].lr)
    jax_opt_state, _, _ = init_optimizer('sgd', jax_params, **optimizer_configs['sgd']._asdict())
    
    dummy_grads = jax.tree_map(lambda x: jnp.ones_like(x), jax_params)
    torch_grads = {
        name: torch.ones_like(param) for name, param in torch_model.named_parameters()
    }
    
    new_jax_params, _ = sgd_update(jax_params, dummy_grads, {}, optimizer_configs['sgd'])
    
    for param, grad in zip(torch_model.parameters(), torch_grads.values()):
        param.grad = grad
    torch_optimizer.step()
    
    compare_parameters(new_jax_params, torch_model)

def test_momentum_equivalence(models, optimizer_configs):
    """test momentum optimizer matches PyTorch."""
    jax_params, config, torch_model = models
    
    torch_optimizer = torch.optim.SGD(
        torch_model.parameters(),
        lr=optimizer_configs['momentum'].lr,
        momentum=optimizer_configs['momentum'].beta
    )
    jax_opt_state, _, _ = init_optimizer('momentum', jax_params, **optimizer_configs['momentum']._asdict())
    
    dummy_grads = jax.tree_map(lambda x: jnp.ones_like(x), jax_params)
    torch_grads = {
        name: torch.ones_like(param) for name, param in torch_model.named_parameters()
    }
    
    new_jax_params, new_jax_state = momentum_update(
        jax_params, dummy_grads, jax_opt_state, optimizer_configs['momentum']
    )
    
    for param, grad in zip(torch_model.parameters(), torch_grads.values()):
        param.grad = grad
    torch_optimizer.step()
    
    compare_parameters(new_jax_params, torch_model)

def test_adam_equivalence(models, optimizer_configs):
    """test Adam optimizer matches PyTorch."""
    jax_params, config, torch_model = models
    
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=optimizer_configs['adam'].lr,
        betas=(optimizer_configs['adam'].beta1, optimizer_configs['adam'].beta2),
        eps=optimizer_configs['adam'].eps,
        weight_decay=optimizer_configs['adam'].weight_decay
    )
    jax_opt_state, _, _ = init_optimizer('adam', jax_params, **optimizer_configs['adam']._asdict())
    
    dummy_grads = jax.tree_map(lambda x: jnp.ones_like(x), jax_params)
    torch_grads = {
        name: torch.ones_like(param) for name, param in torch_model.named_parameters()
    }
    
    # apply multiple updates to test momentum behavior
    for i in range(5):
        new_jax_params, jax_opt_state = adamw_update(
            jax_params, dummy_grads, jax_opt_state, optimizer_configs['adam']
        )
        jax_params = new_jax_params
        
        for param, grad in zip(torch_model.parameters(), torch_grads.values()):
            param.grad = grad
        torch_optimizer.step()
        
        compare_parameters(new_jax_params, torch_model)

def test_optimizer_momentum_accumulation(models, optimizer_configs):
    """test that momentum states are accumulated correctly."""
    jax_params, config, torch_model = models
    
    jax_opt_state, _, _ = init_optimizer('adam', jax_params, **optimizer_configs['adam']._asdict())
    
    grad_sequence = [
        jax.tree_map(lambda x: jnp.ones_like(x), jax_params),
        jax.tree_map(lambda x: jnp.ones_like(x) * 2, jax_params),
        jax.tree_map(lambda x: jnp.ones_like(x) * -1, jax_params)
    ]
    
    for grads in grad_sequence:
        new_jax_params, jax_opt_state = adamw_update(
            jax_params, grads, jax_opt_state, optimizer_configs['adam']
        )
        jax_params = new_jax_params
        
        assert any(jnp.any(v != 0) for v in jax.tree_leaves(jax_opt_state['gradient_mean']))
        assert any(jnp.any(v != 0) for v in jax.tree_leaves(jax_opt_state['gradient_squared_mean']))

def test_numerical_stability(models):
    """test model behavior with extreme input values."""
    jax_params, config, torch_model = models
    
    large_tokens = jnp.array([config.vocab_size-1] * 8)
    torch_large_tokens = torch.tensor([config.vocab_size-1] * 8)
    
    jax_output = gpt2_forward(jax_params, config, large_tokens)
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
    
    jax_output = gpt2_forward(jax_params, config, zero_tokens)
    with torch.no_grad():
        torch_output = torch_model(torch_zero_tokens).logits
    
    np.testing.assert_allclose(
        jax_output,
        convert_torch_to_jax(torch_output),
        rtol=RTOL,
        atol=ATOL
    )