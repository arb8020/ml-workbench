import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any
from training import AdamConfig, SGDConfig, MomentumConfig

RTOL = 1e-4  
ATOL = 1e-4  

def convert_torch_to_jax(tensor: torch.Tensor, transpose_weights: bool = False) -> jax.Array:
    """convert PyTorch tensor to JAX array"""
    array = jnp.array(tensor.detach().cpu().numpy())
    if transpose_weights:
        return array.T
    return array

def convert_jax_to_torch(array: jax.Array, transpose_weights: bool = False) -> torch.Tensor:
    """convert JAX array to PyTorch tensor."""
    if transpose_weights:
        array = array.T
    return torch.from_numpy(np.array(array))

def assert_equal_parameters(jax_params: Dict, torch_params: Dict, param_mapping: Dict, rtol: float = RTOL, atol: float = ATOL):
    """
    compare JAX and PyTorch parameters using provided mapping.
    """
    flat_jax_params = {
        '/'.join(str(x) for x in path): value 
        for path, value in jax.tree_util.tree_leaves_with_path(jax_params)
    }
    
    for jax_path, jax_value in flat_jax_params.items():
        if jax_path not in param_mapping:
            continue
            
        torch_name = param_mapping[jax_path]
        torch_param = torch_params[torch_name]
        
        needs_transpose = 'weight' in torch_name and 'embedding' not in torch_name
        
        np.testing.assert_allclose(
            jax_value,
            convert_torch_to_jax(torch_param, transpose_weights=needs_transpose),
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch in parameter {jax_path} (torch: {torch_name})"
        )

@pytest.fixture
def sample_input():
    """basic input sequence for testing."""
    return jnp.array([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])

@pytest.fixture
def optimizer_configs():
    """common optimizer configurations."""
    return {
        'adam': AdamConfig(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01),
        'sgd': SGDConfig(lr=0.01),
        'momentum': MomentumConfig(lr=0.01, beta=0.9)
    }

@pytest.fixture
def sample_causal_mask():
    """generate sample causal mask for testing."""
    seq_len = 8
    mask = jnp.triu(jnp.ones((seq_len, seq_len)) * float('-inf'), k=1)
    return mask

@pytest.fixture
def torch_device():
    """get appropriate torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')