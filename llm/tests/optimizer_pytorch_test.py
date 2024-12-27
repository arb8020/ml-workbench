import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any
from test_utils import convert_torch_to_jax, convert_jax_to_torch, RTOL, ATOL
from training import (
    init_optimizer, AdamConfig, SGDConfig, MomentumConfig,
    adamw_update, sgd_update, momentum_update
)

@pytest.fixture
def simple_params():
    """create simple parameter structure for testing optimizers."""
    return {
        'w1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b1': jnp.array([0.1, 0.2]),
        'w2': jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        'b2': jnp.array([0.3, 0.4])
    }

@pytest.fixture
def simple_torch_model(simple_params):
    """create PyTorch model with same parameters."""
    class SimpleModel(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.w1 = torch.nn.Parameter(convert_jax_to_torch(params['w1']))
            self.b1 = torch.nn.Parameter(convert_jax_to_torch(params['b1']))
            self.w2 = torch.nn.Parameter(convert_jax_to_torch(params['w2']))
            self.b2 = torch.nn.Parameter(convert_jax_to_torch(params['b2']))

    return SimpleModel(simple_params)

@pytest.fixture
def optimizer_configs():
    """common optimizer configurations."""
    return {
        'adam': AdamConfig(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01),
        'sgd': SGDConfig(lr=0.01),
        'momentum': MomentumConfig(lr=0.01, beta=0.9)
    }

def create_matching_gradients(params: Dict) -> Dict:
    """create gradients matching parameter structure."""
    return jax.tree_map(lambda x: jnp.ones_like(x), params)

def compare_parameters(new_jax_params: Dict[str, jnp.ndarray], torch_model: torch.nn.Module, 
                      rtol=RTOL, atol=ATOL, error_msg_prefix: str = "Mismatch"):
    torch_params_dict = dict(torch_model.named_parameters())
    
    for jax_name, jax_value in new_jax_params.items():
        if jax_name not in torch_params_dict:
            raise ValueError(f"Parameter '{jax_name}' not found in PyTorch model.")
        
        torch_param = torch_params_dict[jax_name]
        
        torch_param_converted = convert_torch_to_jax(torch_param)
        
        if jax_value.shape != torch_param_converted.shape:
            raise AssertionError(
                f"{error_msg_prefix} in '{jax_name}': "
                f"shape mismatch (JAX: {jax_value.shape}, PyTorch: {torch_param_converted.shape})"
            )
        
        np.testing.assert_allclose(
            jax_value,
            torch_param_converted,
            rtol=rtol,
            atol=atol,
            err_msg=f"{error_msg_prefix} in '{jax_name}'"
        )

def test_sgd_single_step(simple_params, simple_torch_model, optimizer_configs):
    """test single SGD update matches PyTorch."""
    torch_optimizer = torch.optim.SGD(
        simple_torch_model.parameters(),
        lr=optimizer_configs['sgd'].lr
    )
    
    jax_grads = create_matching_gradients(simple_params)
    torch_grads = {
        name: torch.ones_like(param)
        for name, param in simple_torch_model.named_parameters()
    }
    
    new_jax_params, _ = sgd_update(
        simple_params,
        jax_grads,
        {},
        optimizer_configs['sgd']
    )
    
    for param, grad in zip(simple_torch_model.parameters(), torch_grads.values()):
        param.grad = grad
    torch_optimizer.step()
    
    compare_parameters(new_jax_params, simple_torch_model, error_msg_prefix="Mismatch")

def test_momentum_single_step(simple_params, simple_torch_model, optimizer_configs):
    """test single momentum update matches PyTorch."""
    torch_optimizer = torch.optim.SGD(
        simple_torch_model.parameters(),
        lr=optimizer_configs['momentum'].lr,
        momentum=optimizer_configs['momentum'].beta
    )
    
    jax_opt_state, _, _ = init_optimizer(
        'momentum',
        simple_params,
        **optimizer_configs['momentum']._asdict()
    )
    
    jax_grads = create_matching_gradients(simple_params)
    torch_grads = {
        name: torch.ones_like(param)
        for name, param in simple_torch_model.named_parameters()
    }
    
    new_jax_params, new_jax_state = momentum_update(
        simple_params,
        jax_grads,
        jax_opt_state,
        optimizer_configs['momentum']
    )
    
    for param, grad in zip(simple_torch_model.parameters(), torch_grads.values()):
        param.grad = grad
    torch_optimizer.step()
    
    compare_parameters(new_jax_params, simple_torch_model, error_msg_prefix="Mismatch")

def test_adam_single_step(simple_params, simple_torch_model, optimizer_configs):
    """test single Adam update matches PyTorch."""
    torch_optimizer = torch.optim.AdamW(
        simple_torch_model.parameters(),
        lr=optimizer_configs['adam'].lr,
        betas=(optimizer_configs['adam'].beta1, optimizer_configs['adam'].beta2),
        eps=optimizer_configs['adam'].eps,
        weight_decay=optimizer_configs['adam'].weight_decay
    )
    
    jax_opt_state, _, _ = init_optimizer(
        'adam',
        simple_params,
        **optimizer_configs['adam']._asdict()
    )
    
    jax_grads = create_matching_gradients(simple_params)
    torch_grads = {
        name: torch.ones_like(param)
        for name, param in simple_torch_model.named_parameters()
    }
    
    new_jax_params, new_jax_state = adamw_update(
        simple_params,
        jax_grads,
        jax_opt_state,
        optimizer_configs['adam']
    )
    
    for param, grad in zip(simple_torch_model.parameters(), torch_grads.values()):
        param.grad = grad
    torch_optimizer.step()
    
    compare_parameters(new_jax_params, simple_torch_model, error_msg_prefix="Mismatch")

def test_adam_momentum_accumulation(simple_params, simple_torch_model, optimizer_configs):
    """test Adam momentum accumulation over multiple steps."""
    torch_optimizer = torch.optim.AdamW(
        simple_torch_model.parameters(),
        lr=optimizer_configs['adam'].lr,
        betas=(optimizer_configs['adam'].beta1, optimizer_configs['adam'].beta2),
        eps=optimizer_configs['adam'].eps,
        weight_decay=optimizer_configs['adam'].weight_decay
    )
    
    jax_opt_state, _, _ = init_optimizer(
        'adam',
        simple_params,
        **optimizer_configs['adam']._asdict()
    )
    
    jax_params = simple_params
    
    for i in range(5):
        scale = (i + 1.0)
        jax_grads = jax.tree_map(lambda x: jnp.ones_like(x) * scale, simple_params)
        torch_grads = {
            name: torch.ones_like(param) * scale
            for name, param in simple_torch_model.named_parameters()
        }
        
        jax_params, jax_opt_state = adamw_update(
            jax_params,
            jax_grads,
            jax_opt_state,
            optimizer_configs['adam']
        )
        
        for param, grad in zip(simple_torch_model.parameters(), torch_grads.values()):
            param.grad = grad
        torch_optimizer.step()
    
    compare_parameters(jax_params, simple_torch_model, error_msg_prefix="Mismatch after momentum accumulation")

def test_adam_weight_decay(simple_params, simple_torch_model, optimizer_configs):
    """test Adam weight decay behavior."""
    config = optimizer_configs['adam']._replace(weight_decay=0.1, lr=0.1)
    
    torch_optimizer = torch.optim.AdamW(
        simple_torch_model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    jax_opt_state, _, _ = init_optimizer('adam', simple_params, **config._asdict())
    jax_params = simple_params
    
    for _ in range(5):
        jax_grads = jax.tree_map(lambda x: jnp.zeros_like(x), simple_params)
        torch_grads = {
            name: torch.zeros_like(param)
            for name, param in simple_torch_model.named_parameters()
        }
        
        jax_params, jax_opt_state = adamw_update(jax_params, jax_grads, jax_opt_state, config)
        
        for param, grad in zip(simple_torch_model.parameters(), torch_grads.values()):
            param.grad = grad
        torch_optimizer.step()
    
    torch_params_dict = dict(simple_torch_model.named_parameters())
    
    for jax_name, jax_value in jax_params.items():
        if jax_name not in torch_params_dict:
            raise ValueError(f"Parameter '{jax_name}' not found in PyTorch model.")
        
        torch_param = torch_params_dict[jax_name]
        torch_param_converted = convert_torch_to_jax(torch_param)
        
        original_param = simple_params[jax_name]
        assert jnp.all(jnp.abs(jax_value) < jnp.abs(original_param)), \
            f"Weight decay not working for '{jax_name}'"
        
        if jax_value.shape != torch_param_converted.shape:
            raise AssertionError(
                f"Mismatch in '{jax_name}': shape mismatch (JAX: {jax_value.shape}, PyTorch: {torch_param_converted.shape})"
            )
        
        np.testing.assert_allclose(
            jax_value,
            torch_param_converted,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Mismatch in '{jax_name}' after weight decay"
        )

def test_optimizer_numerical_stability(simple_params, optimizer_configs):
    """test optimizers with extreme values."""
    large_grads = jax.tree_map(lambda x: jnp.ones_like(x) * 1e6, simple_params)
    
    small_grads = jax.tree_map(lambda x: jnp.ones_like(x) * 1e-6, simple_params)
    
    for opt_name, config in optimizer_configs.items():
        opt_state, update_fn, _ = init_optimizer(opt_name, simple_params, **config._asdict())
        
        try:
            new_params, _ = update_fn(simple_params, large_grads, opt_state, config)
            for param in jax.tree_util.tree_leaves(new_params):
                assert not jnp.any(jnp.isnan(param)), f"{opt_name} produced NaNs with large gradients"
                assert not jnp.any(jnp.isinf(param)), f"{opt_name} produced Infs with large gradients"
        except Exception as e:
            pytest.fail(f"Optimizer '{opt_name}' failed with large gradients: {e}")
        
        try:
            new_params, _ = update_fn(simple_params, small_grads, opt_state, config)
            for param in jax.tree_util.tree_leaves(new_params):
                assert not jnp.any(jnp.isnan(param)), f"{opt_name} produced NaNs with small gradients"
                assert not jnp.any(jnp.isinf(param)), f"{opt_name} produced Infs with small gradients"
        except Exception as e:
            pytest.fail(f"Optimizer '{opt_name}' failed with small gradients: {e}")

