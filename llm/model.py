import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Optional, Tuple, Union
from transformers import GPT2LMHeadModel
from functools import partial

# Activations

def relu(x: jax.Array) -> jax.Array:
    """
    rectified linear unit activation function - max(0,x)
    householder 1941: https://doi.org/10.1007/BF02478220
    """ 
    return jax.nn.relu(x)

def gelu(x: jax.Array) -> jax.Array:
    """
    Gaussian Error Linear Unit activation - provides smooth gradients and improves training compared to ReLU
    hendrycks/gimpel 2016: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))

def gelu_exact(x: jax.Array) -> jax.Array:
    """
    Gaussian Error Linear Unit activation - provides smooth gradients and improves training compared to ReLU
    exact implementation matches pytorch more closely than above approximation
    hendrycks/gimpel 2016: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

# Normalization

def layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array, eps: float = 1e-5) -> jax.Array:
    """
    layer normalization stabilizes training by normalizing activations using learnable scale/shift parameters
    ba et al, 2016: https://arxiv.org/abs/1607.06450
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

# Layers

def linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """
    basic linear transform that maps inputs to outputs using learnable weights and biases
    """
    return jnp.dot(x, weight) + bias

def gelu_ffn(x: jax.Array, in_weight: jax.Array, in_bias: jax.Array, out_weight: jax.Array, out_bias: jax.Array) -> jax.Array:
    """
    feed-forward network with GELU used in GPT2
    """

    projected_up = gelu(linear(x, in_weight, in_bias)) # (seq_len, embed_dim) -> (seq_len, 4 * embed_dim)
    output = linear(projected_up, out_weight, out_bias) # (seq_len, 4 * embed_dim) -> (seq_len, embed_dim)
    return output

# Attention

def create_causal_mask(seq_len: int):
    """
    creates triangular mask to ensure autoregressive property by preventing attention to future tokens
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))

def multi_head_attn(x: jax.Array, w_qkv: jax.Array, b_qkv: jax.Array, w_proj: jax.Array, b_proj: jax.Array, n_head: int, causal_mask: jax.Array) -> Tuple[jax.Array, Optional[Dict[str, jax.Array]]]:
    """
    standard multi-head attention that processes input sequence in parallel, enabling attention heads to focus on different aspects of the input
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    seq_len, embed_dim = x.shape # (seq_len, embed_dim)
    head_dim = embed_dim // n_head

    x_qkv = jnp.dot(x, w_qkv) + b_qkv # (seq_len, embed_dim) @ (embed_dim, 3 * embed_dim) -> (seq_len, 3*embed_dim) 
    x_qkv_heads = x_qkv.reshape(seq_len, 3, n_head, head_dim) # (seq_len, 3 * embed_dim) -> (seq_len, 3, n_head, head_dim) 
    xq, xk, xv = x_qkv_heads[:, 0], x_qkv_heads[:, 1], x_qkv_heads[:, 2] # (seq_len, 3, n_head, head_dim) -> (seq_len, n_head, head_dim)

    xq = xq.transpose(1, 0, 2) # (n_head, seq_len, head_dim) 
    xkt = xk.transpose(1, 2, 0) # (n_head, head_dim, seq_len)
    xv = xv.transpose(1, 0, 2) # (n_head, seq_len, head_dim)

    scaled_scores = jnp.matmul(xq, xkt)/ jnp.sqrt(head_dim) # (n_head, seq_len, head_dim) @ (n_head, head_dim, seq_len) -> (n_head, seq_len, seq_len)
    masked_scores = jnp.where(causal_mask == 0, float('-inf'), scaled_scores) # (n_head, seq_len, seq_len)
    attn_weights = jax.nn.softmax(masked_scores, axis=-1)

    context_weights = jnp.matmul(attn_weights, xv) # (n_head, seq_len, seq_len) @ (n_head, seq_len, head_dim) -> (n_head, seq_len, head_dim)
    transposed_context_weights = context_weights.transpose(1, 0, 2) # (n_head, seq_len, head_dim) -> (seq_len, n_head, head_dim)
    reshaped_context_weights = transposed_context_weights.reshape(seq_len, embed_dim) # (seq_len, n_head, head_dim) -> (seq_len, embed_dim)
    token_logits = jnp.dot(reshaped_context_weights, w_proj) + b_proj # (seq_len, embed_dim) @ (embed_dim, embed_dim) -> (seq_len, embed_dim)

    return token_logits

# Model Parameters

class ModelConfig(NamedTuple):
    """
    defines model architecture parameters for easy initialization and modification of model variants
    """
    vocab_size: int
    embedding_dim: int
    context_len: int
    n_head: int
    n_blocks: int

def init_gpt2_params(key, model_config: ModelConfig, scaling_factor: float = 0.02) -> Dict[str, Any]:
    """
    template for initializing parameters of a GPT2 style transformer model
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    keys = jax.random.split(key, 7 + 10 * model_config.n_blocks)

    params = {
        'token_embedding': jax.random.normal(keys[0], (model_config.vocab_size, model_config.embedding_dim)) * scaling_factor,
        'positional_embedding': jax.random.normal(keys[1], (model_config.context_len, model_config.embedding_dim)) * scaling_factor,
        'output_projection': jax.random.normal(keys[2], (model_config.embedding_dim, model_config.vocab_size)) * scaling_factor,
        'lnf': {
            'gamma': jnp.ones((model_config.embedding_dim,)),
            'beta': jnp.zeros((model_config.embedding_dim,)),
        },
    }

    for i in range(model_config.n_blocks):
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
    """

    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config

    
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.n_embd,
        context_len=config.n_positions,
        n_head=config.n_head,
        n_blocks=config.n_layer
    )

    converted_params = {}
    converted_params['token_embedding'] = jnp.array(model.transformer.wte.weight.detach().numpy())
    converted_params['positional_embedding'] = jnp.array(model.transformer.wpe.weight.detach().numpy())
    converted_params['output_projection'] = jnp.array(model.lm_head.weight.detach().numpy().T)
    converted_params['lnf'] = {
        'gamma': jnp.array(model.transformer.ln_f.weight.detach().numpy()),
        'beta': jnp.array(model.transformer.ln_f.bias.detach().numpy())
    }

    for i in range(model_config.n_blocks):
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

# Forward Pass

def gpt2_forward(model_params: Dict[str, Any], model_config: ModelConfig, tokens: jax.Array) -> Tuple[jax.Array, Optional[Dict[str, jax.Array]]]:
    """
    full GPT2 forward pass, uses transformer_block
    """
    seq_len = tokens.shape[0] # (seq_len,)
    token_embeds = model_params['token_embedding'][tokens] # (seq_len,) -> (seq_len, embed_dim)
    pos_embeds = model_params['positional_embedding'][:seq_len]
    x = token_embeds + pos_embeds 

    causal_mask = create_causal_mask(seq_len)

    for i in range(model_config.n_blocks):
        x = transformer_block(x, model_params[f'block_{i}'], model_config, causal_mask) # (seq_len, embed_dim) -> (seq_len, embed_dim)

    x = layer_norm(x, model_params['lnf']['gamma'], model_params['lnf']['beta'])
    token_logits = jnp.dot(x, model_params['output_projection']) # (seq_len, embed_dim) -> (seq_len, vocab_size)
    return token_logits


def transformer_block(x: jax.Array, block_params: Dict[str, Dict[str, jax.Array]], model_config: ModelConfig, causal_mask: jax.Array) -> jax.Array:
    """
    full GPT2 transformer block, attention + FFN with residual connections and layer norms
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    residual = x
    normed_x = layer_norm(x, block_params['ln1']['gamma'], block_params['ln1']['beta'])

    context = multi_head_attn(
        normed_x,
        block_params['attn_in']['weight'],
        block_params['attn_in']['bias'],
        block_params['attn_out']['weight'],
        block_params['attn_out']['bias'],
        model_config.n_head,
        causal_mask
    )

    context = context + residual
    context_residual = context
    normed_context = layer_norm(context, block_params['ln2']['gamma'], block_params['ln2']['beta'])

    enhanced_context = gelu_ffn(
        normed_context,
        block_params['ffn_in']['weight'],
        block_params['ffn_in']['bias'],
        block_params['ffn_out']['weight'],
        block_params['ffn_out']['bias']
    )

    enhanced_context = enhanced_context + context_residual
    return enhanced_context
