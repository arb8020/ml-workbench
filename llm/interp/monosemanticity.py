import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, List, Any, Optional, Tuple, Union, Callable
from transformers import GPT2LMHeadModel
from functools import partial
import torch
import json

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

ModelConfig = Union[GPTConfig]


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


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """
    basic linear transform that maps inputs to outputs using learnable weights and biases
    """
    return jnp.dot(x, weight) + bias

def layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array, eps: float = 1e-5) -> jax.Array:
    """
    layer normalization stabilizes training by normalizing activations using learnable scale/shift parameters
    ba et al, 2016: https://arxiv.org/abs/1607.06450
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

def gelu(x: jax.Array) -> jax.Array:
    """
    Gaussian Error Linear Unit activation - provides smooth gradients and improves training compared to ReLU
    hendrycks/gimpel 2016: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))

def gelu_ffn(x: jax.Array, in_weight: jax.Array, in_bias: jax.Array, out_weight: jax.Array, out_bias: jax.Array) -> jax.Array:
    """
    feed-forward network with GELU used in GPT2
    """

    projected_up = gelu(linear(x, in_weight, in_bias)) # (seq_len, embed_dim) -> (seq_len, 4 * embed_dim)
    output = linear(projected_up, out_weight, out_bias) # (seq_len, 4 * embed_dim) -> (seq_len, embed_dim)
    return output

def create_causal_mask(seq_len: int, start_pos: int = 0):
    """
    creates triangular mask to ensure autoregressive property by preventing attention to future tokens
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    mask = jnp.zeros((seq_len, seq_len), dtype=jnp.float32)
    if seq_len > 1:
        mask = jnp.full((seq_len, seq_len), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seq_len, start_pos)), mask])
    return mask

class KVCache(NamedTuple):
    """
    grouped query attention, allows for a flexible amount of kv_heads (queries attend to groups of k/v tensors)
    finds a middle ground between multi-head attention performance and managing memory bandwidth of reloading k/v tensors in kv cached inference
    ainslie et al, 2023: https://arxiv.org/pdf/2305.13245
    """
    keys: jax.Array    # (n_layers, seq_len, n_head, head_dim)
    values: jax.Array  # (n_layers, seq_len, n_head, head_dim)

    @classmethod
    def init(cls, model_config: ModelConfig) -> 'KVCache':

        if model_config.n_kv_heads is not None:
            cache_heads = model_config.n_kv_heads
        else:
            cache_heads = model_config.n_heads
        shape = (model_config.n_layers, model_config.context_len, cache_heads, model_config.embedding_dim // model_config.n_heads)

        return cls(keys=jnp.zeros(shape), values=jnp.zeros(shape))

    def update(self, xk: jax.Array, xv: jax.Array, block_idx: int, cur_pos: int, n_rep: int = 1) -> Tuple[jax.Array, jax.Array, 'KVCache']:
        cached_keys = jax.lax.dynamic_update_slice(self.keys, xk[None, ...], (block_idx, cur_pos, 0, 0)) # add block dim
        cached_values = jax.lax.dynamic_update_slice(self.values, xv[None, ...],  (block_idx, cur_pos, 0, 0))

        keys = jnp.repeat(cached_keys[block_idx], n_rep, axis=1) # repeat across head dim for grouped query attention
        values = jnp.repeat(cached_values[block_idx], n_rep, axis=1)

        return keys, values, KVCache(keys=cached_keys, values=cached_values)

def grouped_query_attn(x: jax.Array, w_qkv: jax.Array, b_qkv: jax.Array, w_proj: jax.Array, b_proj: jax.Array, model_config: ModelConfig, causal_mask: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, Optional['KVCache']]:
    """
    standard attention that processes input sequence in parallel, enabling attention heads to focus on different aspects of the input
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """

    seq_len, embed_dim = x.shape

    head_dim = embed_dim // model_config.n_heads
    if model_config.n_kv_heads is not None:
        n_rep = model_config.n_heads // model_config.n_kv_heads
    else:
        n_rep = 1

    x_qkv = jnp.dot(x, w_qkv) + b_qkv
    x_qkv_heads = x_qkv.reshape(seq_len, 3, model_config.n_heads, head_dim)
    xq, xk, xv = x_qkv_heads[:, 0], x_qkv_heads[:, 1], x_qkv_heads[:, 2]

    if kv_cache is not None:
        xk, xv, kv_cache = kv_cache.update(xk, xv, block_idx, cur_pos, n_rep)
        xk = xk[:cur_pos + seq_len]
        xv = xv[:cur_pos + seq_len]
    else:
        xk = jnp.repeat(xk, n_rep, axis=1)
        xv = jnp.repeat(xv, n_rep, axis=1)

    xq = xq.transpose(1, 0, 2)
    xkt = xk.transpose(1, 2, 0)
    xv = xv.transpose(1, 0, 2)

    scaled_scores = jnp.matmul(xq, xkt)/ jnp.sqrt(head_dim)

    if cur_pos == 0:
        scaled_scores = scaled_scores + causal_mask

    mask_val = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
    mask = jnp.where(scaled_scores != 0.0, scaled_scores, mask_val)
    padded_logits = jnp.where((mask >= mask_val * 0.5), scaled_scores, mask_val)
    attn_weights = jax.nn.softmax(padded_logits, axis=-1)

    context_weights = jnp.matmul(attn_weights, xv)
    transposed_context_weights = context_weights.transpose(1, 0, 2)
    reshaped_context_weights = transposed_context_weights.reshape(seq_len, embed_dim)
    token_logits = jnp.dot(reshaped_context_weights, w_proj) + b_proj

    return token_logits, kv_cache

def gpt2_forward(model_params: Dict[str, Any], model_config: ModelConfig, tokens: jax.Array, causal_mask: jax.Array, cur_pos: int = 0, kv_cache: Optional[KVCache] = None, record_residual: bool = False) -> Tuple[jax.Array, Optional['KVCache'], Optional[List]]:
    """
    full GPT2 forward pass, uses transformer_block
    radford et al, 2019: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """
    seq_len = tokens.shape[0] # (seq_len,)
    token_embeds = model_params['token_embedding'][tokens] # (seq_len,) -> (seq_len, embed_dim)

    if record_residual:
        recorded_residual = []

    if kv_cache is not None:
        pos_embeds = model_params['positional_embedding'][cur_pos:cur_pos + seq_len]
    else:
        pos_embeds = model_params['positional_embedding'][:seq_len]

    x = token_embeds + pos_embeds

    for i in range(model_config.n_layers):
        x, kv_cache = gpt2_block(x, model_params[f'block_{i}'], model_config, causal_mask, cur_pos=cur_pos, block_idx=i, kv_cache=kv_cache) # (seq_len, embed_dim) -> (seq_len, embed_dim)
        if record_residual:
            recorded_residual.append(x)

    x = layer_norm(x, model_params['lnf']['gamma'], model_params['lnf']['beta'])
    token_logits = jnp.dot(x, model_params['output_projection']) # (seq_len, embed_dim) -> (seq_len, vocab_size)

    if not record_residual:
        return token_logits, kv_cache, None

    return token_logits, kv_cache, recorded_residual

def gpt2_block(x: jax.Array, block_params: Dict[str, Dict[str, jax.Array]], model_config: ModelConfig, causal_mask: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, Optional['KVCache']]:
    """
    full GPT2 transformer block, attention + FFN with residual connections and layer norms
    vaswani et al, 2017: https://arxiv.org/abs/1706.03762
    """
    residual = x
    normed_x = layer_norm(x, block_params['ln1']['gamma'], block_params['ln1']['beta'])

    context, kv_cache = grouped_query_attn(
        normed_x,
        block_params['attn_in']['weight'],
        block_params['attn_in']['bias'],
        block_params['attn_out']['weight'],
        block_params['attn_out']['bias'],
        model_config,
        causal_mask,
        cur_pos=cur_pos,
        block_idx=block_idx,
        kv_cache=kv_cache
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
    return enhanced_context, kv_cache

def collect_activations(model_params, model_config, dataset, record_residual=True):
    """collect residual stream activations from model forward passes"""
    activations = []

    for batch in dataset:
        causal_mask = create_causal_mask(batch.shape[0])
        _, _, residuals = gpt2_forward(
            model_params,
            model_config,
            batch,
            causal_mask,
            record_residual=record_residual
        )

        # Stack all layer activations
        batch_activations = jnp.stack(residuals)
        activations.append(batch_activations)

    return jnp.concatenate(activations, axis=1)

def init_autoencoder(key, input_dim: int, hidden_dim: int, ) -> Dict:
    """initializes autoencoder"""

    enc_key, dec_key = jax.random.split(key)

    autoencoder_params = {
        'w_enc': jax.random.normal(enc_key, (hidden_dim, input_dim)) * 0.01,
        'b_enc': jnp.zeros((hidden_dim,)),
        'w_dec': jax.random.normal(dec_key, (input_dim, hidden_dim)) * 0.01,
        'b_dec': jnp.zeros((input_dim,)),
    }

    return autoencoder_params

def autoencoder_fwd(x: jax.Array, params: Dict[str, jnp.ndarray], activation_fn: Callable = jax.nn.relu) -> Tuple[jax.Array, jax.Array]:
    """ autoencoder fwd pass """
    representation = activation_fn(jnp.dot(params['w_enc'], x) + params['b_enc'])
    reconstruction = jnp.dot(params['w_dec'], representation) + params['b_dec']

    return representation, reconstruction

def sae_metrics(sae_params, x: jax.Array):
    """ metrics for training sparse autoencoder """
    h_rep, x_recon = autoencoder_fwd(x, sae_params)

    # L2 reconstruction loss
    l2_loss = jnp.mean((x - x_recon) ** 2)

    # L1 sparsity loss
    l1_loss = jnp.mean(jnp.abs(h_rep))

    # L0 sparsity metric (average number of non-zero features)
    l0_metric = jnp.mean(jnp.sum(h_rep != 0, axis=-1))

    return l2_loss, l1_loss, l0_metric
