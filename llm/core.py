import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Optional, Tuple, Union
from transformers import GPT2LMHeadModel
from functools import partial
from models import ModelConfig

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

def rms_norm(x: jax.Array, gamma: jax.Array, eps: float = 1e-5) -> jax.Array:
    """
    zhang/sennrich 2019: https://arxiv.org/abs/1910.07467
    re-centering invariance in layernorm not necessary, just uses root mean square to regularize a layer, computationally cheaper, also one less param to learn
    """
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(mean_square + eps) # rsqrt -> 1/jnp.sqrt
    return gamma * x_norm

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

def swiglu(x: jax.Array, gate_proj: jax.Array, up_proj: jax.Array, down_proj: jax.Array) -> jax.Array:
    """
    SwiGLU activation that combines swish gating with linear units to improve transformer performance over GELU
    dauphin et al, 2016: https://arxiv.org/pdf/1612.08083
    shazeer 2020: https://arxiv.org/pdf/2002.05202
    """

    gate = jax.nn.silu(jnp.dot(x, gate_proj)) # (seq_len, embed_dim) -> (seq_len, hidden_dim)
    projected_up = jnp.dot(x, up_proj) # (seq_len, embed_dim) -> (seq_len, hidden_dim)
    gated_output = gate * projected_up # (seq_len, hidden_dim) -> (seq_len, hidden_dim) (element-wise multiplication)
    back_down = jnp.dot(gated_output, down_proj) # (seq_len, hidden_dim) -> (seq_len, embed_dim)

    return back_down

# Positional Encoding

def apply_rotary_emb(xq: jax.Array, xk: jax.Array, rotation_matrices: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    applies rotary position embeddings to queries/keys by rotating vector pairs to encode relative positions while preserving token similarity
    works by splitting vectors into real/imaginary and using theta for position encoding and semantic meaning in magnitude
    think abt polar form trickery
    su et al, 2021: https://arxiv.org/abs/2104.09864
    """
    seq_len = xq.shape[0]
    rotation_matrices = rotation_matrices[:seq_len]
    # split into real/imaginary pairs
    reshape_xq = xq.reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.reshape(*xk.shape[:-1], -1, 2)
    # convert pairs to z = a + bi
    complex_xq = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    complex_xk = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # apply rotation angle matrix
    rotated_xq = complex_xq * rotation_matrices[:, None, :]
    rotated_xk = complex_xk * rotation_matrices[:, None, :]
    # convert back to real number, concatenate real/imaginary part
    xq_out = jnp.stack((jnp.real(rotated_xq), jnp.imag(rotated_xq)), axis=-1).reshape(*rotated_xq.shape[:-1], -1)
    xk_out = jnp.stack((jnp.real(rotated_xk), jnp.imag(rotated_xk)), axis=-1).reshape(*rotated_xk.shape[:-1], -1)
    return xq_out, xk_out

    return jax.vmap(adjust_frequency)(raw_frequencies)

def initialize_rotation_factors(dim: int, seq_len: int, theta: float = 500000.0, use_ntk_scaling: bool = True):
    """
    precomputes complex rotation factors for RoPE to avoid redundant computation during attention
    captures range of patterns by using diff frequencies depending on the embedding dimension's index
    su et al, 2021: https://arxiv.org/abs/2104.09864
    """
    # early dim -> high freq -> rotates fast -> local patterns
    # later dim -> low freq -> rotates slow -> global patterns
    frequencies = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))

    # NTK aware scaling
    if use_ntk_scaling:
        frequencies = apply_ntk_scaling(frequencies)

    # multiply each pos by each frequency
    positions = jnp.arange(seq_len)
    frequencies = jnp.outer(positions, frequencies)

    # convert to complex rotation factors with euler's formula
    return jnp.exp(1j * frequencies) # 1j is the imaginary unit

def apply_ntk_scaling(frequencies: jax.Array, scaling_factor: int = 8, low_freq_factor: int = 1, high_freq_factor: int = 4, original_context: int = 4096):
  """
  allows for extended context length from pretrained length by scaling base frequencies used to initialize rotation matrices in RoPE
  applies neural tangent kernel theory to the rotary positional embedding to nonlinearly scale RoPE's base
  bloc97, 2023: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
  """

  low_freq_wavelen = original_context / low_freq_factor
  high_freq_wavelen = original_context / high_freq_factor

  def scale_frequencies(freq):
    wavelen = 2 * jnp.pi / freq

    def interpolate_scaling(_):
      smooth = (original_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
      return (1 - smooth) * freq / scaling_factor + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / scaling_factor, interpolate_scaling, None),
      None
    )

  return jax.vmap(scale_frequencies)(frequencies)

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

# Attention

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

def grouped_query_attn(x: jax.Array, w_qkv: jax.Array, b_qkv: jax.Array, w_proj: jax.Array, b_proj: jax.Array, model_config: ModelConfig, causal_mask: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, 'KVCache']:
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

def grouped_query_attn_llama(x: jax.Array, wq: jax.Array, wk: jax.Array, wv: jax.Array, wo: jax.Array, model_config: ModelConfig, causal_mask: jax.Array, rotation_matrices: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, 'KVCache']:
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

    xq = jnp.dot(x,wq)
    xk = jnp.dot(x,wk)
    xv = jnp.dot(x,wv)

    kv_head = model_config.n_kv_heads if model_config.n_kv_heads else model_config.n_heads

    xq = xq.reshape(seq_len, model_config.n_heads, head_dim)
    xk = xk.reshape(seq_len, kv_head, head_dim)
    xv = xv.reshape(seq_len, kv_head, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, rotation_matrices)

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
    token_logits = jnp.dot(reshaped_context_weights, wo)

    return token_logits, kv_cache

# Forward Pass

def llama_forward(model_params: Dict[str, Any], model_config: ModelConfig, tokens: jax.Array, causal_mask: jax.Array, cur_pos: int = 0, kv_cache: Optional[KVCache] = None, rotation_factors: jax.Array = None) -> Tuple[jax.Array, 'KVCache']:

    # seq_len = tokens.shape[0] # (seq_len,)
    token_embeds = model_params['tok_embeddings'][tokens] # (seq_len,) -> (seq_len, embed_dim)
    x = token_embeds

    for i in range(model_config.n_layers):
        x, kv_cache = llama_block(x, model_params[f'layer_{i}'], model_config, causal_mask, cur_pos=cur_pos, rotation_matrices=rotation_factors, block_idx=i, kv_cache=kv_cache) # (seq_len, embed_dim) -> (seq_len, embed_dim)

    x = rms_norm(x, model_params['norm']['weight'])
    token_logits = jnp.dot(x, model_params['output'].T) # (seq_len, embed_dim) -> (seq_len, vocab_size)
    return token_logits, kv_cache

def llama_block(x: jax.Array, block_params: Dict[str, Dict[str, jax.Array]], model_config: ModelConfig, causal_mask: jax.Array, rotation_matrices: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, 'KVCache']:

    residual = x
    normed_x = rms_norm(x, block_params['attention_norm']['weight'])

    context, kv_cache = grouped_query_attn_llama( # llama params defined differently, ok to make a new function??
        normed_x,
        block_params['attention']['wq'],
        block_params['attention']['wk'],
        block_params['attention']['wv'],
        block_params['attention']['wo'],
        model_config,
        causal_mask,
        rotation_matrices=rotation_matrices,
        cur_pos=cur_pos,
        block_idx=block_idx,
        kv_cache=kv_cache
    )

    context = context + residual
    context_residual = context
    normed_context = rms_norm(context, block_params['ffn_norm']['weight'])

    enhanced_context = swiglu(normed_context, block_params['feed_forward']['w1'], block_params['feed_forward']['w3'], block_params['feed_forward']['w2'])
    enhanced_context = enhanced_context + context_residual
    return enhanced_context, kv_cache

def gpt2_forward(model_params: Dict[str, Any], model_config: ModelConfig, tokens: jax.Array, causal_mask: jax.Array, cur_pos: int = 0, kv_cache: Optional[KVCache] = None) -> Tuple[jax.Array, 'KVCache']:
    """
    full GPT2 forward pass, uses transformer_block
    radford et al, 2019: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """
    seq_len = tokens.shape[0] # (seq_len,)
    token_embeds = model_params['token_embedding'][tokens] # (seq_len,) -> (seq_len, embed_dim)

    if kv_cache is not None:
        pos_embeds = model_params['positional_embedding'][cur_pos:cur_pos + seq_len]
    else:
        pos_embeds = model_params['positional_embedding'][:seq_len]

    x = token_embeds + pos_embeds

    for i in range(model_config.n_layers):
        x, kv_cache = gpt2_block(x, model_params[f'block_{i}'], model_config, causal_mask, cur_pos=cur_pos, block_idx=i, kv_cache=kv_cache) # (seq_len, embed_dim) -> (seq_len, embed_dim)

    x = layer_norm(x, model_params['lnf']['gamma'], model_params['lnf']['beta'])
    token_logits = jnp.dot(x, model_params['output_projection']) # (seq_len, embed_dim) -> (seq_len, vocab_size)
    return token_logits, kv_cache

def gpt2_block(x: jax.Array, block_params: Dict[str, Dict[str, jax.Array]], model_config: ModelConfig, causal_mask: jax.Array, cur_pos: int = 0, block_idx: int = 0, kv_cache: Optional[KVCache] = None) -> jax.Array:
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
