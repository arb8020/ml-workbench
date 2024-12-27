import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Callable, Any, Tuple, Optional, Union
from functools import partial
from core import ModelConfig, gpt2_forward, llama_forward, create_causal_mask, KVCache

# Tokenization

class Tokenizer(NamedTuple):
    encode: Callable
    decode: Callable

def count_token_pairs(tokens):
    token_pairs = {}
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        token_pairs[pair] = token_pairs.get(pair, 0) + 1
    return token_pairs

def merge_pair(tokens, top_pair, new_id):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == top_pair:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def encode(text, merges):
    tokens = list(text.encode('utf-8'))
    for pair, new_id in merges:
        tokens = merge_pair(tokens, pair, new_id)
    return tokens

def decode(encoded_tokens, id_to_token):
    return b''.join(id_to_token[token] for token in encoded_tokens).decode('utf-8', errors='replace')

def learn_bpe(text, target_vocab_size):
    tokens = list(text.encode('utf-8'))
    token_to_id = {i: bytes([i]) for i in range(256)}  
    id_to_token = {i: bytes([i]) for i in range(256)}  
    next_id = 256
    merges = []

    while len(token_to_id) < target_vocab_size:
        if len(token_to_id) % 100 == 0:
            print('vocab at: ', len(token_to_id))

        token_pairs = count_token_pairs(tokens)
        if not token_pairs:
            print("no more pairs to merge. current vocab size:", len(token_to_id))
            break

        top_pair = max(token_pairs, key=token_pairs.get)
        new_token = id_to_token[top_pair[0]] + id_to_token[top_pair[1]]
        token_to_id[new_token] = next_id
        id_to_token[next_id] = new_token
        merges.append((top_pair, next_id))

        tokens = merge_pair(tokens, top_pair, next_id)
        next_id += 1

        if len(token_to_id) >= target_vocab_size:
            break

    return token_to_id, id_to_token, merges

# Pretraining Data

class DatasetConfig(NamedTuple):
    name: str
    train_split: str  
    val_split: str
    test_split: str
    text_column: str

def load_text_dataset() -> Dict[str,str]:
    shakespeare_url = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
    shakespeare_text = resp.text[10460:]

    # Determine split sizes
    train_size = int(0.8 * len(shakespeare_text))
    val_size = int(0.1 * len(shakespeare_text))

    # Create splits
    train_text = shakespeare_text[:train_size]
    validation_text = shakespeare_text[train_size:train_size + val_size]
    test_text = shakespeare_text[train_size + val_size:]

    # Return as a dictionary
    return {
        'train': train_text,
        'validation': validation_text,
        'test': test_text
    }

    return shakespeare_text

def create_random_batches(tokens: jax.Array, batch_size: int, seq_len: int, key):
    """
    simply takes random slices of training token array and returns batches
    """

    max_start_idx = tokens.shape[0] - seq_len - 1
    start_indices = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=max_start_idx
    )

    def get_sequence(start_idx):
        return jax.lax.dynamic_slice(tokens, (start_idx,), (seq_len + 1,))

    sequences = jax.vmap(get_sequence)(start_indices)

    input_batches, target_batches = sequences[:, :-1], sequences[:, 1:]

    return input_batches, target_batches

def prepare_dataset(text: str, tokenizer: Any) -> jax.Array:
    """
    encodes text using specified tokenizer
    """

    return jnp.array(tokenizer.encode(text), dtype=jnp.int32)

# Loss 

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    standard cross entropy loss for predicted vs actual distribution
    """

    num_classes = logits.shape[-1]
    onehot_targets = jax.nn.one_hot(targets, num_classes)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(onehot_targets * log_probs, axis=-1)

    mean_loss = jnp.mean(loss)

    return mean_loss

# Optimizers

class SGDConfig(NamedTuple):
    lr: float

class MomentumConfig(NamedTuple):
    lr: float
    beta: float

class AdamConfig(NamedTuple):
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

OptConfig = Union[SGDConfig, MomentumConfig, AdamConfig]
OptimizerState = Dict

def init_sgd_state(params: Dict, opt_config: SGDConfig) -> OptimizerState:
    """
    initializes optimizer state for stochastic gradient descent
    robbins/monro: doi:10.1214/AOMS/1177729586
    """ 

    return {} # sgd doesn't really have a state

def init_sgdm_state(params: Dict, config: MomentumConfig) -> OptimizerState:
    """
    initializes optimizer state for stochastic gradient descent with momentum
    rumelhart et al, 1986: doi:10.1038/323533a0
    """

    return {
        'momentum': jax.tree_map(jnp.zeros_like, params)
    }

def init_adam_state(params: Dict, opt_config: AdamConfig) -> OptimizerState:
    """
    initializes optimizer state struct for adam
    kingma/ba et al, 2014: https://arxiv.org/abs/1412.6980
    """

    gradient_mean = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    gradient_squared_mean = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    return {
        'gradient_mean': jax.tree_map(jnp.zeros_like, params),
        'gradient_squared_mean': jax.tree_map(jnp.zeros_like, params),
        'step': 0
    }

def init_optimizer(name: str, params: Dict, **opt_kwargs) -> Tuple[Dict, Callable, NamedTuple]:
    """
    initialize optimizer state and return update function with config
    """

    init_fn, update_fn, config_cls = OPTIMIZERS[name]
    config = config_cls(**opt_kwargs)
    state = init_fn(params, config)
    return state, update_fn, config

def sgd_update(params: Dict, grads: Dict, state: OptimizerState, config: SGDConfig) -> Tuple[Dict, OptimizerState]:
    """
    update step for stochastic gradient descent
    robbins/monro: doi:10.1214/AOMS/1177729586
    """

    param_updates = jax.tree_map(
        lambda p, g: -config.lr * g,
        params, 
        grads
    )
    
    new_params = jax.tree_map(
        lambda p, u: p + u,
        params,
        param_updates
    )
    
    return new_params, state

def momentum_update(params: Dict, grads: Dict, state: OptimizerState, config: MomentumConfig) -> Tuple[Dict, OptimizerState]:
    """
    update step for stochastic gradient descent with momentum
    rumelhart et al, 1986: doi:10.1038/323533a0
    """

    new_momentum = jax.tree_map(
        lambda m, g: config.beta * m + g,
        state['momentum'],
        grads
    )
    
    param_updates = jax.tree_map(
        lambda m: -config.lr * m,
        new_momentum
    )
    
    new_params = jax.tree_map(
        lambda p, u: p + u,
        params,
        param_updates
    )
    
    return new_params, {'momentum': new_momentum}

def adamw_update(params: Dict, grads: Dict, opt_state: OptimizerState, opt_config: AdamConfig) -> Tuple[Dict, OptimizerState]:
    """
    updates adam, supports weight decay
    loschilov/hutter, 2017: https://arxiv.org/abs/1711.05101
    """

    current_step = opt_state['step'] + 1

    gradient_mean_biased = jax.tree.map(
        lambda prev_mean, gradient: opt_config.beta1 * prev_mean + (1 - opt_config.beta1) * gradient,
        opt_state['gradient_mean'],
        grads
    )

    gradient_squared_mean_biased = jax.tree.map(
        lambda prev_squared_mean, gradient: opt_config.beta2 * prev_squared_mean + (1 - opt_config.beta2) * gradient**2,
        opt_state['gradient_squared_mean'],
        grads
    )

    gradient_mean_corrected = jax.tree.map(
        lambda mean: mean / (1 - opt_config.beta1**current_step),
        gradient_mean_biased
    )

    gradient_squared_mean_corrected = jax.tree.map(
        lambda squared_mean: squared_mean / (1 - opt_config.beta2**current_step),
        gradient_squared_mean_biased
    )

    param_updates = jax.tree.map(
        lambda mean, squared_mean, param: -opt_config.lr * (mean / (jnp.sqrt(squared_mean) + opt_config.eps) + opt_config.weight_decay * param),
        gradient_mean_corrected,
        gradient_squared_mean_corrected,
        params
    )

    updated_params = jax.tree.map(
        lambda param, update: param + update, params, param_updates)

    updated_opt_state = {
        'gradient_mean': gradient_mean_biased,
        'gradient_squared_mean': gradient_squared_mean_biased,
        'step': current_step
    }

    return updated_params, updated_opt_state

OPTIMIZERS = {
    'sgd': (init_sgd_state, sgd_update, SGDConfig),
    'momentum': (init_sgdm_state, momentum_update, MomentumConfig),
    'adam': (init_adam_state, adamw_update, AdamConfig)
}

# Training 

class TrainConfig(NamedTuple):
    num_epochs: int
    batches_per_epoch: int 
    batch_size: int
    batch_seq_len: int
    seed: int
    
def batch_forward(params: Dict, model_config: Dict, model_name: str, input_batch: jnp.ndarray, model_kwargs: Dict = {},) -> jnp.ndarray:
    """
    automatically vectorizes a specified model forward pass
    """ 
    causal_mask = create_causal_mask(input_batch.shape[1])
    return jax.vmap(lambda x: model_dict[model_name](params, model_config, x, causal_mask=causal_mask, **model_kwargs))(input_batch)

@partial(jax.jit, static_argnames=['train_config', 'model_config', 'model_name', 'opt_name'])
def train_step(params: Dict, model_config: ModelConfig, model_name: str, tokens: jax.Array, opt_state: OptimizerState, opt_config: OptConfig, opt_name: str, train_config: TrainConfig, key: jax.random.PRNGKey, model_kwargs: Dict = {}) -> Tuple[Dict, OptimizerState, float]:
    """
    jit-compiled training step - creates batches + computes gradient + optimizer update
    rumelhart/hinton/willians, 1986: https://www.nature.com/articles/323533a0
    """

    def batch_loss_fn(params, input_batch, target_batch):
        logits, _ = batch_forward(params, model_config, model_name, input_batch, model_kwargs)
        batch_loss = cross_entropy_loss(logits, target_batch)
        return batch_loss

    batch_size = train_config.batch_size
    seq_len = train_config.batch_seq_len
    input_batch, target_batch = create_random_batches(tokens, batch_size, seq_len, key)
    loss, grads = jax.value_and_grad(batch_loss_fn)(params, input_batch, target_batch)

    _, opt_update_fn, _ = OPTIMIZERS[opt_name]
    new_params, new_opt_state = opt_update_fn(params, grads, opt_state, opt_config)

    return new_params, new_opt_state, loss

def train(params: Dict, model_config: ModelConfig, model_name: str, tokens: jnp.ndarray, opt_config: OptConfig, opt_name: str, train_config: TrainConfig, key=jax.random.PRNGKey(0), model_kwargs: Dict = {}):
    """
    training loop
    """

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")

    device = jax.devices()[0]
    print(f"Using device: {device}")

    key = jax.random.PRNGKey(train_config.seed)

    opt_init_fn, _, config_cls = OPTIMIZERS[opt_name]
    init_opt_state = opt_init_fn(params, opt_config)

    params = jax.device_put(params, device)
    tokens = jax.device_put(tokens, device)
    opt_state = jax.device_put(init_opt_state, device)

    print('training ...')
    for epoch in range(train_config.num_epochs):
        epoch_loss = 0.0
        for _ in range(train_config.batches_per_epoch):
            key, subkey = jax.random.split(key, 2)
            new_params, opt_state, loss = train_step(
                params, model_config, model_name, tokens, opt_state, opt_config, opt_name, train_config, key=subkey, model_kwargs=model_kwargs,
            )
            epoch_loss += loss

            params = new_params
        avg_loss = epoch_loss / train_config.batches_per_epoch
        print(f"Epoch {epoch + 1}/{train_config.num_epochs}, Average Loss: {avg_loss}")

    return params

# Evaluation

class EvalConfig(NamedTuple):
    seq_len: int = 1024
    stride: int = 512
         
@partial(jax.jit, static_argnames=['model_config', 'model_name', 'eval_config'])
def sliding_ppl_eval_step(params: Dict, model_config: ModelConfig, model_name: str, tokens: jax.Array, start_idx: int, eval_config: EvalConfig, model_kwargs: Dict = {}) -> float:
    """
    jit-compiled perplexity evaluation step
    """
    input_seq = jax.lax.dynamic_slice(tokens, (start_idx,), (eval_config.seq_len,))
    target_seq = jax.lax.dynamic_slice(tokens, (start_idx + 1,), (eval_config.seq_len,))
    causal_mask = create_causal_mask(eval_config.seq_len)
    
    logits, _ = model_dict[model_name](params, model_config, input_seq, causal_mask, **model_kwargs)
    loss = cross_entropy_loss(logits, target_seq)
    return loss

    return perplexity

def eval_perplexity(params: Dict, model_config: ModelConfig, model_name: str, tokens: jnp.ndarray, eval_config: EvalConfig = EvalConfig(), model_kwargs: Dict = {}):
    """
    evaluates model perplexity with sliding window
    huggingface: https://github.com/huggingface/transformers/blob/main/docs/source/en/perplexity.md
    """
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")
        
    device = jax.devices()[0]
    params = jax.device_put(params, device)
    tokens = jax.device_put(tokens, device)
    
    print('evaluating perplexity...')
    total_loss = 0.0
    total_tokens = 0
    
    for start_idx in range(0, len(tokens) - eval_config.seq_len, eval_config.stride):
        loss = sliding_ppl_eval_step(params, model_config, model_name, tokens, start_idx, eval_config, model_kwargs)
        total_loss += loss * eval_config.seq_len 
        total_tokens += eval_config.seq_len
        
    avg_loss = total_loss / total_tokens
    perplexity = jnp.exp(avg_loss)
    print(f'perplexity: {perplexity}')
    
    return perplexity

model_dict = {'gpt2': gpt2_forward, 'llama': llama_forward}

# Inference

class SamplerConfig(NamedTuple):
    temp: float = 1.0
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_tokens: int = 100
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)

def sample_token(logits: jax.Array, sampler_config: SamplerConfig) -> jax.Array:
    """
    samples token from a distribution, includes filtering parameters like temperature/top_k for controlling generation
    radford et al, 2019: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    nguyen et al, 2024: https://arxiv.org/abs/2407.01082 (min_p)
    holtzman et al, 2019: https://arxiv.org/abs/1904.09751 (top_p)
    """
        
    scaled_logits = logits / sampler_config.temp
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    
    if sampler_config.min_p:
        max_prob = jnp.max(probs)
        min_p_threshold = sampler_config.min_p * max_prob
        scaled_logits = jnp.where(probs < min_p_threshold, float('-inf'), scaled_logits)
        return jax.random.categorical(sampler_config.key, scaled_logits)
        
    elif sampler_config.top_p:
        sorted_indices = jnp.argsort(probs, axis=-1)[::-1]
        cumsum = jnp.cumsum(probs[sorted_indices])
        sorted_mask = cumsum <= sampler_config.top_p
        mask = jnp.zeros_like(probs).at[sorted_indices].set(sorted_mask)
        scaled_logits = jnp.where(mask, scaled_logits, float('-inf'))
        return jax.random.categorical(sampler_config.key, scaled_logits)
        
    elif sampler_config.top_k:
        top_k = jnp.minimum(sampler_config.top_k, probs.shape[-1])
        top_k_probs = jnp.sort(probs, axis=-1)[-top_k:]
        threshold = top_k_probs[0]
        scaled_logits = jnp.where(probs < threshold, float('-inf'), scaled_logits)
        return jax.random.categorical(sampler_config.key, scaled_logits)
        
    return jax.random.categorical(sampler_config.key, scaled_logits)

def generate(params: Dict[str, Any], model_config: ModelConfig, model_name, tokens: jax.Array, sampler_config: SamplerConfig, model_kwargs: Dict = {}, stream: bool = True, tokenizer: Tokenizer = None) -> jax.Array:
    """
    generates text with kv caching
    radford et al, 2019: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """
    
    gen_tokens = jnp.array([], dtype=jnp.int32)
    cur_pos = 0
    seq_len = tokens.shape[0]

    kv_cache = KVCache.init(model_config)
    causal_mask = create_causal_mask(seq_len, cur_pos)

    logits, kv_cache = model_dict[model_name](params, model_config, tokens, causal_mask, cur_pos=cur_pos, kv_cache=kv_cache, **model_kwargs)

    next_token = sample_token(logits[-1:], sampler_config)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))
    tokens = jnp.concatenate((tokens, next_token))

    cur_pos = seq_len

    while cur_pos < sampler_config.max_tokens:
        logits, kv_cache = model_dict[model_name](params, model_config, next_token, causal_mask=causal_mask, cur_pos=cur_pos, kv_cache=kv_cache, **model_kwargs)
        
        next_token = sample_token(logits[-1:], sampler_config)
        if stream and tokenizer:
            print(tokenizer.decode(next_token), end='', flush=True)
            
        gen_tokens = jnp.concatenate((gen_tokens, next_token))
        tokens = jnp.concatenate((tokens, next_token))
        cur_pos += 1

    if stream:
        print()
    
    print(f'generated: {len(gen_tokens)} tokens')
    return gen_tokens
