import pytest
import jax
import jax.numpy as jnp
from models import GPTConfig, init_gpt2_params
from core import KVCache, create_causal_mask
from training import (
    AdamConfig, TrainConfig, EvalConfig, SamplerConfig,
    generate, eval_perplexity, train,
    prepare_dataset, create_random_batches,
    Tokenizer, encode, decode, learn_bpe
)

@pytest.fixture
def small_gpt2_config():
    """small GPT2 config for testing."""
    return GPTConfig(
        vocab_size=100,
        embedding_dim=32,
        context_len=16,
        n_heads=2,
        n_layers=2
    )

@pytest.fixture
def training_config():
    """basic training configuration."""
    return TrainConfig(
        num_epochs=2,
        batches_per_epoch=5,
        batch_size=4,
        batch_seq_len=8,
        seed=42
    )

@pytest.fixture
def opt_config():
    """adam optimizer configuration."""
    return AdamConfig(
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01
    )

@pytest.fixture
def sample_text():
    """sample text for testing tokenization and training."""
    return """The Mercedes-Benz CLR GTR was a racing version of the CLR, competing in the FIA GT Championship and 24 Hours of Le Mans. Here are some key features and facts about the Mercedes-Benz CLR GTR:
-_Engine_: 5.7L V8 engine, producing 600 horsepower
_Transmission_: 6-speed sequential manual
- _Drive_: Rear-wheel drive
-_Body style_: 2-door coupé
-_Length/Width/Height: 4,730 mm (186.2 in) / 1,990 mm (78.3 in) / 1,140 mm (44.9 in)
- _Wheelbase_: 2,670 mm (105.1 in)
- _Curb weight_: around 1,000 kg (2,204 lbs)
-_Top speed_: over 320 km/h (200 mph)
-_0-100 km/h (0-62 mph): around 3 seconds

The CLR GTR was a purpose-built racing car, featuring advanced aerodynamics, a lightweight chassis, and a powerful engine. It competed in several high-profile racing events, including:
1998 24 Hours of Le Mans
1998 FIA GT Championship
- 1999 FIA GT Championship

Unfortunately, the CLR GTR suffered a series of high-profile accidents, including a dramatic flip at Le Mans, which led to its withdrawal from competition.
If you have any specific questions or would like more information, feel free to ask!""" 

@pytest.fixture
def simple_tokenizer(sample_text):
    """creates a basic BPE tokenizer trained on sample data."""
    token_to_id, id_to_token, merges = learn_bpe(sample_text, target_vocab_size=100)
    return Tokenizer(
        encode=lambda text: encode(text, merges),
        decode=lambda tokens: decode(tokens, id_to_token)
    )

def test_data_pipeline(sample_text, simple_tokenizer):
    """test data processing pipeline from text to training batches."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    assert isinstance(tokens, jax.Array)
    assert tokens.ndim == 1
    assert tokens.shape[0] > 0
    
    key = jax.random.PRNGKey(0)
    input_batch, target_batch = create_random_batches(
        tokens, batch_size=2, seq_len=4, key=key
    )
    
    assert input_batch.shape == (2, 4)
    assert target_batch.shape == (2, 4)
    
    decoded_text = simple_tokenizer.decode(tokens[:10].tolist())
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0

def test_kv_cache_generation(small_gpt2_config):
    """test generation with KV cache."""
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_gpt2_config)
    
    prompt = jnp.array([1, 2, 3])
    sampler_config = SamplerConfig(
        temp=0.7,
        max_tokens=5,
        key=key
    )
    
    generated = generate(
        params,
        small_gpt2_config,
        "gpt2",
        prompt,
        sampler_config,
        stream=False
    )
    
    assert len(generated) == sampler_config.max_tokens - len(prompt) + 1
    assert jnp.all(generated < small_gpt2_config.vocab_size)

def test_training_pipeline(small_gpt2_config, training_config, opt_config, sample_text, simple_tokenizer):
    """test complete training pipeline."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_gpt2_config)
    
    trained_params = train(
        params,
        small_gpt2_config,
        "gpt2",
        tokens,
        opt_name="adam",
        opt_config=opt_config,
        train_config=training_config,
        key=key
    )
    
    assert not jax.tree_util.tree_all(
        jax.tree_map(lambda x, y: jnp.array_equal(x, y), params, trained_params)
    )

def test_evaluation(small_gpt2_config, sample_text, simple_tokenizer):
    """test perplexity evaluation."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_gpt2_config)
    
    eval_config = EvalConfig(seq_len=4, stride=2)
    ppl = eval_perplexity(
        params,
        small_gpt2_config,
        "gpt2",
        tokens,
        eval_config
    )
    
    assert isinstance(ppl, jax.Array)
    assert ppl.ndim == 0
    assert ppl >= 1.0

def test_end_to_end(small_gpt2_config, training_config, opt_config, sample_text, simple_tokenizer):
    """test complete workflow: tokenization → training → generation → evaluation."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_gpt2_config)
    
    trained_params = train(
        params,
        small_gpt2_config,
        "gpt2",
        tokens,
        opt_name="adam",
        opt_config=opt_config,
        train_config=training_config,
        key=key
    )
    
    prompt_tokens = tokens[:4]
    sampler_config = SamplerConfig(max_tokens=8, key=key)
    generated = generate(
        trained_params,
        small_gpt2_config,
        "gpt2",
        prompt_tokens,
        sampler_config
    )
    
    decoded_text = simple_tokenizer.decode(generated.tolist())
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
    
    eval_config = EvalConfig(seq_len=4, stride=2)
    ppl = eval_perplexity(
        trained_params,
        small_gpt2_config,
        "gpt2",
        tokens,
        eval_config
    )
    
    assert isinstance(ppl, jax.Array)
    assert ppl >= 1.0