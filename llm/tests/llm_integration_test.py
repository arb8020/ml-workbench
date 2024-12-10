import pytest
import jax
import jax.numpy as jnp
from model import ModelConfig, gpt2_forward, init_gpt2_params
from training import (
    AdamConfig, DatasetConfig, TrainConfig, Tokenizer, batch_forward, prepare_dataset, create_random_batches,
    cross_entropy_loss, train, generate, encode, decode, learn_bpe
)

# Fixtures
@pytest.fixture
def small_model_config():
    """small model config for testing"""

    return ModelConfig(
        vocab_size=100,        
        embedding_dim=32,      
        context_len=16,       
        n_head=2,             
        n_blocks=2            
    )

@pytest.fixture
def opt_config():
    """basic adam optimizer configuration"""

    return AdamConfig(
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )

@pytest.fixture
def sample_text():
    """sample text with repeating pattern for training tests"""
    
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

If you have any specific questions or would like more information, feel free to ask!""" # Simple repeating pattern

@pytest.fixture
def simple_tokenizer(sample_text):
    """creates a basic BPE tokenizer trained on sample data"""

    token_to_id, id_to_token, merges = learn_bpe(sample_text, target_vocab_size=100)
    return Tokenizer(  # Return the Tokenizer object directly
        encode=lambda text: encode(text, merges),
        decode=lambda tokens: decode(tokens, id_to_token)
    )

@pytest.fixture
def train_config():
    """basic training configuration"""

    return TrainConfig(
        num_epochs=2,
        batches_per_epoch=5,
        batch_size=4,
        batch_seq_len=8,
        seed=42
    )

# Integration Tests
def test_data_pipeline(sample_text, simple_tokenizer):
    """test entire data pipeline from text to batches"""

    # test tokenization
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    assert isinstance(tokens, jax.Array)
    assert tokens.ndim == 1
    assert tokens.shape[0] > 0
    
    # test batch creation
    key = jax.random.PRNGKey(0)
    input_batch, target_batch = create_random_batches(
        tokens, batch_size=2, seq_len=4, key=key
    )
    
    assert input_batch.shape == (2, 4)
    assert target_batch.shape == (2, 4)

def test_model_architecture(small_model_config, sample_text, simple_tokenizer):
    """test model initialization and forward pass"""

    # initialize model
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_model_config)
    
    # get tokens from sample text fixture 
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    batch_size, seq_len = 2, 8
    key, subkey = jax.random.split(key)
    batch_tokens, _ = create_random_batches(tokens, batch_size, seq_len, subkey)
    
    # test forward pass
    logits = batch_forward(params, small_model_config, "gpt2", batch_tokens)
    expected_shape = (batch_size, seq_len, small_model_config.vocab_size)
    assert logits.shape == expected_shape

def test_training(small_model_config, train_config, opt_config, sample_text, simple_tokenizer):
    """test that we can run training and loss decreases"""

    tokens = prepare_dataset(sample_text, simple_tokenizer)
    
    # initialize model
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_model_config)

    # test training
    trained_params = train(
        params,
        small_model_config,
        "gpt2",
        tokens, 
        opt_name="adam",
        opt_config=opt_config,  
        train_config=train_config,
        key=key
    )
    
    assert isinstance(trained_params, dict)
    assert not jax.tree_util.tree_all(
        jax.tree_map(lambda x, y: jnp.array_equal(x, y), params, trained_params)
    )

def test_generation(small_model_config, sample_text, simple_tokenizer):
    """test generation"""

    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_model_config)
    
    # get real tokens from sample text 
    tokens = prepare_dataset(sample_text, simple_tokenizer)[:4] 
    prompt_tokens = tokens.reshape(-1) # make sure one dim 
    
    generated = generate(
        params,
        small_model_config,
        "gpt2",
        prompt_tokens,
        max_new=10,
        key=key
    )
    
    assert isinstance(generated, jax.Array)
    assert generated.shape[0] == 10  # should generate requested number of tokens
    assert jnp.all(generated < small_model_config.vocab_size)  # ensure valid token ids

def test_end_to_end(small_model_config, train_config, opt_config, sample_text, simple_tokenizer):
    """test full workflow: data → training → generation"""

    # prepare data
    tokens = prepare_dataset(sample_text, simple_tokenizer) 
    
    # train model
    key = jax.random.PRNGKey(0)
    params = init_gpt2_params(key, small_model_config)
    
    trained_params = train(
        params,
        small_model_config,
        "gpt2",
        tokens, 
        opt_name="adam",
        opt_config=opt_config,  
        train_config=train_config,
        key=key
    )
    
    # generate from trained model
    prompt_tokens = tokens[:4]

    generated = generate(
        trained_params,
        small_model_config,
        "gpt2",
        prompt_tokens,
        max_new=8,
        key=key
    )
    
    assert isinstance(generated, jax.Array)
    assert generated.shape[0] == 8
    assert jnp.all(generated < small_model_config.vocab_size)