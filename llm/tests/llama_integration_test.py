import pytest
import jax
import jax.numpy as jnp
from models import LlamaConfig, init_llama_params
from core import KVCache, create_causal_mask, initialize_rotation_factors
from training import (
    AdamConfig, TrainConfig, EvalConfig, SamplerConfig,
    generate, eval_perplexity, train,
    prepare_dataset, create_random_batches,
    Tokenizer, encode, decode, learn_bpe
)

@pytest.fixture
def small_llama_config():
    """small Llama config for testing."""
    return LlamaConfig(
        vocab_size=100,
        embedding_dim=32,
        ffn_dim_multiplier=2.0,
        multiple_of=32,
        n_heads=2,
        n_kv_heads=2,
        n_layers=2,
        norm_eps=1e-5,
        rope_theta=10000.0,
        context_len=16,
        use_scaled_rope=False
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

def test_kv_cache_generation(small_llama_config):
    """test generation with KV cache and RoPE."""
    key = jax.random.PRNGKey(0)
    params = init_llama_params(key, small_llama_config)
    
    prompt = jnp.array([1, 2, 3])
    prompt_len = len(prompt)
    
    rotation_factors = initialize_rotation_factors(
        small_llama_config.embedding_dim // small_llama_config.n_heads,
        prompt_len + 5,  
        small_llama_config.rope_theta
    )
    
    sampler_config = SamplerConfig(
        temp=0.7,
        max_tokens=5,
        key=key
    )
    
    generated = generate(
        params,
        small_llama_config,
        "llama",
        prompt,
        sampler_config,
        model_kwargs = {'rotation_factors': rotation_factors},
        stream=False,
        
    )
    
    assert len(generated) == sampler_config.max_tokens - len(prompt) + 1
    assert jnp.all(generated < small_llama_config.vocab_size)

def test_training_pipeline(small_llama_config, training_config, opt_config, sample_text, simple_tokenizer):
    """test complete training pipeline."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    
    key = jax.random.PRNGKey(0)
    params = init_llama_params(key, small_llama_config)
    
    rotation_factors = initialize_rotation_factors(
        small_llama_config.embedding_dim // small_llama_config.n_heads,
        16,
        small_llama_config.rope_theta
    )

    trained_params = train(
        params,
        small_llama_config,
        "llama",
        tokens,
        opt_name="adam",
        opt_config=opt_config,
        train_config=training_config,
        key=key,
        model_kwargs = {'rotation_factors': rotation_factors},
    )
    
    assert not jax.tree_util.tree_all(
        jax.tree_map(lambda x, y: jnp.array_equal(x, y), params, trained_params)
    )

def test_evaluation(small_llama_config, sample_text, simple_tokenizer):
    """test perplexity evaluation."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    key = jax.random.PRNGKey(0)
    params = init_llama_params(key, small_llama_config)
    
    rotation_factors = initialize_rotation_factors(
        small_llama_config.embedding_dim // small_llama_config.n_heads,
        4,
        small_llama_config.rope_theta
    )
    
    eval_config = EvalConfig(seq_len=4, stride=2)
    ppl = eval_perplexity(
        params,
        small_llama_config,
        "llama",
        tokens,
        eval_config,
        model_kwargs = {'rotation_factors': rotation_factors},
    )
    
    assert isinstance(ppl, jax.Array)
    assert ppl.ndim == 0
    assert ppl >= 1.0

def test_end_to_end(small_llama_config, training_config, opt_config, sample_text, simple_tokenizer):
    """test complete workflow: tokenization → training → generation → evaluation."""
    tokens = prepare_dataset(sample_text, simple_tokenizer)
    
    key = jax.random.PRNGKey(0)
    params = init_llama_params(key, small_llama_config)
    
    prompt_tokens = tokens[:4]
    prompt_len = len(prompt_tokens)
    
    rotation_factors = initialize_rotation_factors(
        small_llama_config.embedding_dim // small_llama_config.n_heads,
        prompt_len + 8,  # For prompt + max_tokens
        small_llama_config.rope_theta
    )
    
    trained_params = train(
        params,
        small_llama_config,
        "llama",
        tokens,
        opt_name="adam",
        opt_config=opt_config,
        train_config=training_config,
        key=key,
        model_kwargs = {'rotation_factors': rotation_factors},
    )
    
    
    sampler_config = SamplerConfig(max_tokens=8, key=key)
    generated = generate(
        trained_params,
        small_llama_config,
        "llama",
        prompt_tokens,
        sampler_config,
        model_kwargs = {'rotation_factors': rotation_factors}
    )
    
    decoded_text = simple_tokenizer.decode(generated.tolist())
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
    
    eval_config = EvalConfig(seq_len=4, stride=2)
    ppl = eval_perplexity(
        trained_params,
        small_llama_config,
        "llama",
        tokens,
        eval_config,
        model_kwargs = {'rotation_factors': rotation_factors},
    )
    
    assert isinstance(ppl, jax.Array)
    assert ppl >= 1.0

def test_cli_generation(tmp_path):
    """Test generation pipeline using the same setup as the CLI."""
    from models import load_llama_params
    from training import generate, SamplerConfig
    from core import initialize_rotation_factors
    import numpy as np
    from transformers import AutoTokenizer
    
    # Create a mock input
    test_input = "chip"
    
    # Load model and tokenizer (similar to CLI)
    try:
        print('Loading LLaMA model...')
        params, model_config = load_llama_params("/content/drive/MyDrive/Llama3.2-1B/")
        
        print("Loading LLaMA tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/content/drive/MyDrive/Llama3.2-1B/",
            local_files_only=True,
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="</s>",
            bos_token="<s>"
        )
        
        # # Print model configuration for debugging
        # print("\nModel Configuration:")
        # for key, value in model_config.__dict__.items():
        #     print(f"{key}: {value}")
            
        # Print parameter shapes
        # print("\nParameter shapes:")
        for layer_idx in range(model_config.n_layers):
            layer = params[f'layer_{layer_idx}']
            # print(f"\nLayer {layer_idx} feed forward shapes:")
            # print(f"w1: {layer['feed_forward']['w1'].shape}")
            # print(f"w2: {layer['feed_forward']['w2'].shape}")
            # print(f"w3: {layer['feed_forward']['w3'].shape}")
        
        print(f'\nModel output shape: {params["output"].T}')
        # Initialize components needed for generation
        rotation_factors = initialize_rotation_factors(
            model_config.embedding_dim // model_config.n_heads,
            model_config.context_len,
            model_config.rope_theta
        )
        
        # Set up sampler config (similar to CLI defaults)
        sampler_config = SamplerConfig(
            temp=1.0,
            min_p=0.1,
            max_tokens=50,
            key=jax.random.PRNGKey(0)
        )
        
        # Tokenize input
        input_tokens = np.array(tokenizer.encode(test_input))
        print(f"\nInput tokens shape: {input_tokens.shape}")
        
        # Generate
        model_kwargs = {'rotation_factors': rotation_factors}
        output_tokens = generate(
            params,
            model_config,
            'llama',
            input_tokens,
            sampler_config,
            model_kwargs,
            stream=False,
            tokenizer=tokenizer
        )
        
        # Decode output
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f'\nGenerated text: {generated_text}')
        
        assert len(generated_text) > 0
        
    except Exception as e:
        print(f"\nError occurred during test:")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        raise