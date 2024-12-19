# config.py

import torch

class Config:
    # Общие параметры
    block_size = 128
    batch_size = 64
    max_epochs = 10
    learning_rate = 3e-4
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Параметры модели
    n_embd = 512
    n_head = 8
    n_layer = 6

    # New parameters for GQA/MQA
    num_kv_heads = 8  # Number of key/value heads (can be less than n_head for GQA/MQA)

    # Пути к файлам
    data_path = 'C:/Users/matve/Documents/combined_c3_part_100.txt'
    save_dir = 'checkpoints'
    checkpoint_prefix = 'model_epoch_'
    tokenizer_dir = 'tokenizer'

    # Токенизация
    tokenizer_vocab_size = 30000
    tokenizer_min_frequency = 2

    # Инференс параметры
    temperature = 0.7
    top_k = 50
    top_p = 0.9
    max_new_tokens = 128

    # Дополнительно
    vocab_size = None

    # Параметры для RoPE (can be handled by FlashAttention, but we'll keep it for now)
    rotary_dim = n_embd // n_head

    # Flag to use FlashAttention
    use_flash = True