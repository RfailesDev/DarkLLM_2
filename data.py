# data.py

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
import os
from config import Config

class TextDataset(Dataset):
    """Датасет для текстовых данных с использованием BPE токенизации."""

    def __init__(self, file_path, block_size, tokenizer_path=None):
        self.block_size = block_size

        # Инициализация или загрузка токенизатора
        if tokenizer_path and os.path.exists(os.path.join(tokenizer_path, 'tokenizer.json')):
            self.tokenizer = ByteLevelBPETokenizer.from_file(
                os.path.join(tokenizer_path, 'vocab.json'),
                os.path.join(tokenizer_path, 'merges.txt')
            )
            print("Токенизатор загружен из существующего файла.")
        else:
            self.tokenizer = self.train_tokenizer(file_path, tokenizer_path)

        # Загрузка и токенизация текста
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.token_ids = self.tokenizer.encode(text).ids
        self.vocab_size = self.tokenizer.get_vocab_size()

    def train_tokenizer(self, file_path, tokenizer_path):
        """Обучение BPE токенизатора и сохранение его."""
        tokenizer = ByteLevelBPETokenizer()

        # Обучение токенизатора
        tokenizer.train(
            files=[file_path],
            vocab_size=Config.tokenizer_vocab_size,
            min_frequency=Config.tokenizer_min_frequency,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )

        # Сохранение токенизатора
        if tokenizer_path:
            os.makedirs(tokenizer_path, exist_ok=True)
            tokenizer.save_model(tokenizer_path)
            print(f"Токенизатор обучен и сохранен в {tokenizer_path}")

        return tokenizer

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1: idx + self.block_size + 1], dtype=torch.long)
        return x, y

def get_data_loaders(file_path, block_size, batch_size, device, tokenizer_path='tokenizer', num_workers=4):
    """Создание загрузчиков данных для обучения и валидации с использованием BPE токенизации."""
    dataset = TextDataset(file_path, block_size, tokenizer_path)
    vocab_size = dataset.vocab_size

    n = int(0.9 * len(dataset))
    n_val = len(dataset) - n
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, vocab_size, dataset.tokenizer