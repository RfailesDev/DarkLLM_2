# train.py

import torch
from model import GPTLanguageModel
from data import get_data_loaders
from config import Config
from tqdm import tqdm
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Train GPT Language Model with RoPE and SwiGLU')
    parser.add_argument('--epochs', type=int, default=Config.max_epochs, help='Количество эпох')
    parser.add_argument('--num_workers', type=int, default=4, help='Количество воркеров для DataLoader')
    parser.add_argument('--batch_size', type=int, default=None, help='Размер батча (по умолчанию из конфига)')
    parser.add_argument('--early_stop', type=int, default=3, help='Количество эпох без улучшения для ранней остановки')
    args = parser.parse_args()

    config = Config()
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device(config.device)
    print(f"Используется устройство: {device}")

    # Загрузка данных
    train_loader, val_loader, vocab_size, tokenizer = get_data_loaders(
        config.data_path,
        config.block_size,
        config.batch_size,
        device,
        tokenizer_path=config.tokenizer_dir,
        num_workers=args.num_workers
    )
    config.vocab_size = vocab_size  # Установка размера словаря в конфигурацию

    # Инициализация модели
    model = GPTLanguageModel(config).to(device)
    print(f"Размер словаря: {vocab_size}")
    print(model)

    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)

    # Смешанная точность
    scaler = torch.cuda.amp.GradScaler()

    # Цикл обучения
    for epoch in range(1, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        for xb, yb in loop:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                _, loss = model(xb, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        # Сохранение модели
        save_checkpoint(model, optimizer, epoch, loss.item(), config)

def save_checkpoint(model, optimizer, epoch, loss, config):
    """Сохранение модели и оптимизатора."""
    os.makedirs(config.save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    save_path = os.path.join(config.save_dir, f"{config.checkpoint_prefix}{epoch}.pt")
    torch.save(checkpoint, save_path)
    print(f"Сохранено: {save_path}")

if __name__ == '__main__':
    main()