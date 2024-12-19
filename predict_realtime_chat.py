# predict_realtime_chat.py - генерация текста в реальном времени

import torch
from model import GPTLanguageModel
from data import TextDataset
import torch.nn.functional as F
from config import Config
import os


class RealTimeChatbot:
    def __init__(self, checkpoint_path, data_path, device='cpu', tokenizer_dir='tokenizer'):
        self.config = Config()
        self.device = torch.device(device)
        print(f"Используется устройство: {self.device}")

        # Загрузка токенизатора
        if os.path.exists(os.path.join(tokenizer_dir, 'vocab.json')):
            from tokenizers import ByteLevelBPETokenizer
            self.tokenizer = ByteLevelBPETokenizer.from_file(
                vocab_filename=os.path.join(tokenizer_dir, 'vocab.json'),
                merges_filename=os.path.join(tokenizer_dir, 'merges.txt')
            )
            print("Токенизатор загружен.")
        else:
            raise FileNotFoundError(f"Токенизатор в {tokenizer_dir} не найден.")

        # Установка размера словаря
        self.config.vocab_size = self.tokenizer.get_vocab_size()

        # Инициализация модели
        self.model = GPTLanguageModel(self.config).to(self.device)

        # Загрузка контрольной точки
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Контрольная точка {checkpoint_path} не найдена.")
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, filename):
        """Загрузка контрольной точки модели."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        if epoch is not None and loss is not None:
            print(f"Загружена модель из эпохи {epoch} с потерей {loss:.4f}")

    def generate_realtime(self, prompt, max_new_tokens=128, temperature=1.0, top_k=50, top_p=0.9):
        """Генерация текста в режиме реального времени с использованием генератора."""
        if prompt:
            context = self.tokenizer.encode(prompt).ids
            context = torch.tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            context = torch.tensor([[self.tokenizer.token_to_id('<s>')]], dtype=torch.long, device=self.device)

        generated_tokens = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Обрезаем context до block_size
                context_cropped = context[:, -self.config.block_size:]

                output = self.model(context_cropped)

                # Извлекаем logits из кортежа
                logits, _ = output

                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token_idx.item())

                # Декодируем текущий токен
                decoded_token = self.tokenizer.decode([next_token_idx.item()])

                # Возвращаем текущий токен как чанк
                yield decoded_token

                # Добавляем сгенерированный токен к context
                context = torch.cat((context, next_token_idx), dim=1)

                # Если сгенерирован токен конца последовательности, завершаем генерацию
                if next_token_idx.item() == self.tokenizer.token_to_id('</s>'):
                    break


def main():
    checkpoint_path = 'checkpoints_rope_swigu_small/model_epoch_28.pt'
    data_path = Config.data_path
    device = Config.device
    tokenizer_dir = Config.tokenizer_dir

    chatbot = RealTimeChatbot(checkpoint_path, data_path, device, tokenizer_dir)

    while True:
        prompt = input("Введите текст (или 'exit' для выхода): ")
        if prompt:
            if prompt.lower() == 'exit':
                break


            print("Ответ:", end=" ")
            response_generator = chatbot.generate_realtime(
                prompt,
                max_new_tokens=Config.max_new_tokens,
                temperature=Config.temperature,
                top_k=Config.top_k,
                top_p=Config.top_p
            )
            for chunk in response_generator:
                print(chunk, end="", flush=True)
            print()


if __name__ == '__main__':
    main()