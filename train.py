import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os 

# --- КОНФИГУРАЦИЯ ---
FILE_NAME = 'book.txt'
MODEL_PATH = 'minigpt_checkpoint.pt'
BLOCK_SIZE = 64
BATCH_SIZE = 16
EMBED_SIZE = 64
HEADS = 4
LR = 0.001
EPOCHS = 300 

# --- 1. АРХИТЕКТУРА МОДЕЛИ ---
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, block_size):
        super(MiniGPT, self).__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos)
        out = tok_emb + pos_emb
        out = self.transformer(out)
        logits = self.fc_out(out)
        return logits

# --- 2. ПОДГОТОВКА ДАННЫХ И ТОКЕНИЗАЦИЯ ---
try:
    with open(FILE_NAME, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Успешно прочитан файл: {FILE_NAME}, размер текста: {len(text)} символов.")
except FileNotFoundError:
    print(f"Ошибка: файл '{FILE_NAME}' не найден. Использую fallback текст.")
    text = "<|user|>привет<|model|>нормально" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Данные закодированы в тензор размером: {data.shape}")

# --- 3. НАСТРОЙКИ ОБУЧЕНИЯ И ИНИЦИАЛИЗАЦИЯ ---
model = MiniGPT(vocab_size, EMBED_SIZE, HEADS, BLOCK_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# --- 4. ЦИКЛ ОБУЧЕНИЯ ---
print("Начинаю обучение...")
model.train()

for epoch in range(EPOCHS):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    xb = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    yb = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

    logits = model(xb)
    B, T, C = logits.shape
    loss = criterion(logits.view(B*T, C), yb.view(B*T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Эпоха {epoch}, Ошибка: {loss.item():.4f}")

print("Обучение завершено.")

# --- 5. СОХРАНЕНИЕ МОДЕЛИ ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"Модель сохранена в файл {MODEL_PATH}")
