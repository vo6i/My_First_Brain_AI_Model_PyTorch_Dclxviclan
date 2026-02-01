import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import os

# --- КОНФИГУРАЦИЯ ---
BLOCK_SIZE = 64
EMBED_SIZE = 64
HEADS = 4
MODEL_PATH = 'minigpt_checkpoint.pt'
FILE_NAME = 'book.txt' # Используется для загрузки словаря

# --- АРХИТЕКТУРА МОДЕЛИ ---
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_embedding(pos)
        out = self.transformer(out)
        return self.fc_out(out)

# --- ПОДГОТОВКА ДАННЫХ И ТОКЕНИЗАЦИЯ ---
if os.path.exists(FILE_NAME):
    with open(FILE_NAME, 'r', encoding='utf-8') as f: text = f.read()
else:
    # Fallback текст, если book.txt не найден (должен содержать все токены)
    text = "<|user|>привет<|model|>нормально" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- ЗАГРУЗКА МОДЕЛИ ---
model = MiniGPT(vocab_size, EMBED_SIZE, HEADS, BLOCK_SIZE)
if os.path.exists(MODEL_PATH):
    # Загружаем модель на CPU, что важно для HF Spaces с базовым тарифом
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- ЛОГИКА ГЕНЕРАЦИИ С ТОКЕНАМИ И НАСТРОЙКАМИ ---
def predict(prompt, max_length, temperature):
    if not prompt: return "Введите текст"
    
    # Принудительно добавляем токен Модели к запросу пользователя
    full_prompt = prompt.strip() + "<|model|>"
    
    context_tokens = encode(full_prompt)[-BLOCK_SIZE:]
    context = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)
    
    generated_tokens = []
    for _ in range(max_length):
        cond = context[:, -BLOCK_SIZE:]
        with torch.no_grad():
            logits = model(cond)[:, -1, :]
            
            if temperature == 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Остановка генерации, если модель сгенерировала начало токена '<'
            if decode([next_token.item()]) == '<':
                 break

            context = torch.cat((context, next_token), dim=1)
            generated_tokens.append(next_token.item())
            
    return decode(generated_tokens)

# --- ИНТЕРФЕЙС GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 MiniGPT Chat с настройками")
    with gr.Row():
        with gr.Column():
            # Подсказка пользователю начинать с токена для лучшей работы
            input_text = gr.Textbox(label="Ваш запрос (начинайте с <|user|>)", placeholder="Напишите начало фразы...", lines=3)
            max_len_slider = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Максимальная длина ответа")
            temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Температура (0=детерминированный)")
            btn = gr.Button("Сгенерировать")

        output_text = gr.Textbox(label="Ответ модели", lines=10)
    
    btn.click(fn=predict, inputs=[input_text, max_len_slider, temp_slider], outputs=[output_text])

if __name__ == "__main__":
    demo.launch()
