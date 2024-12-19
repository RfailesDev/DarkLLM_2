# model.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash-attn is not installed. Falling back to slow attention.")

class RMSNorm(nn.Module):
    """RMS Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_parameter('bias', None)
        self.eps = eps

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        rms = torch.sqrt(var + self.eps)
        return x / rms * self.weight

class MultiHeadAttention(nn.Module):
    """MHA or GQA depending on config."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.scale = self.head_size ** -0.5

        self.num_kv_heads = config.num_kv_heads
        self.n_kv_groups = self.n_head // self.num_kv_heads

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(dim=self.head_size)
        self.config = config
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Grouped-Query Attention
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.config.use_flash and HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))
        return y

class SwiGLU(nn.Module):
    """SwiGLU activation."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

class FeedForward(nn.Module):
    """Полносвязный слой с активацией SwiGLU."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd * 2),
            SwiGLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Блок Transformer с RoPE и SwiGLU."""
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd) # Use RMSNorm
        self.sa = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.n_embd) # Use RMSNorm
        self.ffwd = FeedForward(config)

    def forward(self, x):
        y = self.sa(self.ln1(x))
        x = x + y
        y = self.ffwd(self.ln2(x))
        x = x + y
        return x

class GPTLanguageModel(nn.Module):
    """Языковая модель на основе GPT с RoPE и SwiGLU."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd) # Use RMSNorm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx