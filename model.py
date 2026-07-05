import math

import torch
import torch.nn as nn
import torch.nn.functional as nnfunc

from config import GPTConfig


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return nnfunc.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(nnfunc, "scaled_dot_product_attention")
        if not self.flash:
            causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        if self.flash:
            is_causal = past_kv is None
            y = nnfunc.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if past_kv is None:
                att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = nnfunc.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v 

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None, past_kv_list=None, use_cache=False):
        device = idx.device
        B, T = idx.size()

        past_len = 0 if past_kv_list is None else past_kv_list[0][0].size(2)
        assert past_len + T <= self.config.block_size, (
            f"sequence length {past_len + T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        new_kv_list = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kv_list is None else past_kv_list[i]
            x, kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kv_list.append(kv)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nnfunc.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_list

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True):
        was_training = self.training
        self.eval()
        past_kv_list = None
        if use_cache:
            prompt_len = min(idx.size(1), self.config.block_size)
            max_generatable = self.config.block_size - prompt_len
            if max_new_tokens > max_generatable:
                print(f"[generate] warning: prompt_len={prompt_len} + max_new_tokens={max_new_tokens} "
                      f"would exceed block_size={self.config.block_size}; the KV-cache has no "
                      f"sliding-window eviction, so capping to {max_generatable} new tokens. "
                      f"Pass use_cache=False to instead keep generating via a re-cropped window.")
                max_new_tokens = max_generatable

        for _ in range(max_new_tokens):
            if past_kv_list is None:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            else:
                idx_cond = idx[:, [-1]]

            logits, _, new_kv = self.forward(idx_cond, use_cache=use_cache, past_kv_list=past_kv_list)
            if use_cache:
                past_kv_list = new_kv

            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = nnfunc.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                sorted_mask = (cum_probs - probs) > top_p
                sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = nnfunc.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train(was_training)
        return idx

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay &= param_dict.keys()
        no_decay &= param_dict.keys()

        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0, "not all parameters were grouped"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
