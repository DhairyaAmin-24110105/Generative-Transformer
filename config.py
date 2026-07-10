from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256     
    vocab_size: int = 65     
    n_layer: int = 6          
    n_head: int = 6          
    n_embd: int = 384        
    dropout: float = 0.2      
    bias: bool = False        


PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1),
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2),
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=512, dropout=0.2),
}
