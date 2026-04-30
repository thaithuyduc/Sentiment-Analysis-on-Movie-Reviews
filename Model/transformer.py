import torch
import torch.nn as nn
import math
from collections import OrderedDict
from typing import Literal

class InputEmbedding(nn.Module):
    def __init__(self, vocab:dict[str, int], emb_dim: int):
        super().__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=vocab['<PAD>'])
        self.dropout_emb = nn.Dropout(p=0.35)
    
    # Sinusoidal Positional Embedding
    def positional_encoding(self, num_tokens, emb_dim, device):
        position = torch.arange(num_tokens, device=device).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, emb_dim, 2, device=device).float() * -(math.log(10000.0) / emb_dim))
        
        pe = torch.zeros(num_tokens, emb_dim, device=device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe # (num_tokens, emb_dim)

    def forward(self, tokens_idx: torch.Tensor):
        # tokens_idx: (B, num_tokens)
        mask = (tokens_idx == self.vocab['<PAD>']) # (B, num_tokens) Boolean type
        
        tokens_emb = self.embedding(tokens_idx) # (B, num_tokens, emb_dim)
        tokens_emb = self.dropout_emb(tokens_emb) # (B, num_tokens, emb_dim)
        
        B, num_tokens, emb_dim = tokens_emb.shape
        device = tokens_emb.device

        positional_emb = self.positional_encoding(num_tokens, emb_dim, device) # (num_tokens, emb_dim)
        positional_emb = positional_emb.unsqueeze(0) # (1, num_tokens, emb_dim)
        
        input_emb = tokens_emb + positional_emb # (B, num_tokens, emb_dim)

        return input_emb, mask
    

class TransformerLayer(nn.Module):
    def __init__(self, emb_dim: torch.Tensor, num_heads: int, ffn_dim: int):
        super().__init__()
        self.multi_attention = nn.MultiheadAttention(embed_dim = emb_dim, num_heads = num_heads,
                                                    dropout=0.0, bias=True, add_bias_kv=False,
                                                    add_zero_attn=False, kdim=None, vdim=None,
                                                    batch_first=True)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.drop_out = nn.Dropout(p=0.35)
    
        self.feed_forward = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(emb_dim, ffn_dim)),
            # ("ln1", nn.LayerNorm(ffn_dim)),
            ("GELU1", nn.GELU()),
            ("drop1", nn.Dropout(p=0.35)),
            
            ("fc2", nn.Linear(ffn_dim, emb_dim)),
            # ("ln2", nn.LayerNorm(128)),
            # ("relu2", nn.LeakyReLU()),
            # ("drop2", nn.Dropout(p=0.2)),

            # ("fc3", nn.Linear(128, 128)),
            # ("ln3", nn.LayerNorm(128)),
            # ("relu3", nn.LeakyReLU()),
            # ("drop3", nn.Dropout(p=0.35)),
            
            # ('out', nn.Linear(128, num_classes))
        ]))

    def forward(self, input_emb_query, input_emb_key, input_emb_value, key_padding_mask):
        
        # input_emb_query, input_emb_key, input_emb_value: (B, num_tokens, emb_dim)
        attn_output, attn_output_weights = self.multi_attention(input_emb_query, input_emb_key, 
                                                                input_emb_value, key_padding_mask=key_padding_mask)
        # attn_output: (B, num_tokens, emb_dim)
        output = attn_output + input_emb_query # (B, num_tokens, emb_dim)

        output = self.layer_norm1(output) # (B, num_tokens, emb_dim)

        ffn_output = self.feed_forward(output) # (B, num_tokens, emb_dim)
        ffn_output = self.drop_out(ffn_output) # (B, num_tokens, emb_dim)
        
        final_output = output + ffn_output # (B, num_tokens, emb_dim)
        final_output = self.layer_norm2(final_output) # (B, num_tokens, emb_dim)
        
        return final_output
    

class EncoderTransformerModel(nn.Module):
    def __init__(self, vocab:dict[str, int], emb_dim: int, num_heads: int, ffn_dim: int,
                num_transformer_layers:int, pooling_fn: Literal['mean', 'max', 'sum'], num_classes:int):
        super().__init__()
        self.input_emb = InputEmbedding(vocab, emb_dim)
        self.transformer_layer = TransformerLayer(emb_dim, num_heads, ffn_dim)
        
        self.transformer_layer = nn.ModuleList([
            TransformerLayer(emb_dim, num_heads, ffn_dim) for _ in range(num_transformer_layers)
        ])
        
        self.pooling_fn = pooling_fn

        self.mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(emb_dim, 128)),
            ("ln1", nn.LayerNorm(128)),
            ("relu1", nn.LeakyReLU()),
            ("drop1", nn.Dropout(p=0.35)),
            
            # ("fc2", nn.Linear(128, 128)),
            # ("ln2", nn.LayerNorm(128)),
            # ("relu2", nn.LeakyReLU()),
            # ("drop2", nn.Dropout(p=0.1)),

            # ("fc3", nn.Linear(128, 128)),
            # ("ln3", nn.LayerNorm(128)),
            # ("relu3", nn.LeakyReLU()),
            # ("drop3", nn.Dropout(p=0.35)),
            
            ('out', nn.Linear(128, num_classes))
        ]))
    
    def forward(self, tokens_idx: torch.Tensor):
        # tokens_idx: (B, num_tokens)
        input_emb, mask = self.input_emb(tokens_idx)
        # input_emb_for_transformer: (B, num_tokens, emb_dim)
        # mask: (B, num_tokens)
        for layer in self.transformer_layer:
            output = layer(input_emb, input_emb, input_emb, key_padding_mask=mask)
            # output (B, num_tokens, emb_dim)
        
        mask = mask.unsqueeze(-1) # (B, num_tokens, 1)
        output = output.masked_fill(mask, 0.0) # (B, num_tokens, emb_dim) ; used to zero out emb_dim of all <PAD> tokens

        if self.pooling_fn == 'mean':
            lengths = (~mask).sum(dim=1) # (B, 1)
            sum_output = output.sum(dim=1) # (B, emb_dim)
            final_output = sum_output / lengths # (B, emb_dim)
        elif self.pooling_fn == 'sum':
            final_output = output.sum(dim=1) # (B, emb_dim)
        elif self.pooling_fn == 'max':
            output_for_max = output.masked_fill(mask, float('-inf')) # (B, num_tokens, emb_dim)
            final_output, max_value_index = output_for_max.max(dim=1)
            # final_output: (B, emb_dim)

        output_mlp = self.mlp(final_output) # (B, num_classes)

        return output_mlp