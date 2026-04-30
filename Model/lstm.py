import math
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Literal
from collections import OrderedDict

# Define Attention Mechanism
class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_size:int, bidirectional:bool):
        super().__init__()
        if bidirectional == True:
            factor = 2
        else:
            factor = 1
        self.query = nn.Parameter(torch.randn(factor * hidden_size)) 
        # (bidirections * hidden_size) or (hidden_size)

    def forward(self, hidden_states, mask):
        # hidden_states: (B, num_tokens, bidirections * hidden_size)
        # mask: (B, num_tokens, 1)
        
        # compute attention scores with dot_product
        mask = mask.squeeze() # (B, num_tokens)
        attn_scores = torch.matmul(hidden_states, self.query) / math.sqrt(hidden_states.size(-1)) # (B, num_tokens)

        attn_scores = attn_scores.masked_fill(mask==0, float('-inf')) # (B, num_tokens)

        # compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, num_tokens)

        # Weighted sum
        final_hidden_state = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1) # (B, bidirections * hidden_size)

        return final_hidden_state
    

class LSTMCell(nn.Module):
    def __init__(self, emb_dim:int, hidden_size:int, num_layers:int, dropout:float, bidirectional:bool,
                proj_size:int, pooling_fn: Literal['mean', 'sum', 'max', 'attention']|None = None):
        super().__init__()
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers,
                           bias=True, batch_first=True, dropout=dropout, bidirectional=bidirectional, 
                            proj_size=proj_size)
        self.pooling_fn = pooling_fn
        self.bidirectional = bidirectional
        self.attn_pooling = GlobalAttentionPooling(hidden_size=hidden_size, bidirectional=bidirectional)

    def forward(self, tokens_emb, mask):
        # Since there are some sequence having Padding, values, 
        # we must tell RNN to avoid these values, achieve this using pack_padded_sequence
        
        # mask: 1 where token != pad and unknown, 0 where token == pad ; (B, num_tokens, 1)
        # tokens_emb: (B, num_tokens, emb_dim)

        # compute real length for each sequence in batch
        lengths = mask.sum(dim=1).long() # (B, 1, 1)
        lengths = lengths.squeeze(-1) # (B, )
        
        packed = pack_padded_sequence(tokens_emb, lengths.cpu(), 
                                      batch_first = True, enforce_sorted = False)

        # pass through lstm cell
        output_packed, (last_hidden_state, last_cell_state) = self.lstm(packed)
        # output_packed: PackedSequence object
        # last_hidden_state: (bidirections * num_layers, B, hidden_size) ; last_cell_state: (bidirections * num_layers, B, hidden_size)

        # get tensor back from output_packed
        output, out_lengths = pad_packed_sequence(
            output_packed,
            total_length=tokens_emb.shape[1],
            batch_first=True
        )
        # output: (B, num_tokens, bidirections * hidden_size)
        # out_lengths: (B,) , this contains real length (not include pad) for each sequence in batch

        out_lengths = out_lengths.to(output.device)

        if self.pooling_fn == 'mean':
            final_hidden_state = output.sum(dim=1) / out_lengths.unsqueeze(-1) # (B, bidirections * hidden_size)
        
        elif self.pooling_fn == 'sum':
            final_hidden_state = output.sum(dim=1) # (B, bidirections * hidden_size)
        
        elif self.pooling_fn == 'max':
            output_mask = output.masked_fill(mask==0, float('-inf')) # (B, num_tokens, bidirections * hidden_size)
            final_hidden_state, max_value_index = output_mask.max(dim=1)
            # final_hidden_state: # (B, bidirections * hidden_size)

        elif self.pooling_fn == 'attention':
            final_hidden_state = self.attn_pooling(output, mask) # (B, bidirections * hidden_size)
        
        else:
            final_hidden_state = last_hidden_state.transpose(0, 1) # (B, bidirections * num_layers, hidden_size)

            if self.bidirectional == True:
                forward_hidden_state = final_hidden_state[:, -2, :] # (B, hidden_size)
                backward_hidden_state = final_hidden_state[:, -1, :] # (B, hidden_size)
                final_hidden_state = torch.cat([forward_hidden_state, backward_hidden_state], dim=1) # (B, bidirections * hidden_size)
            else:
                final_hidden_state = final_hidden_state[:, -1, :] # (B, hidden_size)

        # final_hidden_state: (B, bidirections * hidden_size) or (B, hidden_size)
        return final_hidden_state
    

class LSTMModel(nn.Module):
    def __init__(self, num_classes:int, vocab:dict[str, int], emb_dim:int, hidden_size:int, num_layers:int,
                bidirectional:bool, proj_size:int, pooling_fn: Literal['mean', 'sum', 'max', 'attention']|None):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim
        self.pooling_fn = pooling_fn

        # Make Padding Embeddings not contribute to gradient during backward pass -> not to be updated
        self.embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=vocab['<PAD>']) 
        self.dropout_emb = nn.Dropout(p=0.35)

        self.lstm = LSTMCell(emb_dim=emb_dim, hidden_size=hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, dropout=0, 
                           pooling_fn=pooling_fn, proj_size=proj_size)
        # dropout only used when using stack RNNs
        
        if bidirectional == True:
            factor = 2
        else:
            factor = 1
        
        self.mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(factor * hidden_size, 64)),
            ("ln1", nn.LayerNorm(64)),
            ("relu1", nn.LeakyReLU()),
            ("drop1", nn.Dropout(p=0.35)),
            
            ("fc2", nn.Linear(64, 128)),
            ("ln2", nn.LayerNorm(128)),
            ("relu2", nn.LeakyReLU()),
            ("drop2", nn.Dropout(p=0.35)),

            ("fc3", nn.Linear(128, 128)),
            ("ln3", nn.LayerNorm(128)),
            ("relu3", nn.LeakyReLU()),
            ("drop3", nn.Dropout(p=0.35)),
            
            ('out', nn.Linear(128, num_classes))
        ]))

    def forward(self, tokens_idx):
        # tokens_idx: (B, num_tokens)
        tokens_emb = self.embedding(tokens_idx) # (B, num_tokens, emb_dim)
        tokens_emb = self.dropout_emb(tokens_emb)

        # mask: 1 where token != pad and unknown, 0 where token == pad
        mask = (tokens_idx != self.vocab['<PAD>']).float().unsqueeze(-1)  # (B, num_tokens, 1)

        last_hidden_state = self.lstm(tokens_emb, mask) # (B, bidirections * hidden_size) or (B, hidden_size)

        output = self.mlp(last_hidden_state) # (B, num_classes)
        return output