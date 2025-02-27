import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Model Architecture
# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Tokenizer function
def simple_tokenizer(text):
    return text.strip().split()

def numericalize(sentence, vocab):
    return [vocab[SOS_TOKEN]] + [vocab[token] for token in sentence if token in vocab] + [vocab[EOS_TOKEN]]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerPseudo2Code(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src: (batch, src_seq_len), tgt: (batch, tgt_seq_len)
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)  # (batch, src_seq_len, d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)  # (batch, tgt_seq_len, d_model)
        src_emb = src_emb.transpose(0, 1)  # (src_seq_len, batch, d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_seq_len, batch, d_model)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)  # (tgt_seq_len, batch, tgt_vocab_size)
        return output.transpose(0, 1)  # (batch, tgt_seq_len, tgt_vocab_size)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 3


# Model loading and generation function
def load_model_and_generate(model_path, pseudo_text, temperature=1.0, max_len=50):
    # Load the saved model and vocabularies
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # Initialize the model with the same vocab sizes
    loaded_model = TransformerPseudo2Code(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab)
    ).to(DEVICE)
    
    # Load the model weights
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    # Generate code using the loaded model
    tokens = simple_tokenizer(pseudo_text)
    src_indices = numericalize(tokens, src_vocab)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Start decoding with SOS token
    tgt_indices = [tgt_vocab[SOS_TOKEN]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = loaded_model(src_tensor, tgt_tensor)
        
        next_token_logits = output[0, -1, :] / temperature
        # Apply softmax to get probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        # Sample from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1).item()
        tgt_indices.append(next_token)
        
        if next_token == tgt_vocab[EOS_TOKEN]:
            break
            
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    generated_tokens = [inv_tgt_vocab[idx] for idx in tgt_indices 
                       if idx not in (tgt_vocab[SOS_TOKEN], tgt_vocab[EOS_TOKEN])]
    
    return " ".join(generated_tokens)

# Test the model loading and generation function
# print("\nTesting model loading and code generation:")
# test_examples = [
#     "create integers x1, y1, x2, y2",
#     "set max = a at 0",
#     "let n be an integer"
# ]

# for example in test_examples:
#     print(f"\nPseudo Instruction: {example}")
    
#     # Generate with different temperatures
#     for temp in [0.5, 0.7]:
#         generated = load_model_and_generate("transformer_pseudo2code.pth", example, temperature=temp)
#         print(f"Generated Code (temp={temp}):\n{generated}")