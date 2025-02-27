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
import json

# ----- Tokenization and Vocabulary -----
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

with open("vocabc2p.json", "r") as f:
    loaded_vocab = json.load(f)

loaded_src_vocab = loaded_vocab["src_vocab"]
loaded_tgt_vocab = loaded_vocab["tgt_vocab"]

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

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
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    def forward(self, src, tgt):
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(src.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output.transpose(0, 1)
    


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model (assuming train_dataset is already defined)
model = TransformerPseudo2Code(
    src_vocab_size=35170, 
    tgt_vocab_size=24974).to(device)

# Load model checkpoint and set to evaluation mode
model.load_state_dict(torch.load("transformer_code2pseudo.pth", map_location=device))
model.eval()

def generate_output(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    tokens = simple_tokenizer(src_sentence)
    src_indices = numericalize(tokens, src_vocab)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab[SOS_TOKEN]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = torch.argmax(output[0, -1, :]).item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab[EOS_TOKEN]:
            break
    
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    generated_tokens = [
        inv_tgt_vocab[idx] for idx in tgt_indices 
        if idx not in (tgt_vocab[SOS_TOKEN], tgt_vocab[EOS_TOKEN])
    ]
    return " ".join(generated_tokens)

# ----- Inference Example -----
# sample_code = "int x1;"
# generated_pseudo = generate_output(model, sample_code, loaded_src_vocab, loaded_tgt_vocab, device)
# print("\nSample C++ Code:")
# print(sample_code)
# print("\nGenerated Pseudocode:")
# print(generated_pseudo)