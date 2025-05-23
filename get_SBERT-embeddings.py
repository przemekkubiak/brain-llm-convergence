import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import pandas as pd

# LOAD DATA

d = np.load('data/aligned_data_subject_2.npz', allow_pickle=True)
d = d['aligned_data']

# Load linguistic features
df = pd.read_csv('data/features.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Load SentenceTransformer model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
device = torch.device("mps")

model[0].auto_model.to(device)

numeric_cols = [col for col in df.columns if col not in ['word']]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

for item in d:
    words = item['words']
    rows = df[df['word'].isin(words)]
    avg_features = rows[numeric_cols].mean().to_dict()
    for col, val in avg_features.items():
        item[f"avg_{col}"] = val
    

model[0].auto_model.config.output_hidden_states = True
model[0].auto_model.config.output_attentions = True

for item in d:
    words = item['words']
    tokens = model.tokenize(words)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model[0].auto_model(**tokens, return_dict=True)

    with torch.no_grad():
        outputs = model[0].auto_model(**tokens, return_dict=True)

    # Store embeddings per layer
    for i, layer_hidden in enumerate(outputs.hidden_states):
        # E.g. mean pool the hidden state
        layer_embedding = layer_hidden.mean(dim=1).cpu().numpy()
        item[f"layer_{i+1}_embeddings"] = layer_embedding

    # Store attention per layer
    for i, attn in enumerate(outputs.attentions):
        item[f"layer_{i+1}_attention"] = attn.cpu().numpy()
        
np.savez('data/aligned_data_subject_2_embeddings_features.npz', aligned_data=d)
