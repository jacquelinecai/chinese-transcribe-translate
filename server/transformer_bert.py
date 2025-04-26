import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import json
from collections import Counter
from itertools import chain
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------ Model Components ------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, V)

    def split_heads(self, x):
        b, seq_len, _ = x.size()
        return x.view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        b, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, seq_len, self.num_heads * self.d_k)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.feed_forward(x)
        return self.norm2(x + self.dropout(x2))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.feed_forward(x)
        return self.norm3(x + self.dropout(x2))

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        nopeak = torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal=1).bool().to(tgt.device)
        tgt_mask = tgt_mask & ~nopeak
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_emb = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_emb = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)

# ------------------ Dataset ------------------

class TranslationDataset(data.Dataset):
    def __init__(self, path, src_vocab=None, tgt_vocab=None, min_freq=2):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        try:
            data_list = json.loads(text)
            if not isinstance(data_list, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            data_list = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.src_sentences = [d['english'].lower().split() for d in data_list]
        self.tgt_sentences = [d['chinese'].lower().split() for d in data_list]

        specials = ['<pad>', '<unk>', '<sos>', '<eos>']
        if src_vocab is None:
            ctr = Counter(chain.from_iterable(self.src_sentences))
            tokens = [w for w, c in ctr.items() if c >= min_freq]
            self.src_vocab = {tok: i for i, tok in enumerate(specials + tokens)}
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            ctr = Counter(chain.from_iterable(self.tgt_sentences))
            tokens = [w for w, c in ctr.items() if c >= min_freq]
            self.tgt_vocab = {tok: i for i, tok in enumerate(specials + tokens)}
        else:
            self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        src_ids = [self.src_vocab.get(w, self.src_vocab['<unk>']) for w in src]
        tgt_ids = [self.tgt_vocab.get(w, self.tgt_vocab['<unk>']) for w in tgt]
        src_ids = [self.src_vocab['<sos>']] + src_ids + [self.src_vocab['<eos>']]
        tgt_ids = [self.tgt_vocab['<sos>']] + tgt_ids + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded

# ------------------ Label Smoothing ------------------

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        ignore = target == self.ignore_index
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return self.criterion(pred, true_dist)

# ------------------ BERTScore ------------------

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').eval().cpu()

def encode_with_bert(sentences):
    with torch.no_grad():
        inputs = tokenizer_bert(sentences, padding=True, truncation=True, return_tensors="pt")
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

def compute_bert_score(preds, targets):
    pred_embeds = encode_with_bert(preds)
    target_embeds = encode_with_bert(targets)
    sims = cosine_similarity(pred_embeds, target_embeds)
    return np.mean(np.diag(sims))

# ------------------ Train ------------------

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TranslationDataset("../data/translation2019zh/translation2019zh_train.json")
    valid_ds = TranslationDataset("../data/translation2019zh/translation2019zh_valid.json",
                                  src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    train_loader = data.DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_loader = data.DataLoader(valid_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = Transformer(len(train_ds.src_vocab), len(train_ds.tgt_vocab),
                        d_model=384, num_heads=6, num_layers=4, d_ff=1536, max_seq_length=100, dropout=0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = LabelSmoothingLoss(0.1, len(train_ds.tgt_vocab), ignore_index=train_ds.tgt_vocab['<pad>'])

    for epoch in range(20):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = criterion(out.view(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(device), tgt.to(device)
                out = model(src, tgt[:, :-1])
                val_loss += criterion(out.view(-1, out.size(-1)), tgt[:, 1:].reshape(-1)).item()

                pred_ids = out.argmax(-1).cpu().tolist()
                tgt_ids = tgt[:, 1:].cpu().tolist()

                for p, t in zip(pred_ids, tgt_ids):
                    all_preds.append(" ".join([train_ds.tgt_vocab.get(i, '') for i in p]))
                    all_targets.append(" ".join([train_ds.tgt_vocab.get(i, '') for i in t]))

        bert = compute_bert_score(all_preds, all_targets)
        print(f"           Valid Loss: {val_loss/len(valid_loader):.4f}, BERTScore: {bert:.4f}")

    torch.save(model.state_dict(), "transformer_model_small.pth")
    print("Saved transformer_model_small.pth")

if __name__ == "__main__":
    train_transformer()
