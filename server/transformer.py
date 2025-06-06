import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import json
from collections import Counter
from itertools import chain

# ------------------ Model Components ------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
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
        return x.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_out))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
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
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        size = tgt.size(1)
        nopeak = torch.triu(torch.ones(1, size, size), diagonal=1).bool().to(tgt.device)
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
            data_list = [json.loads(l) for l in text.splitlines() if l.strip()]

        self.src_sentences = [list(d['chinese'].lower()) for d in data_list]
        self.tgt_sentences = [d['english'].lower().split() for d in data_list]

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

# ------------------ Training Utilities ------------------

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
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
    
# ------------------ Learning Rate Scheduler ------------------

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Create a learning rate scheduler with warmup and decay
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ------------------ Training ------------------

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d_model = 256
    num_heads = 4
    num_layers = 3
    d_ff = 1024
    max_seq_length = 256
    dropout = 0.35

    train_ds = TranslationDataset("../data/translation2019zh/translation2019zh_train.json")
    valid_ds = TranslationDataset(
        "../data/translation2019zh/translation2019zh_valid.json",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab
    )

    train_loader = data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = data.DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)

    criterion = LabelSmoothingLoss(label_smoothing=0.1, tgt_vocab_size=tgt_vocab_size, ignore_index=train_ds.tgt_vocab['<pad>'])
    optimizer = optim.Adam(transformer.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()
    num_epochs = 30
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = total_steps // 10

    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    global_step = 0
    
    print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(num_epochs):
        total_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            output = transformer(src_batch, tgt_batch[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step}, LR: {current_lr:.6f}")

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        transformer.eval()
        val_loss = 0
        with torch.no_grad():
            for src_batch, tgt_batch in valid_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                output = transformer(src_batch, tgt_batch[:, :-1])
                val_loss += criterion(output.view(-1, output.size(-1)), tgt_batch[:, 1:].reshape(-1)).item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f"           Valid Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().half() for k, v in transformer.state_dict().items()}
            print(f"           New best model saved! (Loss: {best_val_loss:.4f})")
        transformer.train()

    torch.save(best_model_state, "transformer_lr_fixed.pth")
    print("Best model saved as transformer_lr_fixed.pth")

    final_model_state = {k: v.cpu().half() for k, v in transformer.state_dict().items()}
    torch.save(final_model_state, "transformer_lr_fixed_final.pth")
    print("Final model saved as transformer_lr_fixed_final.pth")
    return transformer

# ------------------ Run ------------------

if __name__ == "__main__":
    model = train_transformer()
